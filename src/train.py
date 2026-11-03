"""
Entrenamiento de las distintas variantes (baseline / pinn_soft / pinn_alt).

Uso:
    python -m src.train --variant baseline --epochs 100
    python -m src.train --variant pinn_soft --epochs 100 --lambda_phys 0.1
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import load_tep
from src.models import build_model
from src.physics import TERMINOS, physics_loss


def set_seed(seed: int) -> None:
    """Fija las semillas para que un entrenamiento sea reproducible por los
    otros dos integrantes del equipo (mismo split, misma inicialización)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unscale_torch(x_scaled: torch.Tensor, scaler: StandardScaler, device) -> torch.Tensor:
    """Des-escala un tensor normalizado con un StandardScaler ya ajustado,
    usando únicamente operaciones de PyTorch. Esto es importante: usar
    `scaler.inverse_transform` (numpy) rompería el grafo de autodiferenciación
    y el término físico dejaría de poder retropropagar gradiente hacia la
    red — la PINN entrenaría exactamente igual que el baseline sin que nadie
    se diera cuenta, porque el error no lanza ninguna excepción.
    """
    mean = torch.tensor(scaler.mean_, dtype=x_scaled.dtype, device=device)
    scale = torch.tensor(scaler.scale_, dtype=x_scaled.dtype, device=device)
    return x_scaled * scale + mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--variant", type=str, required=True,
                    choices=["baseline", "pinn_soft", "pinn_alt"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_phys", type=float, default=0.0,
                    help="Peso del término físico (ignorado si variant=baseline)")
    p.add_argument("--patience", type=int, default=15,
                    help="Épocas sin mejora en val antes de parar. 0 = sin early stopping")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42,
                    help="Semilla de la inicialización y del barajado de lotes")
    p.add_argument("--split_seed", type=int, default=42,
                    help="Semilla de la partición entrenamiento/validación (fija para todos)")
    p.add_argument("--train_sims", type=int, default=0,
                    help="Nº de simulaciones fault-free para entrenar (0 = todas, ~400)")
    p.add_argument("--terms", type=str, default="mezcla,purga,producto",
                    help="Restricciones físicas activas (separadas por comas): mezcla, purga, producto")
    p.add_argument("--residual", action="store_true",
                    help="El modelo aprende la corrección sobre la persistencia (ver models.py)")
    p.add_argument("--no_bar", action="store_true",
                    help="Sin barra de progreso (evita errores de consola en ejecuciones largas)")
    p.add_argument("--faulty_runs", type=int, default=100,
                    help="Simulaciones por fallo en el test faulty (0 = todas, ~9,6M filas)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_csv", type=str, default="results/log.csv")
    return p.parse_args()


def evaluate_mse(model, loader, device) -> float:
    model.eval()
    total_se, total_n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_se += torch.sum((pred - y) ** 2).item()
            total_n += y.numel()
    return total_se / total_n


def metricas_faulty(model, ds, device, onset: int = 160):
    """MSE medio tras el inicio del fallo y mediana (por simulación) de ese MSE.
    La media global está dominada por unos pocos fallos graves; la mediana por
    simulación es más estable."""
    import pandas as pd
    model.eval()
    mse = []
    with torch.no_grad():
        for X, y in DataLoader(ds, batch_size=8192, shuffle=False):
            mse.append(((model(X.to(device)) - y.to(device)) ** 2).mean(dim=1).cpu())
    mse = torch.cat(mse).numpy()
    df = pd.DataFrame({"f": ds.fault, "r": ds.run, "s": ds.sample, "mse": mse})
    df = df[df.s > onset]
    por_sim = df.groupby(["f", "r"])["mse"].mean()
    return float(df.mse.mean()), float(por_sim.median())


def evaluate_phys(model, loader, y_scaler, device) -> float:
    """Pérdida física media de las predicciones del modelo (sin gradiente).
    Sirve también para el baseline: indica cuánto cumple ya el balance sin
    habérselo impuesto."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, _ in loader:
            pred = model(X.to(device))
            pred_phys = unscale_torch(pred, y_scaler, device)
            total += physics_loss(pred_phys).item() * len(X)
            n += len(X)
    return total / n


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_splits, x_scaler, y_scaler, n_xmv = load_tep(
        args.data_dir, val_fraction=args.val_fraction, seed=args.split_seed,
        faulty_runs_per_fault=args.faulty_runs or None,
        train_sims=args.train_sims or None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        for name, ds in test_splits.items()
    }

    input_dim = train_ds.X.shape[1]
    output_dim = train_ds.y.shape[1]
    model = build_model(input_dim, output_dim, variant=args.variant,
                        residual=args.residual, x_scaler=x_scaler, y_scaler=y_scaler).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    use_physics = args.variant != "baseline" and args.lambda_phys > 0

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Un checkpoint por configuración, para no sobrescribir entre ejecuciones.
    stem = "baseline" if args.variant == "baseline" else f"{args.variant}_lam{args.lambda_phys:g}"
    terms = tuple(t.strip() for t in args.terms.split(",") if t.strip())
    if set(terms) - set(TERMINOS):
        raise SystemExit(f"--terms desconocido: {terms}. Opciones: {TERMINOS}")
    if args.variant != "baseline" and terms != TERMINOS:
        stem += "_" + "-".join(terms)
    if args.residual:
        stem += "_res"
    if args.train_sims:
        stem += f"_n{args.train_sims}"
    if args.seed != 42:
        stem += f"_s{args.seed}"
    ckpt_path = ckpt_dir / f"{stem}.pt"

    best_val_mse = float("inf")
    best_val_phys = float("nan")
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{args.variant}] epoch {epoch}/{args.epochs}",
                    disable=args.no_bar)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = mse_loss(pred, y)
            if use_physics:
                # pred está en espacio normalizado (misma escala que y). Para que
                # el residuo físico tenga sentido, se des-escala antes de pasarlo
                # a physics_loss, preservando el grafo de autodiferenciación.
                pred_phys = unscale_torch(pred, y_scaler, device)
                loss = loss + args.lambda_phys * physics_loss(pred_phys, terms)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        val_mse = evaluate_mse(model, val_loader, device)
        val_phys = evaluate_phys(model, val_loader, y_scaler, device)
        print(f"[{args.variant}] epoch {epoch}: val_mse={val_mse:.6f}  val_phys={val_phys:.5f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_phys = val_phys
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping en epoch {epoch} "
                    f"(sin mejora en val desde epoch {best_epoch}, "
                    f"mejor val_mse={best_val_mse:.6f})"
                )
                break

    # Recarga el MEJOR checkpoint (según val), no el de la última época, para
    # evaluar en test — evita reportar un modelo que ya había empezado a
    # sobreajustar.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    mse_ff = evaluate_mse(model, test_loaders["fault_free"], device)
    mse_f = evaluate_mse(model, test_loaders["faulty"], device)
    print(f"MSE test (fault-free): {mse_ff:.5f}")
    print(f"MSE test (faulty):     {mse_f:.5f}")
    mse_fd, mse_fd_med = metricas_faulty(model, test_splits["faulty"], device)
    print(f"MSE faulty tras el fallo: media {mse_fd:.5f} | mediana por simulación {mse_fd_med:.5f}")
    print(f"Modelo guardado en {ckpt_path} (mejor val_mse={best_val_mse:.6f}, epoch {best_epoch})")

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cabecera = [
        "variant", "epochs_run", "best_epoch", "batch_size", "lr", "lambda_phys",
        "seed", "mse_val_best", "phys_val_best", "mse_fault_free", "mse_faulty", "checkpoint",
        "train_sims", "residual", "terms", "split_seed",
        "mse_faulty_despues", "mse_faulty_despues_mediana",
    ]
    if log_path.exists():
        with open(log_path, newline="") as f:
            primera = next(csv.reader(f), [])
        if primera != cabecera:   # formato antiguo: se conserva aparte
            antiguo = log_path.with_name(log_path.stem + "_antiguo.csv")
            log_path.replace(antiguo)
            print(f"Formato de log antiguo guardado en {antiguo}")
    new_file = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(cabecera)
        writer.writerow([
            args.variant, epoch, best_epoch, args.batch_size, args.lr, args.lambda_phys,
            args.seed, best_val_mse, best_val_phys, mse_ff, mse_f, str(ckpt_path),
            args.train_sims or "all", int(args.residual),
            "-".join(terms) if args.variant != "baseline" and args.lambda_phys > 0 else "",
            args.split_seed, mse_fd, mse_fd_med,
        ])
    print(f"Resultado registrado en {log_path}")


if __name__ == "__main__":
    main()
