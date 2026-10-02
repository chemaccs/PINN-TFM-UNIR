"""
Entrenamiento de las distintas variantes (baseline / pinn_soft / pinn_alt).

Uso:
    python -m src.train --variant baseline --epochs 100
    python -m src.train --variant pinn_soft --epochs 100 --lambda_phys 0.1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import load_tep
from src.models import build_model
from src.physics import physics_loss


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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_splits, x_scaler, y_scaler, n_xmv = load_tep(args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        for name, ds in test_splits.items()
    }

    input_dim = train_ds.X.shape[1]
    output_dim = train_ds.y.shape[1]
    model = build_model(input_dim, output_dim, variant=args.variant).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    use_physics = args.variant != "baseline" and args.lambda_phys > 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{args.variant}] epoch {epoch}/{args.epochs}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = mse_loss(pred, y)
            if use_physics:
                # Nota: physics_loss espera xmeas en unidades físicas; si trabajáis
                # en espacio escalado, des-escalar `pred` antes de pasarlo, o
                # documentar que el residuo es relativo (ver src/physics.py).
                loss = loss + args.lambda_phys * physics_loss(pred)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    # Evaluación final y logging
    mse_ff = evaluate_mse(model, test_loaders["fault_free"], device)
    mse_f = evaluate_mse(model, test_loaders["faulty"], device)
    print(f"MSE test (fault-free): {mse_ff:.5f}")
    print(f"MSE test (faulty):     {mse_f:.5f}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.variant}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Modelo guardado en {ckpt_path}")

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "variant", "epochs", "batch_size", "lr", "lambda_phys",
                "mse_fault_free", "mse_faulty", "checkpoint",
            ])
        writer.writerow([
            args.variant, args.epochs, args.batch_size, args.lr, args.lambda_phys,
            mse_ff, mse_f, str(ckpt_path),
        ])
    print(f"Resultado registrado en {log_path}")


if __name__ == "__main__":
    main()
