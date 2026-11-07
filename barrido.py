"""
Barrido de tamaño de entrenamiento: ¿a partir de cuántas simulaciones fault-free
el MLP iguala a la PINN? Entrena cada brazo con n = 5, 10, 25, ... simulaciones
y varias semillas, y registra todo en un CSV. Se puede interrumpir y repetir:
salta las combinaciones ya hechas.

La validación y el test son siempre los mismos (solo cambia cuánto se entrena),
y los subconjuntos de simulaciones están anidados (n=10 incluye las de n=5).

Brazos por defecto:
    base        MLP sin física
    base_res    MLP residual (aprende la corrección sobre la persistencia)
    pinn        PINN (mezcla + purga), lambda=1
    pinn_res    PINN residual (mezcla + purga), lambda=1

Uso (desde la raíz). Para repartirlo, cada persona usa su propio --log:
    python barrido.py --data_dir data_muestra --log results/barrido_chema.csv --sizes 5 10 25
    python barrido.py --data_dir data_muestra --log results/barrido_jorge.csv --sizes 50 100
    python barrido.py --data_dir data_muestra --log results/barrido_arturo.csv --sizes 200 0
(0 = todas las simulaciones de entrenamiento, ~400)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ARMS = {
    "base":     dict(variant="baseline", lam=0.0, residual=False),
    "base_res": dict(variant="baseline", lam=0.0, residual=True),
    "pinn":     dict(variant="pinn_soft", lam=1.0, residual=False),
    "pinn_res": dict(variant="pinn_soft", lam=1.0, residual=True),
    "pinn10_res": dict(variant="pinn_soft", lam=10.0, residual=True),
}
TERMS = "mezcla,purga"


def hecho(df, arm, n, seed) -> bool:
    if df is None or df.empty:
        return False
    a = ARMS[arm]
    m = ((df.variant == a["variant"]) & (df.residual == int(a["residual"]))
         & (df.seed == seed) & (df.train_sims.astype(str) == (str(n) if n else "all")))
    if a["variant"] != "baseline":
        m &= (df.lambda_phys == a["lam"])
    return bool(m.any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_muestra")
    ap.add_argument("--log", default="results/barrido.csv")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 25, 50, 100, 200, 0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=["base", "base_res", "pinn", "pinn_res"],
                    choices=list(ARMS))
    ap.add_argument("--faulty_runs", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=300,
                    help="Máximo de épocas (la parada anticipada corta antes casi siempre)")
    args = ap.parse_args()

    log = Path(args.log)
    total = len(args.sizes) * len(args.seeds) * len(args.arms)
    k = 0
    fallidos = []
    for n in args.sizes:
        for seed in args.seeds:
            for arm in args.arms:
                k += 1
                df = pd.read_csv(log) if log.exists() else None
                if hecho(df, arm, n, seed):
                    print(f"[{k}/{total}] ya hecho: {arm} n={n or 'all'} seed={seed}")
                    continue
                a = ARMS[arm]
                cmd = [sys.executable, "-m", "src.train", "--variant", a["variant"],
                       "--epochs", str(args.epochs),
                       "--seed", str(seed), "--train_sims", str(n),
                       "--data_dir", args.data_dir, "--faulty_runs", str(args.faulty_runs),
                       "--log_csv", str(log), "--no_bar"]
                if a["variant"] != "baseline":
                    cmd += ["--lambda_phys", str(a["lam"]), "--terms", TERMS]
                if a["residual"]:
                    cmd += ["--residual"]
                print(f"[{k}/{total}] {arm} n={n or 'all'} seed={seed}", flush=True)
                for intento in (1, 2, 3):
                    if subprocess.run(cmd).returncode == 0:
                        break
                    print(f"  fallo (intento {intento}/3), se reintenta", flush=True)
                else:
                    fallidos.append((arm, n, seed))
                    print("  se omite esta combinación; vuelve a lanzar el comando para repetirla", flush=True)


    if fallidos:
        print("\nCombinaciones sin completar:", fallidos)


if __name__ == "__main__":
    main()
