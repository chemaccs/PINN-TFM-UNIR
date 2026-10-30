"""
Estudio del tamaño de muestra de entrenamiento (acuerdo con Ivón, 26/10).

Pregunta: ¿cuántas simulaciones fault-free hacen falta para conservar la
información importante del conjunto completo? Se muestrean SIMULACIONES
ENTERAS (nunca filas sueltas: cada par (t, t+1) necesita dos muestras
consecutivas de la misma simulación).

Para cada tamaño n y varias extracciones aleatorias se compara el subconjunto
con el total en:
  - media (en desviaciones típicas del total)
  - desviación típica (cociente)
  - matriz de correlaciones (máxima diferencia absoluta)

Además puede generar una carpeta reducida y EQUILIBRADA (mismas simulaciones
para cada fallo) con los mismos nombres de fichero que `data/`, de modo que
basta `--data_dir data_muestra` en train/evaluate.

Uso (desde la raíz del repo):
    python estudio_muestra.py                       # solo el estudio
    python estudio_muestra.py --generar --n_train 100 --n_test 100 --n_faulty 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import XMEAS_COLS, XMV_COLS

VARS = XMV_COLS + XMEAS_COLS
DTYPES = {c: "float32" for c in VARS}
DTYPES.update({"faultNumber": "int32", "simulationRun": "int32", "sample": "int32"})


def estudio(df: pd.DataFrame, tamanos, n_draws: int, seed: int,
            tol_media=0.05, tol_std=0.05, tol_corr=0.05) -> pd.DataFrame:
    runs = df["simulationRun"].unique()
    full_mean = df[VARS].mean()
    full_std = df[VARS].std().replace(0, np.nan)
    full_corr = df[VARS].corr().to_numpy()
    full_corr = np.nan_to_num(full_corr)

    rows = []
    rng = np.random.RandomState(seed)
    for n in tamanos:
        if n > len(runs):
            continue
        for d in range(n_draws):
            chosen = rng.choice(runs, size=n, replace=False)
            sub = df[df["simulationRun"].isin(chosen)]
            dmean = ((sub[VARS].mean() - full_mean).abs() / full_std).max()
            ratio = (sub[VARS].std() / full_std).dropna()
            dstd = (ratio - 1).abs().max()
            sub_corr = np.nan_to_num(sub[VARS].corr().to_numpy())
            dcorr = np.abs(sub_corr - full_corr).max()
            rows.append({"n_sim": n, "extraccion": d, "max_dif_media_en_std": dmean,
                         "max_dif_std_relativa": dstd, "max_dif_correlacion": dcorr})
    res = pd.DataFrame(rows)
    res["cumple"] = ((res.max_dif_media_en_std <= tol_media)
                     & (res.max_dif_std_relativa <= tol_std)
                     & (res.max_dif_correlacion <= tol_corr))
    return res


def generar(data_dir: Path, out_dir: Path, n_train: int, n_test: int,
            n_faulty: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    def _sub(src, n):
        df = pd.read_csv(data_dir / src, dtype=DTYPES)
        runs = df["simulationRun"].unique()
        chosen = rng.choice(runs, size=min(n, len(runs)), replace=False)
        df = df[df["simulationRun"].isin(chosen)]
        df.to_csv(out_dir / src, index=False)
        print(f"[OK] {src}: {df['simulationRun'].nunique()} simulaciones, {len(df):,} filas")

    _sub("TEP_FaultFree_Training.csv", n_train)
    _sub("TEP_FaultFree_Testing.csv", n_test)

    # Faulty: el MISMO conjunto de simulaciones para cada fallo (equilibrado),
    # leyendo por bloques para no cargar 9,6 millones de filas.
    parts = []
    chosen = None
    for chunk in pd.read_csv(data_dir / "TEP_Faulty_Testing.csv", dtype=DTYPES,
                             chunksize=1_000_000):
        if chosen is None:  # el primer bloque contiene todas las simulaciones del primer fallo
            # Las simulaciones son independientes (semillas distintas), así que
            # las n primeras son una muestra tan válida como cualquier otra y
            # permiten filtrar luego con --faulty_runs sin perder ninguna.
            ids = np.sort(chunk["simulationRun"].unique())
            chosen = set(ids[: min(n_faulty, len(ids))].tolist())
        parts.append(chunk[chunk["simulationRun"].isin(chosen)])
    df_f = pd.concat(parts, ignore_index=True)
    df_f.to_csv(out_dir / "TEP_Faulty_Testing.csv", index=False)
    per_fault = df_f.groupby("faultNumber")["simulationRun"].nunique()
    print(f"[OK] TEP_Faulty_Testing.csv: {len(df_f):,} filas; "
          f"simulaciones por fallo: min {per_fault.min()}, max {per_fault.max()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_csv", default="results/estudio_muestra.csv")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--tamanos", type=int, nargs="+", default=[5, 10, 25, 50, 100, 200, 300, 400],
                    help="Número de simulaciones a probar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generar", action="store_true",
                   help="Genera la carpeta reducida y equilibrada")
    p.add_argument("--out_dir", default="data_muestra")
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--n_faulty", type=int, default=50)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "TEP_FaultFree_Training.csv", dtype=DTYPES)
    n_total = df["simulationRun"].nunique()
    tamanos = [n for n in sorted(args.tamanos) if n < n_total]

    res = estudio(df, tamanos, args.draws, args.seed)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)

    resumen = res.groupby("n_sim").agg(
        dif_media_max=("max_dif_media_en_std", "max"),
        dif_std_max=("max_dif_std_relativa", "max"),
        dif_corr_max=("max_dif_correlacion", "max"),
        extracciones_ok=("cumple", "sum"),
    )
    print("\nPeor caso entre extracciones, por número de simulaciones")
    print("(tolerancias: media 0,05 std; std 5 %; correlación 0,05)\n")
    print(resumen.round(3).to_string())
    ok = resumen[resumen.extracciones_ok == args.draws]
    if len(ok):
        print(f"\nPrimer tamaño que cumple en todas las extracciones: {ok.index[0]} simulaciones")
    else:
        print("\nNingún tamaño cumple todas las tolerancias; revisad las tolerancias o usad todo.")
    print(f"Detalle guardado en {args.out_csv}")

    if args.generar:
        generar(data_dir, Path(args.out_dir), args.n_train, args.n_test,
                args.n_faulty, args.seed)


if __name__ == "__main__":
    main()
