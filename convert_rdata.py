"""
Convierte los .RData del dataset de Rieth et al. (2017) a CSV.

Uso (desde la raíz del repo, con el venv activado):
    pip install pyreadr
    python convert_rdata.py                 # convierte los 4 ficheros de data/
    python convert_rdata.py --data_dir data --skip_existing

Cada .RData contiene un único data.frame; se guarda como CSV con el mismo
nombre base (p. ej. TEP_FaultFree_Training.RData -> TEP_FaultFree_Training.csv),
que es lo que espera src/data.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyreadr

EXPECTED = [
    "TEP_FaultFree_Training",
    "TEP_FaultFree_Testing",
    "TEP_Faulty_Training",
    "TEP_Faulty_Testing",
]
EXPECTED_COLUMNS = {"faultNumber", "simulationRun", "sample"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()
    data_dir = Path(args.data_dir)

    for stem in EXPECTED:
        src = data_dir / f"{stem}.RData"
        dst = data_dir / f"{stem}.csv"
        if not src.exists():
            print(f"[AVISO] No existe {src}, se omite.")
            continue
        if args.skip_existing and dst.exists():
            print(f"[OK] {dst.name} ya existe, se omite.")
            continue

        result = pyreadr.read_r(str(src))
        name, df = next(iter(result.items()))  # único objeto del fichero
        missing = EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{src.name}: faltan columnas {missing}")

        for col in ("faultNumber", "simulationRun", "sample"):
            df[col] = df[col].astype("int32")
        float_cols = df.columns.difference(["faultNumber", "simulationRun", "sample"])
        df[float_cols] = df[float_cols].astype("float32")

        df.to_csv(dst, index=False)
        print(f"[OK] {src.name} (objeto '{name}') -> {dst.name}: "
              f"{len(df):,} filas, {df.shape[1]} columnas, "
              f"{df['simulationRun'].nunique()} simulaciones")


if __name__ == "__main__":
    main()
