"""
Genera una carpeta REDUCIDA y EQUILIBRADA a partir de los cuatro CSV del TEP.

- Se reducen simulaciones enteras (nunca filas sueltas).
- En los ficheros faulty se conservan las MISMAS n primeras simulaciones de
  cada fallo (equilibrado entre los 20 IDV). Las simulaciones son
  independientes, así que las n primeras son una muestra válida.
- Los ficheros faulty (5 y 9,6 millones de filas) se leen por bloques y se
  escriben en disco sobre la marcha: la RAM necesaria es de ~250 MB.
- Al final se escribe results/descripcion_muestra.csv con el tamaño de cada
  fichero resultante (tabla útil para la memoria, apartado de datos).

Uso (desde la raíz del repo):
    python generar_muestra.py
    python generar_muestra.py --n_ff_train 100 --n_ff_test 100 --n_faulty_train 100 --n_faulty_test 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import XMEAS_COLS, XMV_COLS

DTYPES = {c: "float32" for c in XMV_COLS + XMEAS_COLS}
DTYPES.update({"faultNumber": "int32", "simulationRun": "int32", "sample": "int32"})
CHUNK = 1_000_000


def reducir(src: Path, dst: Path, n_sims: int, aleatorio: bool, seed: int) -> dict:
    """Escribe en dst las n_sims simulaciones elegidas, leyendo src por bloques.
    aleatorio=True: elige al azar entre las que hay (fault-free, un solo
    fallo). aleatorio=False: las n primeras (faulty, igual para cada fallo)."""
    t0 = time.time()
    chosen = None
    first = True
    filas = 0
    fallos = set()
    sims = set()
    for chunk in pd.read_csv(src, dtype=DTYPES, chunksize=CHUNK):
        if chosen is None:
            if aleatorio:
                # Fault-free: un solo fallo, el fichero cabe en el primer bloque
                # (500 simulaciones x 960 muestras < 1 millón de filas).
                ids = np.sort(chunk["simulationRun"].unique())
                n = min(n_sims, len(ids))
                ids = np.random.RandomState(seed).choice(ids, size=n, replace=False)
                chosen = set(int(i) for i in ids)
            else:
                # Faulty: los números de simulación van de 1 a 500 (documentación
                # del dataset) y el fichero NO está ordenado por fallo, así que
                # el primer bloque no contiene todas las simulaciones. Se fijan
                # directamente las n primeras: 1..n para todos los fallos.
                chosen = set(range(1, n_sims + 1))
        sub = chunk[chunk["simulationRun"].isin(chosen)]
        if len(sub):
            sub.to_csv(dst, mode="w" if first else "a", header=first, index=False)
            first = False
            filas += len(sub)
            fallos.update(sub["faultNumber"].unique().tolist())
            sims.update(zip(sub["faultNumber"].tolist(), sub["simulationRun"].tolist()))
    por_fallo = pd.Series([f for f, _ in sims]).value_counts()
    return {
        "fichero": dst.name,
        "filas": filas,
        "fallos": len(fallos),
        "sim_por_fallo_min": int(por_fallo.min()),
        "sim_por_fallo_max": int(por_fallo.max()),
        "muestras_por_sim": int(round(filas / max(1, len(sims)))),
        "MB_en_disco": round(dst.stat().st_size / 1e6, 1),
        "segundos": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="data_muestra")
    p.add_argument("--n_ff_train", type=int, default=100)
    p.add_argument("--n_ff_test", type=int, default=100)
    p.add_argument("--n_faulty_train", type=int, default=100)
    p.add_argument("--n_faulty_test", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--solo", nargs="+", default=None,
                   help="Regenerar solo estos ficheros, p. ej.: --solo TEP_Faulty_Testing")
    args = p.parse_args()

    src, dst = Path(args.data_dir), Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    trabajos = [
        ("TEP_FaultFree_Training.csv", args.n_ff_train, True),
        ("TEP_FaultFree_Testing.csv", args.n_ff_test, True),
        ("TEP_Faulty_Training.csv", args.n_faulty_train, False),
        ("TEP_Faulty_Testing.csv", args.n_faulty_test, False),
    ]
    if args.solo:
        trabajos = [t for t in trabajos
                    if any(t[0].startswith(s) or s in t[0] for s in args.solo)]
    filas = []
    for nombre, n, aleatorio in trabajos:
        if not (src / nombre).exists():
            print(f"[AVISO] No existe {src / nombre}, se omite.")
            continue
        print(f"Reduciendo {nombre} ({n} simulaciones)...", flush=True)
        info = reducir(src / nombre, dst / nombre, n, aleatorio, args.seed)
        filas.append(info)
        print(f"  [OK] {info['filas']:,} filas, {info['MB_en_disco']} MB, "
              f"{info['segundos']} s; simulaciones por fallo: "
              f"{info['sim_por_fallo_min']}-{info['sim_por_fallo_max']}")

    out = pd.DataFrame(filas)
    Path("results").mkdir(exist_ok=True)
    previo = Path("results/descripcion_muestra.csv")
    if args.solo and previo.exists():  # conserva las filas de los ficheros no regenerados
        antes = pd.read_csv(previo)
        antes = antes[~antes["fichero"].isin(out["fichero"])]
        out = pd.concat([antes, out], ignore_index=True)
    out.to_csv("results/descripcion_muestra.csv", index=False)
    print("\nResumen de la muestra generada:")
    print(out.to_string(index=False))
    print("\nGuardado en results/descripcion_muestra.csv")


if __name__ == "__main__":
    main()
