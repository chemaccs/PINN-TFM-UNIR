"""
Carga y preprocesado del dataset Tennessee Eastman Process (TEP).

Ajusta COLUMN_MAP si tu conversión del dataset usa otros nombres de columna
(ver data/README.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

XMEAS_COLS = [f"xmeas_{i}" for i in range(1, 42)]   # 41 variables medidas
XMV_COLS = [f"xmv_{i}" for i in range(1, 12)]        # 11 variables manipuladas

COLUMN_MAP = {
    "fault": "faultNumber",
    "run": "simulationRun",
    "time": "sample",
}


@dataclass
class TEPSplit:
    X: np.ndarray          # variables de entrada (t): xmv + xmeas en t
    y: np.ndarray          # variables objetivo: xmeas en t+1 (predicción un paso adelante)
    fault: np.ndarray      # tipo de fallo de cada fila (0 = fault-free)
    run: np.ndarray        # id de simulación de cada fila
    sample: np.ndarray     # índice temporal (columna 'sample') del instante t de cada fila


def _load_csv(path: Path, max_run: int | None = None) -> pd.DataFrame:
    """Lee un CSV del TEP en float32 (la mitad de memoria que el float64 por
    defecto). Si max_run no es None, conserva solo las simulaciones con
    simulationRun <= max_run, leyendo por bloques para no cargar nunca el
    fichero entero (el test faulty tiene 9,6 millones de filas)."""
    dtypes = {c: "float32" for c in XMEAS_COLS + XMV_COLS}
    dtypes.update({"faultNumber": "int32", "simulationRun": "int32", "sample": "int32"})
    if max_run is None:
        df = pd.read_csv(path, dtype=dtypes)
    else:
        parts = []
        for chunk in pd.read_csv(path, dtype=dtypes, chunksize=1_000_000):
            parts.append(chunk[chunk["simulationRun"] <= max_run])
        df = pd.concat(parts, ignore_index=True)
    missing = [c for c in XMEAS_COLS + XMV_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas esperadas en {path.name}: {missing[:5]}... "
            "Revisa COLUMN_MAP / nombres de columna en data/README.md"
        )
    return df


def _build_sequences(df: pd.DataFrame) -> TEPSplit:
    """Construye pares (estado en t) -> (estado en t+1) dentro de cada simulationRun,
    para entrenar un modelo de predicción un paso adelante (one-step-ahead)."""
    X_list, y_list, fault_list, run_list, sample_list = [], [], [], [], []

    run_col = COLUMN_MAP["run"]
    fault_col = COLUMN_MAP["fault"]

    # IMPORTANTE: en los ficheros faulty, simulationRun se repite para cada tipo
    # de fallo (1..500 por cada IDV). Agrupar solo por simulationRun mezclaría
    # simulaciones de fallos distintos y crearía pares (t, t+1) falsos. Una
    # simulación única es la pareja (faultNumber, simulationRun).
    for (_, run_id), g in df.groupby([fault_col, run_col], sort=True):
        g = g.sort_values(COLUMN_MAP["time"], kind="stable")
        feats = g[XMV_COLS + XMEAS_COLS].to_numpy(dtype=np.float32)
        if len(feats) < 2:
            continue
        X_list.append(feats[:-1])
        y_list.append(feats[1:, len(XMV_COLS):])  # objetivo: xmeas en t+1
        fault_list.append(g[fault_col].to_numpy()[:-1])
        run_list.append(np.full(len(feats) - 1, run_id))
        sample_list.append(g[COLUMN_MAP["time"]].to_numpy()[:-1])

    return TEPSplit(
        X=np.concatenate(X_list),
        y=np.concatenate(y_list),
        fault=np.concatenate(fault_list),
        run=np.concatenate(run_list),
        sample=np.concatenate(sample_list),
    )


class TEPDataset(Dataset):
    """Dataset de PyTorch sobre un TEPSplit ya escalado."""

    def __init__(self, split: TEPSplit):
        self.X = torch.from_numpy(split.X).float()
        self.y = torch.from_numpy(split.y).float()
        self.fault = split.fault
        self.run = split.run
        self.sample = split.sample

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_tep(
    data_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    faulty_runs_per_fault: int | None = 100,
    train_sims: int | None = None,
):
    """Carga los ficheros del TEP, construye secuencias one-step-ahead y escala.

    Separa un conjunto de VALIDACIÓN a partir de un subconjunto de las
    simulaciones (simulationRun) del fichero de entrenamiento fault-free,
    nunca por filas sueltas, para evitar fuga de información entre muestras
    de una misma simulación (tal como se documenta en la memoria, apdo. 3.3).
    El escalador se ajusta únicamente sobre la partición de entrenamiento
    resultante (sin incluir las simulaciones de validación).

    Devuelve: (train_ds, val_ds, test_splits, x_scaler, y_scaler, n_xmv)
    donde test_splits es un dict {"fault_free": TEPDataset, "faulty": TEPDataset}
    """
    data_dir = Path(data_dir)

    df_train_full = _load_csv(data_dir / "TEP_FaultFree_Training.csv")
    df_test_ff = _load_csv(data_dir / "TEP_FaultFree_Testing.csv")
    # El test faulty completo (500 simulaciones x 20 fallos) no cabe en la RAM de
    # un portátil. Por defecto se usan las primeras 100 simulaciones de cada
    # fallo (2 millones de filas); None = todas.
    df_test_f = _load_csv(data_dir / "TEP_Faulty_Testing.csv", max_run=faulty_runs_per_fault)

    run_col = COLUMN_MAP["run"]
    run_ids = df_train_full[run_col].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(run_ids)
    n_val = max(1, int(round(len(run_ids) * val_fraction)))
    val_runs = set(run_ids[:n_val])
    pool = run_ids[n_val:]
    # train_sims: usa solo las n primeras simulaciones del orden fijado por `seed`
    # (subconjuntos ANIDADOS: las de n=10 incluyen las de n=5). Validación y test
    # no cambian, así que los resultados son comparables entre tamaños.
    train_runs = set(pool if not train_sims else pool[:train_sims])

    df_train = df_train_full[df_train_full[run_col].isin(train_runs)]
    df_val = df_train_full[df_train_full[run_col].isin(val_runs)]

    train_split = _build_sequences(df_train)
    val_split = _build_sequences(df_val)
    test_ff_split = _build_sequences(df_test_ff)
    test_f_split = _build_sequences(df_test_f)

    n_xmv = len(XMV_COLS)
    x_scaler = StandardScaler().fit(train_split.X)
    y_scaler = StandardScaler().fit(train_split.y)

    def _scale(split: TEPSplit) -> TEPSplit:
        return TEPSplit(
            X=x_scaler.transform(split.X).astype(np.float32),
            y=y_scaler.transform(split.y).astype(np.float32),
            fault=split.fault,
            run=split.run,
            sample=split.sample,
        )

    train_ds = TEPDataset(_scale(train_split))
    val_ds = TEPDataset(_scale(val_split))
    test_splits = {
        "fault_free": TEPDataset(_scale(test_ff_split)),
        "faulty": TEPDataset(_scale(test_f_split)),
    }

    return train_ds, val_ds, test_splits, x_scaler, y_scaler, n_xmv
