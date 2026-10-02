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


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    X_list, y_list, fault_list, run_list = [], [], [], []

    run_col = COLUMN_MAP["run"]
    fault_col = COLUMN_MAP["fault"]

    for run_id, g in df.groupby(run_col):
        g = g.sort_values(COLUMN_MAP["time"])
        feats = g[XMV_COLS + XMEAS_COLS].to_numpy(dtype=np.float32)
        if len(feats) < 2:
            continue
        X_list.append(feats[:-1])
        y_list.append(feats[1:, len(XMV_COLS):])  # objetivo: xmeas en t+1
        fault_list.append(g[fault_col].to_numpy()[:-1])
        run_list.append(np.full(len(feats) - 1, run_id))

    return TEPSplit(
        X=np.concatenate(X_list),
        y=np.concatenate(y_list),
        fault=np.concatenate(fault_list),
        run=np.concatenate(run_list),
    )


class TEPDataset(Dataset):
    """Dataset de PyTorch sobre un TEPSplit ya escalado."""

    def __init__(self, split: TEPSplit):
        self.X = torch.from_numpy(split.X).float()
        self.y = torch.from_numpy(split.y).float()
        self.fault = split.fault
        self.run = split.run

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_tep(data_dir: str | Path, fit_scaler_on: str = "fault_free_train"):
    """Carga los 4 ficheros del TEP, construye secuencias one-step-ahead y escala.

    Devuelve: (train_ds, test_splits, x_scaler, y_scaler)
    donde test_splits es un dict {"fault_free": TEPSplit, "faulty": TEPSplit}
    """
    data_dir = Path(data_dir)

    df_train = _load_csv(data_dir / "TEP_FaultFree_Training.csv")
    df_test_ff = _load_csv(data_dir / "TEP_FaultFree_Testing.csv")
    df_test_f = _load_csv(data_dir / "TEP_Faulty_Testing.csv")

    train_split = _build_sequences(df_train)
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
        )

    train_ds = TEPDataset(_scale(train_split))
    test_splits = {
        "fault_free": TEPDataset(_scale(test_ff_split)),
        "faulty": TEPDataset(_scale(test_f_split)),
    }

    return train_ds, test_splits, x_scaler, y_scaler, n_xmv
