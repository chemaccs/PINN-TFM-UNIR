"""
Evalúa un checkpoint ya entrenado sobre los dos conjuntos de test
(fault-free y faulty), desglosando además el MSE por tipo de fallo.

Uso:
    python -m src.evaluate --checkpoint checkpoints/pinn_soft.pt --variant pinn_soft
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_tep
from src.models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "pinn_soft", "pinn_alt"])
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def mse_by_fault(model, dataset, device, batch_size) -> dict[int, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    results = {}
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            se = ((pred - y) ** 2).mean(dim=1).cpu().numpy()
            for fault_id, err in zip(dataset.fault[: len(se)], se):
                results.setdefault(int(fault_id), []).append(err)
    return {k: float(np.mean(v)) for k, v in sorted(results.items())}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_splits, *_ = load_tep(args.data_dir)
    input_dim = train_ds.X.shape[1]
    output_dim = train_ds.y.shape[1]

    model = build_model(input_dim, output_dim, variant=args.variant).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("MSE por tipo de fallo (0 = fault-free):")
    for name, ds in test_splits.items():
        print(f"\n-- conjunto: {name} --")
        for fault_id, mse in mse_by_fault(model, ds, device, args.batch_size).items():
            print(f"  IDV_{fault_id}: {mse:.5f}")


if __name__ == "__main__":
    main()
