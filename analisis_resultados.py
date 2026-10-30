"""
Análisis de resultados de uno o más checkpoints (ya entrenados) frente a las
referencias simples (persistencia y media), con lo que necesita el contraste
estadístico de la memoria.

Genera en results/:
    resumen_modelos.csv      MSE global y mediana por simulación, y pérdida física
                             de las predicciones frente a la de los datos reales
    mse_por_variable.csv     MSE de cada variable (fault-free y después del fallo)
    mse_por_simulacion.csv   MSE de cada simulación (entrada del test de Wilcoxon)
    comparacion_wilcoxon.csv Wilcoxon apareado por simulación, modelo frente a la
                             referencia (primer checkpoint), con tamaño del efecto
                             r (correlación biserial por rangos) y corrección de Holm

Convenciones:
  - MSE sobre variables normalizadas con el escalador del entrenamiento.
  - Periodo "antes"/"despues": muestra <= 160 / > 160 en test (el fallo empieza
    en la muestra 160). En fault-free el periodo es "todo".
  - Diferencia = MSE(modelo) - MSE(referencia): positiva = el modelo es peor.

Uso (desde la raíz):
    python analisis_resultados.py --data_dir data_muestra \
        --checkpoints checkpoints/baseline.pt checkpoints/pinn_soft_lam1.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, wilcoxon
from torch.utils.data import DataLoader

from src.data import XMEAS_COLS, load_tep
from src.models import build_model
from src.physics import physics_terms
from src.train import unscale_torch

FAULT_ONSET = 160


def predecir(model, ds, bs: int = 8192) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for X, _ in DataLoader(ds, batch_size=bs, shuffle=False):
            out.append(model(X))
    return torch.cat(out)


def pred_persistencia(ds, x_scaler, y_scaler, n_xmv: int) -> torch.Tensor:
    """ŷ(t+1) = xmeas(t), expresado en el espacio normalizado del objetivo."""
    xs_s = torch.tensor(x_scaler.scale_, dtype=torch.float32)
    xs_m = torch.tensor(x_scaler.mean_, dtype=torch.float32)
    ys_s = torch.tensor(y_scaler.scale_, dtype=torch.float32)
    ys_m = torch.tensor(y_scaler.mean_, dtype=torch.float32)
    x_raw = ds.X * xs_s + xs_m
    return (x_raw[:, n_xmv:] - ys_m) / ys_s


def phys_medio(x_scaled: torch.Tensor, y_scaler) -> dict:
    """Pérdida física media por restricción (mezcla, purga, producto) y total."""
    t = physics_terms(unscale_torch(x_scaled, y_scaler, "cpu"))
    out = {f"phys_{k}": float((v**2).mean()) for k, v in t.items()}
    out["phys"] = sum(out.values()) / len(out)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    orden = np.argsort(p)
    aj = np.empty(m)
    previo = 0.0
    for i, idx in enumerate(orden):
        previo = max(previo, (m - i) * p[idx])
        aj[idx] = min(1.0, previo)
    return aj


def wilcoxon_par(ref: np.ndarray, otro: np.ndarray):
    d = otro - ref
    d = d[d != 0]
    n = len(d)
    if n < 1:
        return 0, 1.0, 0.0
    p = float(wilcoxon(d).pvalue)
    rangos = rankdata(np.abs(d))
    wp, wn = rangos[d > 0].sum(), rangos[d < 0].sum()
    return n, p, float((wp - wn) / (wp + wn))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_muestra")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="El primero es la referencia de la comparación")
    ap.add_argument("--faulty_runs", type=int, default=100)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    train_ds, val_ds, tests, x_sc, y_sc, n_xmv = load_tep(
        args.data_dir, faulty_runs_per_fault=args.faulty_runs or None)
    in_dim, out_dim = train_ds.X.shape[1], train_ds.y.shape[1]

    modelos = {}
    for ruta in args.checkpoints:
        m = build_model(in_dim, out_dim, "baseline", residual="_res" in Path(ruta).stem)
        m.load_state_dict(torch.load(ruta, map_location="cpu"))
        modelos[Path(ruta).stem] = m
    nombres = list(modelos) + ["persistencia", "media"]

    filas_sim, filas_var, filas_res = [], [], []
    for conjunto, ds in tests.items():
        y = ds.y
        if conjunto == "fault_free":
            periodo = np.full(len(y), "todo")
        else:
            periodo = np.where(ds.sample <= FAULT_ONSET, "antes", "despues")
        phys_real = {p: phys_medio(y[torch.from_numpy(periodo == p)], y_sc)
                     for p in np.unique(periodo)}

        for nombre in nombres:
            if nombre == "persistencia":
                pred = pred_persistencia(ds, x_sc, y_sc, n_xmv)
            elif nombre == "media":
                pred = torch.zeros_like(y)
            else:
                pred = predecir(modelos[nombre], ds)
            se = (pred - y) ** 2
            mse_fila = se.mean(dim=1).numpy()

            df = pd.DataFrame({"faultNumber": ds.fault, "simulationRun": ds.run,
                               "periodo": periodo, "mse": mse_fila})
            por_sim = df.groupby(["faultNumber", "simulationRun", "periodo"],
                                 as_index=False)["mse"].mean()
            por_sim.insert(0, "conjunto", conjunto)
            por_sim.insert(0, "modelo", nombre)
            filas_sim.append(por_sim)

            for p in np.unique(periodo):
                mask = torch.from_numpy(periodo == p)
                mse_var = se[mask].mean(dim=0).numpy()
                for v, val in zip(XMEAS_COLS, mse_var):
                    filas_var.append({"modelo": nombre, "conjunto": conjunto,
                                      "periodo": p, "variable": v, "mse": float(val)})
                sims = por_sim[por_sim.periodo == p]["mse"]
                filas_res.append({
                    "modelo": nombre, "conjunto": conjunto, "periodo": p,
                    "mse_medio": float(se[mask].mean()),
                    "mse_mediana_por_simulacion": float(sims.median()),
                    **{f"{k}_pred": v for k, v in phys_medio(pred[mask], y_sc).items()},
                    **{f"{k}_real": v for k, v in phys_real[p].items()},
                })
            print(f"[OK] {conjunto} / {nombre}", flush=True)

    sim = pd.concat(filas_sim, ignore_index=True)
    var = pd.DataFrame(filas_var)
    res = pd.DataFrame(filas_res)
    sim.to_csv(out_dir / "mse_por_simulacion.csv", index=False)
    var.to_csv(out_dir / "mse_por_variable.csv", index=False)
    res.to_csv(out_dir / "resumen_modelos.csv", index=False)

    # --- Wilcoxon apareado por simulación: cada modelo frente a la referencia
    ref = list(modelos)[0]
    comp = []
    for otro in list(modelos)[1:]:
        grupos = [("fault_free", 0, "todo")] + [
            ("faulty", f, "despues") for f in sorted(sim[sim.conjunto == "faulty"].faultNumber.unique())]
        filas = []
        for conjunto, f, per in grupos:
            sel = (sim.conjunto == conjunto) & (sim.faultNumber == f) & (sim.periodo == per)
            a = sim[sel & (sim.modelo == ref)].set_index("simulationRun")["mse"]
            b = sim[sel & (sim.modelo == otro)].set_index("simulationRun")["mse"]
            idx = a.index.intersection(b.index)
            n, p, r = wilcoxon_par(a.loc[idx].to_numpy(), b.loc[idx].to_numpy())
            filas.append({"comparacion": f"{otro} frente a {ref}", "conjunto": conjunto,
                          "faultNumber": f, "periodo": per, "n_pares": n,
                          "dif_media": float((b.loc[idx] - a.loc[idx]).mean()),
                          "dif_mediana": float((b.loc[idx] - a.loc[idx]).median()),
                          "p": p, "r_efecto": r})
        t = pd.DataFrame(filas)
        t["p_holm"] = holm(t["p"].to_numpy())
        comp.append(t)
    comp_df = pd.concat(comp, ignore_index=True) if comp else pd.DataFrame()
    comp_df.to_csv(out_dir / "comparacion_wilcoxon.csv", index=False)

    # --- Salida por pantalla
    pd.set_option("display.width", 200)
    print("\n== Resumen (MSE normalizado; phys = pérdida física de las predicciones) ==")
    print(res.round(4).to_string(index=False))

    print("\n== MSE por variable en test fault-free (1,0 = igual que predecir la media) ==")
    sel = var[var.conjunto == "fault_free"]
    tabla = sel.pivot(index="variable", columns="modelo", values="mse").loc[XMEAS_COLS]
    print(tabla[nombres].round(3).to_string())

    print("\n== MSE mediano por simulación, tras el fallo, por IDV ==")
    pos = sim[(sim.conjunto == "faulty") & (sim.periodo == "despues")]
    piv = pos.groupby(["faultNumber", "modelo"])["mse"].median().unstack("modelo")
    print(piv[nombres].round(3).to_string())

    if len(comp_df):
        print("\n== Wilcoxon (dif > 0: el modelo es peor que la referencia) ==")
        print(comp_df.round(4).to_string(index=False))
    print(f"\nGuardado en {out_dir}/")


if __name__ == "__main__":
    main()
