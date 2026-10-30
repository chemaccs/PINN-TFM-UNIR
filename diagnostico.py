"""
Diagnóstico previo a entrenar (pasos 2 y 3 del plan tras la reunión con Ivón).

1) Referencias de error por variable: predictor de la media, persistencia
   (copiar el valor anterior) y fracción de valores repetidos entre muestras
   consecutivas (detecta los analizadores de composición, XMEAS 23-41).
2) Residuos de ecuaciones físicas candidatas, calculados sobre DATOS REALES:
   una ecuación solo es utilizable como restricción si su residuo es cercano a
   cero en fault-free y cambia con los fallos.

Salidas (carpeta results/):
    diag_variables.csv   referencias por variable
    diag_residuos.csv    residuo de cada candidata, por fallo y por periodo
                         (antes / después de la muestra 160, en que empiezan los fallos)

Uso (desde la raíz del repo):
    python diagnostico.py
    python diagnostico.py --data_dir data_muestra --faulty_runs 50
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
FAULT_ONSET = 160  # muestra a partir de la cual actúa el fallo en los ficheros de test

# Volumen molar a condiciones estándar (m3/kmol) y pesos moleculares de D y E
# (Downs y Vogel, 1993; A VERIFICAR con Ivón antes de usarlos en la memoria).
VM = 22.414
MW_D, MW_E = 32.0, 46.0

# Ecuaciones candidatas: cada una devuelve un residuo que debería ser ~0.
CANDIDATAS = {
    # El residuo que usa hoy physics.py (mezcla kscmh y kg/h)
    "global_actual": lambda d: d.xmeas_1 + d.xmeas_2 + d.xmeas_3 + d.xmeas_4
                               - d.xmeas_10 - d.xmeas_17,
    # Mezcla en la entrada del reactor, solo caudales volumétricos (kscmh)
    "alim_reactor_volumetrico": lambda d: d.xmeas_6 - (d.xmeas_1 + d.xmeas_4 + d.xmeas_5),
    # Igual, convirtiendo D y E (kg/h) a kscmh con pesos moleculares
    "alim_reactor_con_D_E": lambda d: d.xmeas_6 - (
        d.xmeas_1 + d.xmeas_4 + d.xmeas_5
        + VM / 1000.0 * (d.xmeas_2 / MW_D + d.xmeas_3 / MW_E)),
    # Cierre de composición: las fracciones molares de cada corriente suman 100 %
    # (analizadores: 23-28 alimentación al reactor A-F, 29-36 purga A-H,
    # 37-41 producto D-H; a verificar con Downs y Vogel).
    "cierre_alimentacion_reactor": lambda d: sum(d[f"xmeas_{i}"] for i in range(23, 29)) - 100.0,
    "cierre_purga": lambda d: sum(d[f"xmeas_{i}"] for i in range(29, 37)) - 100.0,
    "cierre_producto": lambda d: sum(d[f"xmeas_{i}"] for i in range(37, 42)) - 100.0,
}


def _cargar_fault_free(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=DTYPES)


def referencias_por_variable(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """MSE de persistencia y de la media por variable, en unidades normalizadas
    con la media y desviación del entrenamiento fault-free."""
    mean = train[XMEAS_COLS].mean()
    std = train[XMEAS_COLS].std().replace(0, np.nan)

    test = test.sort_values(["simulationRun", "sample"], kind="stable")
    g = test.groupby("simulationRun")
    rows = []
    for c in XMEAS_COLS:
        nxt = g[c].shift(-1)
        valid = nxt.notna()
        cur = test.loc[valid, c]
        nx = nxt[valid]
        mse_pers = float((((nx - cur) / std[c]) ** 2).mean())
        mse_media = float((((nx - mean[c]) / std[c]) ** 2).mean())
        repetidos = float((nx.to_numpy() == cur.to_numpy()).mean())
        rows.append({"variable": c, "mse_persistencia": mse_pers,
                     "mse_prediccion_media": mse_media,
                     "fraccion_valores_repetidos": repetidos})
    out = pd.DataFrame(rows)
    out["tipo_sugerido"] = np.where(out.fraccion_valores_repetidos > 0.3,
                                    "analizador (valor repetido)", "continua")
    return out


def residuos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["periodo"] = np.where(df["sample"] <= FAULT_ONSET, "antes", "despues")
    rows = []
    for nombre, fn in CANDIDATAS.items():
        df["_r"] = fn(df)
        agg = df.groupby(["faultNumber", "periodo"])["_r"].agg(["mean", "std", "count"])
        for (f, per), r in agg.iterrows():
            rows.append({"candidata": nombre, "faultNumber": int(f), "periodo": per,
                         "media": r["mean"], "std": r["std"], "n": int(r["count"])})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--faulty_runs", type=int, default=50)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _cargar_fault_free(data_dir / "TEP_FaultFree_Training.csv")
    test_ff = _cargar_fault_free(data_dir / "TEP_FaultFree_Testing.csv")

    # 1) Referencias por variable
    ref = referencias_por_variable(train, test_ff)
    ref.to_csv(out_dir / "diag_variables.csv", index=False)
    print("== Referencias en test fault-free (unidades normalizadas) ==")
    print(ref.round(3).to_string(index=False))
    print(f"\nMSE medio -> persistencia: {ref.mse_persistencia.mean():.3f} | "
          f"media: {ref.mse_prediccion_media.mean():.3f}")
    cont = ref[ref.tipo_sugerido == "continua"]
    print(f"Solo variables continuas ({len(cont)}) -> persistencia: "
          f"{cont.mse_persistencia.mean():.3f} | media: {cont.mse_prediccion_media.mean():.3f}")

    # 2) Residuos de las candidatas
    res_ff = residuos(test_ff)
    partes = []
    cols = ["faultNumber", "simulationRun", "sample"] + XMEAS_COLS
    ruta = data_dir / "TEP_Faulty_Testing.csv"
    for chunk in pd.read_csv(ruta, dtype=DTYPES, chunksize=1_000_000, usecols=cols):
        partes.append(chunk[chunk["simulationRun"] <= args.faulty_runs])
    faulty = pd.concat(partes, ignore_index=True)
    res = pd.concat([res_ff, residuos(faulty)], ignore_index=True)
    res.to_csv(out_dir / "diag_residuos.csv", index=False)

    print("\n== Residuo de cada candidata (media en fault-free, antes del fallo y después) ==")
    escala = float(train["xmeas_6"].mean())
    for nombre in CANDIDATAS:
        sub = res[res.candidata == nombre]
        ff = sub[(sub.faultNumber == 0) & (sub.periodo == "antes")]
        ff_m = float(ff["media"].iloc[0]) if len(ff) else float("nan")
        ff_s = float(ff["std"].iloc[0]) if len(ff) else float("nan")
        print(f"\n{nombre}: fault-free media={ff_m:.4g}, std={ff_s:.4g} "
              f"(referencia: caudal al reactor ~ {escala:.3g})")
        post = sub[(sub.faultNumber > 0) & (sub.periodo == "despues")].set_index("faultNumber")
        if len(post):
            z = ((post["media"] - ff_m) / ff_s).abs().round(2)
            print("  |desviación en std de fault-free| tras el fallo, por IDV:")
            print("  " + ", ".join(f"{int(k)}:{v}" for k, v in z.items()))
    print(f"\nGuardado en {out_dir}/diag_variables.csv y {out_dir}/diag_residuos.csv")


if __name__ == "__main__":
    main()
