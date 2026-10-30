"""
Diagnóstico de restricciones físicas DINÁMICAS candidatas para la PINN
(complementa a diagnostico.py, que solo mira balances algebraicos).

Idea: una ecuación de acumulación relaciona el CAMBIO de un nivel entre t y t+1
con los caudales que entran y salen. Si se cumple en los datos reales, se puede
usar como residuo físico entre la entrada (t) y la predicción (t+1), que es lo
que la restricción algebraica actual no hace.

Candidatas (todas se ajustan por mínimos cuadrados en fault-free; luego se
comprueba si se mantienen bajo fallo, que es lo que debe pasar con una ley):

  A) Nivel del stripper (XMEAS 15, %):
         x15(t+1) - x15(t)  ~  a * x14(t) + b * x17(t) + c
     x14 = caudal de líquido que sale del separador y entra al stripper (m3/h),
     x17 = caudal de producto del stripper (m3/h). Si es un balance de
     líquido en las mismas unidades, se espera a > 0, b < 0 y |a| ≈ |b|.

  B) Nivel del separador (XMEAS 12, %):
         x12(t+1) - x12(t)  ~  a * x14(t) + c
     (la entrada de condensado no se mide: solo se comprueba la señal de la salida).

  C) Diferencia de presión reactor-separador (XMEAS 7 - XMEAS 13, kPa):
     debe ser casi constante (pérdida de carga del condensador).

Salidas: pantalla y results/diag_dinamico.csv
Uso (desde la raíz):
    python diagnostico_dinamico.py --data_dir data_muestra --faulty_runs 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FAULT_ONSET = 160
COLS = ["xmeas_7", "xmeas_8", "xmeas_12", "xmeas_13", "xmeas_14", "xmeas_15", "xmeas_17"]
KEYS = ["faultNumber", "simulationRun", "sample"]
DTYPES = {c: "float32" for c in COLS}
DTYPES.update({k: "int32" for k in KEYS})

# (nombre, variable objetivo, regresores en t)
DINAMICAS = [
    ("A_nivel_stripper", "xmeas_15", ["xmeas_14", "xmeas_17"]),
    ("B_nivel_separador", "xmeas_12", ["xmeas_14"]),
]


def pares(df: pd.DataFrame) -> pd.DataFrame:
    """Añade el valor en t+1 de cada variable dentro de su simulación."""
    df = df.sort_values(KEYS, kind="stable").reset_index(drop=True)
    g = df.groupby(["faultNumber", "simulationRun"])
    for c in COLS:
        df[c + "_sig"] = g[c].shift(-1)
    df = df.dropna()
    df["periodo"] = np.where(df["sample"] <= FAULT_ONSET, "antes", "despues")
    return df


def ventana(df: pd.DataFrame, objetivo: str, regs: list[str], k: int):
    """Cambio del objetivo en k pasos (x(t+k) - x(t)) y suma de cada regresor
    en los k instantes t..t+k-1, dentro de cada simulación (df ya ordenado)."""
    llave = [df["faultNumber"], df["simulationRun"]]
    y = df.groupby(["faultNumber", "simulationRun"])[objetivo].shift(-k) - df[objetivo]
    cols = [y]
    for r in regs:
        c = df.groupby(["faultNumber", "simulationRun"])[r].cumsum()
        suma = c.groupby(llave).shift(-(k - 1)) - (c - df[r])
        cols.append(suma)
    d = pd.concat(cols, axis=1).dropna()
    return d.iloc[:, 1:].to_numpy(dtype=np.float64), d.iloc[:, 0].to_numpy(dtype=np.float64)


def ols(X: np.ndarray, y: np.ndarray):
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return coef, resid, float(r2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_muestra")
    ap.add_argument("--faulty_runs", type=int, default=100)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--ventanas", type=int, nargs="+", default=[1, 5, 10, 20, 40, 80],
                    help="Pasos k en los que se mide la acumulación (el ruido pesa menos al crecer k)")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    ff = pares(pd.read_csv(data_dir / "TEP_FaultFree_Testing.csv",
                           usecols=KEYS + COLS, dtype=DTYPES))
    partes = []
    for chunk in pd.read_csv(data_dir / "TEP_Faulty_Testing.csv", usecols=KEYS + COLS,
                             dtype=DTYPES, chunksize=1_000_000):
        partes.append(chunk[chunk["simulationRun"] <= args.faulty_runs])
    fa = pares(pd.concat(partes, ignore_index=True))
    fa = fa[fa["periodo"] == "despues"]

    filas = []
    print("== Candidatas dinámicas: ajuste en fault-free y comprobación bajo fallo ==")
    for nombre, objetivo, regs in DINAMICAS:
        def XY(d):
            y = (d[objetivo + "_sig"] - d[objetivo]).to_numpy(dtype=np.float64)
            X = d[regs].to_numpy(dtype=np.float64)
            return X, y

        X0, y0 = XY(ff)
        coef, resid0, r2 = ols(X0, y0)
        sd_y0, sd_r0 = float(y0.std()), float(resid0.std())
        print(f"\n{nombre}: cambio de {objetivo} en un paso ({sd_y0:.4g} de desv. típica)")
        print("  coeficientes: " + ", ".join(f"{r}={c:.4g}" for r, c in zip(regs, coef[:-1]))
              + f", constante={coef[-1]:.4g}")
        print(f"  R² en fault-free: {r2:.3f}  (residuo: desv. típica {sd_r0:.4g})")
        if len(regs) == 2:
            print(f"  cociente de coeficientes: {coef[0] / coef[1]:.2f}  (balance ideal: -1)")
        filas.append({"candidata": nombre, "conjunto": "fault_free", "faultNumber": 0,
                      "r2": r2, "sd_residuo": sd_r0, "ratio_sd_residuo": 1.0})

        res = []
        for f, d in fa.groupby("faultNumber"):
            X, y = XY(d)
            r = y - np.column_stack([X, np.ones(len(X))]) @ coef
            ratio = float(r.std() / sd_r0)
            r2f = float(1 - r.var() / y.var()) if y.var() > 0 else float("nan")
            res.append((int(f), r2f, ratio))
            filas.append({"candidata": nombre, "conjunto": "faulty_despues", "faultNumber": int(f),
                          "r2": r2f, "sd_residuo": float(r.std()), "ratio_sd_residuo": ratio})
        print("  bajo fallo (coef. de fault-free): IDV:ratio de desv. del residuo (1 = igual que en normal)")
        print("  " + ", ".join(f"{f}:{ra:.2f}" for f, _, ra in res))

    print("\n== C) Diferencia de presión reactor - separador (XMEAS 7 - XMEAS 13, kPa) ==")
    d0 = (ff["xmeas_7"] - ff["xmeas_13"]).astype(float)
    print(f"  fault-free: media {d0.mean():.4g}, desv. típica {d0.std():.4g}")
    dd = (fa["xmeas_7"] - fa["xmeas_13"]).astype(float)
    desv = (dd.groupby(fa["faultNumber"]).mean() - d0.mean()).abs() / d0.std()
    print("  desviación de la media bajo fallo en desv. típicas normales, por IDV:")
    print("  " + ", ".join(f"{int(k)}:{v:.2f}" for k, v in desv.items()))
    filas.append({"candidata": "C_dif_presion", "conjunto": "fault_free", "faultNumber": 0,
                  "r2": float("nan"), "sd_residuo": float(d0.std()), "ratio_sd_residuo": 1.0})

    print("\n== Acumulación en ventanas de k pasos (3 min por paso), solo fault-free ==")
    print("  A: nivel stripper ~ a*suma(x14) + b*suma(x17)  |  B: nivel separador ~ a*suma(x14)")
    print(f"  {'k':>4} {'R2_A':>7} {'a':>9} {'b':>9} {'a/b':>7} {'R2_B':>7}")
    for k in args.ventanas:
        XA, yA = ventana(ff, "xmeas_15", ["xmeas_14", "xmeas_17"], k)
        cA, _, r2A = ols(XA, yA)
        XB, yB = ventana(ff, "xmeas_12", ["xmeas_14"], k)
        _, _, r2B = ols(XB, yB)
        print(f"  {k:>4} {r2A:>7.3f} {cA[0]:>9.4g} {cA[1]:>9.4g} {cA[0] / cA[1]:>7.2f} {r2B:>7.3f}")
        filas.append({"candidata": f"A_ventana_{k}", "conjunto": "fault_free", "faultNumber": 0,
                      "r2": r2A, "sd_residuo": float("nan"), "ratio_sd_residuo": float("nan")})
        filas.append({"candidata": f"B_ventana_{k}", "conjunto": "fault_free", "faultNumber": 0,
                      "r2": r2B, "sd_residuo": float("nan"), "ratio_sd_residuo": float("nan")})

    print("\n== Mismo ajuste DENTRO de cada fallo (solo tras el fallo, ajustado por IDV) ==")
    print("  En fault-free los niveles no se mueven (régimen estacionario) y no hay nada que explicar.")
    print("  Aquí se mira si la ley aparece cuando algo sí se mueve. Formato: IDV:R2 (a/b)")
    for k in (1, 10, 40):
        la, lb = [], []
        for f, d in fa.groupby("faultNumber"):
            XA, yA = ventana(d, "xmeas_15", ["xmeas_14", "xmeas_17"], k)
            XB, yB = ventana(d, "xmeas_12", ["xmeas_14"], k)
            if len(yA) < 50 or len(yB) < 50:
                continue
            cA, _, r2A = ols(XA, yA)
            _, _, r2B = ols(XB, yB)
            la.append(f"{int(f)}:{r2A:.2f} ({cA[0] / cA[1]:.1f})")
            lb.append(f"{int(f)}:{r2B:.2f}")
            filas.append({"candidata": f"A_en_fallo_k{k}", "conjunto": "faulty_despues",
                          "faultNumber": int(f), "r2": r2A, "sd_residuo": float("nan"),
                          "ratio_sd_residuo": float("nan")})
            filas.append({"candidata": f"B_en_fallo_k{k}", "conjunto": "faulty_despues",
                          "faultNumber": int(f), "r2": r2B, "sd_residuo": float("nan"),
                          "ratio_sd_residuo": float("nan")})
        print(f"  k={k}  A (nivel stripper): " + ", ".join(la))
        print(f"  k={k}  B (nivel separador): " + ", ".join(lb))

    pd.DataFrame(filas).to_csv(out_dir / "diag_dinamico.csv", index=False)
    print(f"\nGuardado en {out_dir}/diag_dinamico.csv")


if __name__ == "__main__":
    main()
