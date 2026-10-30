"""
Resume los barridos de tamaño de entrenamiento (results/barrido*.csv): curva de
aprendizaje por brazo (media y desviación entre semillas) y diferencia apareada
PINN - baseline por semilla y tamaño.

Salidas en results/: barrido_resumen.csv, barrido_diferencias.csv, barrido_curvas.png

Uso:
    python resumen_barrido.py            # lee results/barrido*.csv
"""
from __future__ import annotations

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICAS = {
    "mse_fault_free": "MSE fault-free (test)",
    "mse_faulty_despues_mediana": "MSE faulty tras el fallo (mediana por simulación)",
    "mse_faulty_despues": "MSE faulty tras el fallo (media)",
}


def main() -> None:
    ficheros = sorted(f for f in glob.glob("results/barrido*.csv")
                      if "resumen" not in f and "diferencias" not in f)
    df = pd.concat([pd.read_csv(f) for f in ficheros], ignore_index=True)
    df = df.drop_duplicates(subset=["variant", "lambda_phys", "residual", "train_sims", "seed"])
    df["n"] = df.train_sims.replace("all", np.nan).astype(float)
    n_max = int(df.n.max()) * 1 if df.n.notna().any() else 400
    df["n"] = df.n.fillna(400.0)  # "all" ~ todas las simulaciones de entrenamiento (~400)
    df["brazo"] = (np.where(df.variant == "baseline", "MLP", "PINN λ=" + df.lambda_phys.astype(str))
                   + np.where(df.residual == 1, " residual", ""))

    agg = df.groupby(["brazo", "n"]).agg(
        semillas=("seed", "nunique"),
        **{f"{m}_media": (m, "mean") for m in METRICAS},
        **{f"{m}_std": (m, "std") for m in METRICAS}).reset_index()
    agg.to_csv("results/barrido_resumen.csv", index=False)
    for m, t in METRICAS.items():
        print(f"\n== {t}: media entre semillas ==")
        print(agg.pivot(index="brazo", columns="n", values=f"{m}_media").round(4).to_string())

    # Diferencia apareada PINN - MLP (misma semilla, mismo tamaño, misma formulación)
    filas = []
    for res in (0, 1):
        base = df[(df.variant == "baseline") & (df.residual == res)].set_index(["n", "seed"])
        pin = df[(df.variant != "baseline") & (df.residual == res)]
        for lam, g in pin.groupby("lambda_phys"):
            g = g.set_index(["n", "seed"])
            idx = g.index.intersection(base.index)
            for n in sorted({i[0] for i in idx}):
                sel = [i for i in idx if i[0] == n]
                for m in METRICAS:
                    d = (g.loc[sel, m] - base.loc[sel, m]).to_numpy()
                    filas.append({"formulacion": "residual" if res else "directa", "lambda": lam,
                                  "n": n, "metrica": m, "dif_media": d.mean(),
                                  "dif_std": d.std(ddof=1) if len(d) > 1 else np.nan,
                                  "semillas": len(d), "pinn_mejor_en": int((d < 0).sum())})
    dif = pd.DataFrame(filas)
    dif.to_csv("results/barrido_diferencias.csv", index=False)
    if len(dif):
        print("\n== Diferencia PINN - MLP (negativa = la PINN mejora), fault-free ==")
        print(dif[dif.metrica == "mse_fault_free"].round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, m in zip(ax, ["mse_fault_free", "mse_faulty_despues_mediana"]):
        for brazo, g in agg.groupby("brazo"):
            g = g.sort_values("n")
            a.errorbar(g.n, g[f"{m}_media"], yerr=g[f"{m}_std"].fillna(0), marker="o",
                       capsize=3, label=brazo)
        a.set_xscale("log"); a.set_xlabel("simulaciones de entrenamiento")
        a.set_ylabel("MSE normalizado"); a.set_title(METRICAS[m], fontsize=9)
        a.grid(alpha=.3)
    ax[1].set_yscale("log")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/barrido_curvas.png", dpi=150)
    print("\nGuardado: results/barrido_resumen.csv, barrido_diferencias.csv, barrido_curvas.png")


if __name__ == "__main__":
    main()
