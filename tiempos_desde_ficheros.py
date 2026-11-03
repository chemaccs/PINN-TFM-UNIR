"""
Estima cuánto tardó cada modelo a partir de la fecha de modificación de los
checkpoints (solo lee, no modifica nada).

Uso (en el ordenador donde se ejecutó, desde la raíz del repo):

    python tiempos_desde_ficheros.py CARPETA_CHECKPOINTS [--ext .pt .pth ...] [--salida tiempos_equipo.csv]

Idea: si los modelos se entrenaron uno detrás de otro, la diferencia entre la
fecha de dos checkpoints consecutivos es lo que tardó el segundo (carga de datos
+ entrenamiento + evaluación del anterior/actual, según cuándo se guarde).
Los saltos mayores que --max_min (por defecto 180 min) se marcan como pausa y se
excluyen del resumen. El primer modelo de cada tanda no tiene tiempo.
"""
import argparse, csv, re, statistics
from datetime import datetime
from pathlib import Path

EXT = [".pt", ".pth", ".keras", ".h5", ".ckpt", ".pkl", ".joblib", ".npz"]
PAT = re.compile(r"_n(\w+?)_s(\d+)")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta")
    ap.add_argument("--ext", nargs="+", default=EXT)
    ap.add_argument("--max_min", type=float, default=180)
    ap.add_argument("--salida", default="tiempos_estimados.csv")
    a = ap.parse_args()
    ext = {e.lower() for e in a.ext}
    fs = [p for p in Path(a.carpeta).rglob("*")
          if p.is_file() and p.suffix.lower() in ext and ".venv" not in p.parts and ".git" not in p.parts]
    if not fs:
        raise SystemExit("No encuentro checkpoints con esas extensiones; usa --ext o revisa la carpeta.")
    fs.sort(key=lambda p: p.stat().st_mtime)
    filas, prev = [], None
    for p in fs:
        t = p.stat().st_mtime
        m = PAT.search(p.name)
        modelo = p.name
        brazo = ("pinn" if "pinn" in modelo else "mlp") + ("_res" if "_res" in modelo else "")
        n, s = (m.group(1), m.group(2)) if m else ("?", "?")
        mins = None if prev is None else (t - prev) / 60
        pausa = mins is not None and mins > a.max_min
        filas.append([modelo, datetime.fromtimestamp(t).isoformat(timespec="seconds"), brazo, n, s,
                      "" if mins is None else round(mins, 1), int(pausa)])
        prev = t
    with open(a.salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fichero", "fecha", "brazo", "n", "semilla", "min_desde_anterior", "posible_pausa"])
        w.writerows(filas)
    grupos = {}
    for f in filas:
        if f[5] != "" and not f[6]:
            grupos.setdefault((f[2], f[3]), []).append(f[5])
    print(f"{len(filas)} ficheros -> {a.salida}\n")
    print(f"{'brazo':10s} {'n':>5s} {'k':>3s} {'mediana(min)':>13s} {'min-max':>14s}")
    for (b, n), v in sorted(grupos.items(), key=lambda x: (x[0][0], str(x[0][1]).zfill(5))):
        print(f"{b:10s} {n:>5s} {len(v):>3d} {statistics.median(v):>13.1f} {min(v):>6.1f}-{max(v):<6.1f}")
    print("\nRevisa el CSV: las filas con posible_pausa=1 no cuentan en el resumen.")

if __name__ == "__main__":
    main()
