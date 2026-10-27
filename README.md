# TFM-PINN-UNIR

Trabajo Fin de Máster (UNIR, Máster en Inteligencia Artificial): **PINN aplicado al proceso Tennessee Eastman**. Predicción dinámica a un paso y evaluación de robustez ante fallos, comparando una red informada por la física (PINN) con una red puramente basada en datos (MLP).

## Qué se compara

- **Tarea:** predecir las medidas del proceso en el instante t+1 (`xmeas(t+1)`) a partir de las variables manipuladas y medidas en t.
- **Modelos:** MLP sin física (baseline) frente a la misma red con un término de pérdida físico (PINN). La arquitectura es idéntica; solo cambia la pérdida.
- **Métrica:** MSE sobre variables normalizadas, calculado por simulación; contraste de Wilcoxon apareado (α = 0,05).
- **Datos:** Rieth et al. (2017), Tennessee Eastman Process, Harvard Dataverse.

## Estructura

```
src/            código del entrenamiento y la evaluación (data, models, physics, train, evaluate)
convert_rdata.py   convierte los .RData del dataset a CSV
data/           ficheros del dataset (no se versionan)
checkpoints/    modelos entrenados (no se versionan)
results/        registro de experimentos (log.csv)
```

## Puesta en marcha

Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

1. Descarga los cuatro ficheros `.RData` del dataset y colócalos en `data/`.
2. Conviértelos a CSV: `python convert_rdata.py`
3. Prueba corta del baseline: `python -m src.train --variant baseline --epochs 3`

Los ficheros faulty son muy grandes (5 y 9,6 millones de filas). Por defecto el test faulty usa las 100 primeras simulaciones de cada fallo (`--faulty_runs`, 0 = todas).

## Variantes y ejemplos

```bash
python -m src.train --variant baseline --epochs 30
python -m src.train --variant pinn_soft --epochs 30 --lambda_phys 1
python -m src.train --variant pinn_soft --epochs 30 --lambda_phys 100 --residual
python -m src.evaluate --checkpoint checkpoints/baseline.pt --variant baseline
```

- `--lambda_phys`: peso del término físico (ignorado en el baseline).
- `--residual`: la red aprende la corrección sobre la persistencia.
- `--seed`: semilla (por defecto 42). Los checkpoints llevan sufijo `_lam<λ>`, `_res` y `_s<semilla>` para no sobrescribirse.

## Estado de la física

El término físico actual es un balance de materia en el punto de mezcla de la entrada al reactor (`src/physics.py`). Es una restricción algebraica provisional: los pesos moleculares y el volumen molar están pendientes de verificar con la tutora.

## Flujo de trabajo

Ramas: `main` (protegida) ← `develop` (integración) ← `<nombre>/<tarea>`. Los PR a `develop` se hacen con squash merge; de `develop` a `main` aprueba una persona distinta del autor. Commits en formato simplificado: `feat`, `fix`, `data`, `exp`, `docs`, `refactor`, `chore`. Los detalles están en la memoria del TFM.

## Registro de experimentos

Cada entrenamiento añade una fila a `results/log.csv`: variante, épocas, mejor época, tamaño de lote, tasa de aprendizaje, λ, semilla, MSE de validación, pérdida física, MSE de test (fault-free y faulty) y checkpoint.
