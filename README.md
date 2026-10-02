# TFM — Modelado dinámico de procesos químicos industriales mediante PINNs
### Aplicación al proceso Tennessee Eastman (TEP)

Máster en Inteligencia Artificial — UNIR
Equipo: Jorge · Arturo · José María
Directora: Ivón Oristela Benítez

---

## 1. Objetivo

Comparar distintas formas de modelar el comportamiento dinámico del **Tennessee Eastman
Process (TEP)** mediante redes neuronales, evaluando si incorporar conocimiento físico del
proceso (balances de materia/energía) mejora la capacidad de generalización frente a una
red puramente orientada a datos, especialmente en escenarios con fallo (`IDV_1`...`IDV_20`).

Métrica de evaluación acordada con la tutora: **Error Cuadrático Medio (MSE)**.

> Este repositorio es un punto de partida funcional, no la entrega final. Las ecuaciones
> físicas concretas (balances de materia por componente, cinética de las reacciones del
> reactor, relaciones de equilibrio del separador/stripper) deben revisarse y ajustarse
> con la tutora antes de considerarlas correctas — ver `src/physics.py`.

## 2. Reparto de trabajo

| Integrante | Entorno | Variante de modelo |
|---|---|---|
| Jorge | Ordenador personal | **Baseline** — red sin información física (solo datos) |
| Arturo | Ordenador personal | **PINN "soft"** — penalización del residuo físico en la función de pérdida |
| José María | Ordenador personal | **PINN alternativa** — arquitectura o estrategia de imposición de física distinta (a definir) |
| Equipo (compartido) | Google Colab | Entrenamientos con GPU y barrido de hiperparámetros sobre la variante más prometedora |

Cada integrante trabaja sobre su propia rama (`baseline`, `pinn-soft`, `pinn-alt`) partiendo
de la misma base de código (`src/`), y registra sus resultados en `results/log.csv`
(ver plantilla en ese fichero).

## 3. Estructura del repositorio

```
tfm-pinn-tep/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          # cómo obtener y colocar el dataset TEP
├── src/
│   ├── data.py             # carga y preprocesado del dataset TEP
│   ├── models.py           # BaselineMLP y PINN (arquitectura configurable)
│   ├── physics.py          # residuos físicos del TEP (PUNTO A REVISAR CON IVÓN)
│   ├── train.py            # bucle de entrenamiento + logging de MSE
│   └── evaluate.py         # evaluación en test (fault-free y por tipo de fallo)
├── notebooks/
│   └── colab_quickstart.ipynb
└── results/
    └── log.csv             # registro común de experimentos (plantilla)
```

## 4. Dataset

El TEP no viene en este repositorio (son varios GB). Instrucciones de descarga y formato
esperado en `data/README.md`. El dataset de referencia es el de **Rieth et al. (2017)**,
disponible en Harvard Dataverse, con simulaciones `fault-free` y `faulty` para
entrenamiento y test.

## 5. Puesta en marcha rápida

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Entrenar el baseline (Jorge)
python -m src.train --variant baseline --epochs 100

# Entrenar la PINN soft (Arturo)
python -m src.train --variant pinn_soft --epochs 100 --lambda_phys 0.1

# Evaluar un modelo entrenado
python -m src.evaluate --checkpoint checkpoints/pinn_soft.pt
```

## 6. Google Colab

`notebooks/colab_quickstart.ipynb` monta el mismo código vía `pip install` directo del
repo (o subiendo el .zip), pensado para entrenamientos más largos con GPU y para el
barrido de hiperparámetros, una vez las tres variantes locales tengan un primer MSE
de referencia.

## 7. Referencia metodológica

El repositorio [`battery-digital-twin-using-pinn`](https://github.com/SelloP28/battery-digital-twin-using-pinn)
(Ivón) se ha usado como inspiración de estructura (separación en módulos, comparación
explícita "con física" vs "solo datos"), **no como fuente de las ecuaciones físicas**,
ya que su dominio (electroquímica de baterías) no es trasladable al TEP.

# Dataset — Tennessee Eastman Process (TEP)

Este proyecto usa el dataset de simulación del TEP publicado por **Rieth, C. A., Amsel,
B. D., Tran, R., & Cook, B. (2017)**, disponible públicamente en Harvard Dataverse:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1

## Ficheros esperados

El dataset se distribuye en formato `.RData` / `.dat`, con una simulación "fault-free"
(proceso en condiciones normales) y 20 simulaciones "faulty" (`IDV_1` a `IDV_20`), tanto
para entrenamiento como para test. Tras descargar y convertir a CSV (hay scripts de
conversión de R a CSV ampliamente documentados para este dataset), coloca los ficheros
aquí con esta estructura:

```
data/
├── TEP_FaultFree_Training.csv
├── TEP_FaultFree_Testing.csv
├── TEP_Faulty_Training.csv
└── TEP_Faulty_Testing.csv
```

## Columnas relevantes

- `faultNumber`: 0 (fault-free) o 1-20 (tipo de fallo, IDV_n)
- `simulationRun`: identificador de cada corrida de simulación
- `sample`: instante temporal dentro de la corrida
- `xmeas_1` ... `xmeas_41`: 41 variables medidas del proceso
- `xmv_1` ... `xmv_11`: 11 variables manipuladas (actuadores)

`src/data.py` asume estos nombres de columna. Si tu conversión del dataset usa otros
nombres, ajusta el diccionario `COLUMN_MAP` al inicio de ese fichero.

## Nota importante

**No subáis el dataset completo al repositorio de GitHub** (supera con facilidad los
límites de tamaño). Añadid `data/*.csv` a `.gitignore` y compartid el dataset ya
descargado entre vosotros por Drive o similar, para que cada uno lo coloque en su
copia local de `data/`.


# Dataset — Tennessee Eastman Process (TEP)

Este proyecto usa el dataset de simulación del TEP publicado por **Rieth, C. A., Amsel,
B. D., Tran, R., & Cook, B. (2017)**, disponible públicamente en Harvard Dataverse:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1

## Ficheros esperados

El dataset se distribuye en formato `.RData` / `.dat`, con una simulación "fault-free"
(proceso en condiciones normales) y 20 simulaciones "faulty" (`IDV_1` a `IDV_20`), tanto
para entrenamiento como para test. Tras descargar y convertir a CSV (hay scripts de
conversión de R a CSV ampliamente documentados para este dataset), coloca los ficheros
aquí con esta estructura:

```
data/
├── TEP_FaultFree_Training.csv
├── TEP_FaultFree_Testing.csv
├── TEP_Faulty_Training.csv
└── TEP_Faulty_Testing.csv
```

## Columnas relevantes

- `faultNumber`: 0 (fault-free) o 1-20 (tipo de fallo, IDV_n)
- `simulationRun`: identificador de cada corrida de simulación
- `sample`: instante temporal dentro de la corrida
- `xmeas_1` ... `xmeas_41`: 41 variables medidas del proceso
- `xmv_1` ... `xmv_11`: 11 variables manipuladas (actuadores)

`src/data.py` asume estos nombres de columna. Si tu conversión del dataset usa otros
nombres, ajusta el diccionario `COLUMN_MAP` al inicio de ese fichero.

## Nota importante

**No subáis el dataset completo al repositorio de GitHub** (supera con facilidad los
límites de tamaño). Añadid `data/*.csv` a `.gitignore` y compartid el dataset ya
descargado entre vosotros por Drive o similar, para que cada uno lo coloque en su
copia local de `data/`.
