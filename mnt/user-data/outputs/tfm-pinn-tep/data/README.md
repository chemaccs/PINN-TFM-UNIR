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
