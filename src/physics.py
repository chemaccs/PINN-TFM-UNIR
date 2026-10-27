"""
Término físico para la PINN aplicada al TEP (versión 2).

Restricción elegida: BALANCE DE MATERIA EN EL PUNTO DE MEZCLA DE LA ENTRADA
AL REACTOR. El caudal de alimentación al reactor (XMEAS 6) debe ser la suma
de las corrientes que llegan a ese punto:

    XMEAS 6  ≈  XMEAS 1 (A) + XMEAS 4 (A+C) + XMEAS 5 (reciclo)
                + (VM/1000) · ( XMEAS 2 / MW_D + XMEAS 3 / MW_E )

Los XMEAS 1, 4, 5 y 6 son caudales volumétricos (kscmh); D y E (XMEAS 2 y 3)
son caudales másicos (kg/h) y se convierten a kscmh con su peso molecular y el
volumen molar estándar (VM = 22,414 m3/kmol). Con esa conversión, el residuo
medio sobre datos reales fault-free pasa de 5,8 kscmh (solo volumétricos) a
1,07 kscmh (2,5 % del caudal) con desviación 0,31 (diagnostico.py, 26/10/2026).

El residuo restante es casi constante, así que se modela como un SESGO
conocido (BIAS) que recoge lo no medido (p. ej. el aporte del stripper) y se
divide por la desviación típica de ruido (SIGMA). La pérdida es entonces
adimensional y del orden de 1 sobre datos reales, comparable al MSE de datos
normalizados, lo que hace interpretable el peso lambda.

LIMITACIONES (decirlas en la memoria):
  - Es una restricción ALGEBRAICA e instantánea (conservación en un punto),
    no una ecuación diferencial del proceso.
  - Pesos moleculares y volumen molar proceden de Downs y Vogel (1993): a
    verificar con la tutora. BIAS y SIGMA se estimaron en fault-free y deben
    recalcularse con `diagnostico.py` si cambia la muestra.
  - Una ley de conservación debe seguir cumpliéndose también bajo fallo; por
    eso aquí NO se espera que el residuo cambie con los fallos.

NOTA SOBRE ESCALADO: la función espera xmeas YA DES-ESCALADOS (unidades
originales). Debe hacerse con operaciones de PyTorch (ver `unscale_torch` en
train.py) para no romper el grafo de autodiferenciación.
"""
from __future__ import annotations

import torch

# Índices 0-based dentro del vector xmeas (xmeas_i -> i-1)
IDX_A, IDX_D, IDX_E, IDX_AC = 0, 1, 2, 3     # xmeas_1, 2, 3, 4
IDX_RECYCLE, IDX_REACTOR_FEED = 4, 5          # xmeas_5, 6

VM = 22.414          # m3/kmol a condiciones estándar
MW_D, MW_E = 32.0, 46.0   # kg/kmol (Downs y Vogel, 1993; a verificar)

# Estimados con diagnostico.py sobre fault-free (kscmh)
BIAS = 1.072
SIGMA = 0.306


def reactor_feed_residual(xmeas_pred: torch.Tensor) -> torch.Tensor:
    """Residuo del balance en la entrada del reactor, en kscmh, ya restado el
    sesgo constante. xmeas_pred: (batch, 41), en unidades originales."""
    inflow = (
        xmeas_pred[:, IDX_A]
        + xmeas_pred[:, IDX_AC]
        + xmeas_pred[:, IDX_RECYCLE]
        + (VM / 1000.0) * (xmeas_pred[:, IDX_D] / MW_D + xmeas_pred[:, IDX_E] / MW_E)
    )
    return xmeas_pred[:, IDX_REACTOR_FEED] - inflow - BIAS


def physics_loss(xmeas_pred: torch.Tensor) -> torch.Tensor:
    """Pérdida física = media del residuo normalizado al cuadrado.
    xmeas_pred debe estar ya des-escalado."""
    r = reactor_feed_residual(xmeas_pred) / SIGMA
    return torch.mean(r**2)
