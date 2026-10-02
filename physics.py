"""
Término físico para la PINN aplicada al TEP.

*** IMPORTANTE — LEER ANTES DE USAR ***

El TEP (Downs & Vogel, 1993) tiene un modelo físico completo con balances de
materia por componente, cinética de las dos reacciones del reactor, y
relaciones de equilibrio líquido-vapor en el separador y el stripper. Ese
modelo completo es complejo y requiere mapear con precisión qué xmeas_i y
xmv_i corresponden a qué corriente/variable del proceso (caudal de
alimentación, composición, presión del reactor, nivel del separador, etc.).

Esta plantilla NO implementa ese modelo completo. Implementa un residuo
físico simplificado y genérico — un balance de materia global en estado
estacionario aproximado — como PUNTO DE PARTIDA funcional para que el
pipeline de entrenamiento compile y dé un primer MSE comparable entre
variantes. Antes de usar los resultados en la memoria del TFM, hay que:

  1. Confirmar con Ivón qué xmeas_i / xmv_i corresponden a qué caudales
     (en el paper original de Downs & Vogel y en el dataset de Rieth et al.
     vienen documentados, pero conviene verificarlo entre los tres).
  2. Sustituir `global_mass_balance_residual` por el balance (o balances)
     que decidáis usar como restricción física real.
  3. Documentar en la memoria qué ecuación se usó y por qué, con su
     justificación física — esto es parte de lo que se evalúa en un TFM,
     no solo el código.
"""
from __future__ import annotations

import torch

# --- Índices de ejemplo dentro del vector xmeas (AJUSTAR TRAS VERIFICAR) ---
# Según Downs & Vogel (1993), a modo de referencia orientativa:
#   xmeas_1..4  : caudales de alimentación (A, D, E, A+C)
#   xmeas_6     : caudal de reactivo al reactor
#   xmeas_10    : caudal de purga
#   xmeas_17    : caudal de producto del stripper
# ESTOS ÍNDICES SON ORIENTATIVOS, NO DEFINITIVOS: verificar contra la
# documentación del dataset antes de usarlos para resultados finales.
FEED_IDX = [0, 1, 2, 3]     # xmeas_1..4 (índice 0-based)
PURGE_IDX = 9                # xmeas_10
PRODUCT_IDX = 16             # xmeas_17


def global_mass_balance_residual(xmeas_pred: torch.Tensor) -> torch.Tensor:
    """Residuo de un balance de materia global muy simplificado:
    entradas ≈ salidas (purga + producto), en estado cuasi-estacionario.

    xmeas_pred: tensor (batch, 41) con las variables medidas predichas por
    la red (ya des-escaladas a unidades físicas si queréis que el residuo
    tenga sentido en magnitud real; si trabajáis en espacio escalado, el
    residuo sigue siendo útil como señal de penalización relativa, pero
    no tiene unidades físicas interpretables).
    """
    inflow = xmeas_pred[:, FEED_IDX].sum(dim=1)
    outflow = xmeas_pred[:, PURGE_IDX] + xmeas_pred[:, PRODUCT_IDX]
    residual = inflow - outflow
    return residual


def physics_loss(xmeas_pred: torch.Tensor) -> torch.Tensor:
    """Pérdida física = MSE del residuo respecto a 0 (balance perfecto)."""
    residual = global_mass_balance_residual(xmeas_pred)
    return torch.mean(residual**2)
