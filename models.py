"""
Arquitecturas de red. Ambas variantes (baseline y PINN) comparten la misma
arquitectura base; lo que cambia es la función de pérdida usada en train.py
(ver src/physics.py para el término físico de la PINN).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Red totalmente conectada: predice xmeas(t+1) a partir de [xmv(t), xmeas(t)].

    Se usa tanto para el baseline (Jorge) como para las variantes PINN
    (Arturo / José María) — la diferencia entre variantes está en la loss,
    no en la arquitectura, para que la comparación de MSE sea justa.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128, 64)):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_dims]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int, output_dim: int, variant: str = "baseline") -> nn.Module:
    """Punto único de construcción del modelo, por si alguna variante necesita
    una arquitectura distinta (p. ej. la variante 'pinn_alt' de José María, que
    según el plan explora una arquitectura con restricciones físicas 'duras').
    De momento todas usan MLP; ajustad aquí si cambiáis de arquitectura.
    """
    if variant in ("baseline", "pinn_soft", "pinn_alt"):
        return MLP(input_dim, output_dim)
    raise ValueError(f"Variante desconocida: {variant}")
