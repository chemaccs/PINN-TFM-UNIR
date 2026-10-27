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

    Se usa tanto para el baseline como para las variantes PINN: la diferencia
    entre variantes está en la loss, no en la arquitectura, para que la
    comparación de MSE sea justa.

    residual=True: la red aprende la CORRECCIÓN sobre la persistencia,
        ŷ(t+1) = xmeas(t) + red(x)
    (todo en el espacio normalizado del objetivo). La última capa se inicia a
    cero, así que al empezar el modelo ya equivale a la persistencia. Sirve
    para que, fuera del régimen de entrenamiento (fallos), la salida no se
    desplome hacia la media. La loss y la evaluación no cambian.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128, 64),
                 residual: bool = False):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_dims]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.Tanh()]
        last = nn.Linear(dims[-1], output_dim)
        layers.append(last)
        self.net = nn.Sequential(*layers)

        self.residual = residual
        self.n_xmv = input_dim - output_dim
        if residual:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            # persistencia = x_xmeas * skip_w + skip_b (se fija con set_persistence)
            self.register_buffer("skip_w", torch.ones(output_dim))
            self.register_buffer("skip_b", torch.zeros(output_dim))

    def set_persistence(self, x_scaler, y_scaler) -> None:
        """Traduce xmeas(t) del espacio normalizado de la entrada al del objetivo."""
        xs, xm = x_scaler.scale_[self.n_xmv:], x_scaler.mean_[self.n_xmv:]
        ys, ym = y_scaler.scale_, y_scaler.mean_
        self.skip_w.copy_(torch.tensor(xs / ys, dtype=torch.float32))
        self.skip_b.copy_(torch.tensor((xm - ym) / ys, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            out = out + x[:, self.n_xmv:] * self.skip_w + self.skip_b
        return out


def build_model(input_dim: int, output_dim: int, variant: str = "baseline",
                residual: bool = False, x_scaler=None, y_scaler=None) -> nn.Module:
    """Punto único de construcción del modelo."""
    if variant not in ("baseline", "pinn_soft", "pinn_alt"):
        raise ValueError(f"Variante desconocida: {variant}")
    model = MLP(input_dim, output_dim, residual=residual)
    if residual and x_scaler is not None and y_scaler is not None:
        model.set_persistence(x_scaler, y_scaler)
    return model
