import math

import torch
from torch import nn

from csi_vae.jobs.vae.gaussian import SingleAntenna

_N_DIMS_MULTIPLE_ANTENNAS: int = 4

def _next_multiple_of_8(n: int) -> int:
    """Round n up to the next multiple of 8."""
    return math.ceil(n / 8) * 8


def _build_fc(in_dim: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    """Build FC block with n_layers, keeping hidden dims as multiples of 8."""
    if n_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    # Geometrically interpolate hidden dims, only internal ones must be multiples of 8
    dims = (
        [in_dim]
        + [_next_multiple_of_8(int(in_dim * ((out_dim / in_dim) ** (i / (n_layers - 1))))) for i in range(1, n_layers)]
        + [out_dim]
    )

    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < n_layers - 1:
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.GELU())

    return nn.Sequential(*layers)


class Delayed(nn.Module):
    """Delayed fusion module for multi-antenna CSI data."""

    def __init__(
        self,
        antennas: list[SingleAntenna],
        n_gaussians: int,
        n_activities: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the delayed fusion module."""
        super().__init__()

        self.__antennas = nn.ModuleList(antennas)
        for param in self.__antennas.parameters():
            param.requires_grad = False

        self.__fc = _build_fc(n_gaussians * 2 * len(antennas), n_activities, n_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the delayed fusion module.

        Arguments:
            x: Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers) if num_antennas > 1,
                or (batch_size, window_size, n_subcarriers) if num_antennas=1.

        Returns:
            Output tensor of shape (batch_size, n_activities).

        """
        outs = []
        for i, antenna in enumerate(self.__antennas):
            _, mu, logvar = antenna(x[:, i] if x.ndim == _N_DIMS_MULTIPLE_ANTENNAS else x)
            outs.append(torch.cat([mu, logvar], dim=1))

        z = torch.cat(outs, dim=1)
        return self.__fc(z)
