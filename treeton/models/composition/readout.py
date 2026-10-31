"""Data generating models to map learned parameters to physical measurements."""

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn


class Readout(nn.Module):
    def __init__(
        self,
        latent: nn.Module,
        readout: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.latent = latent
        self.readout = readout

    def forward(
        self,
        context=None,
        **inputs
    ):
        latent = self.latent(**inputs)
        if self.readout is None:
            return latent

        if context is None:
            return self.readout(latent)

        return self.readout(
            latent,
            context,
        )


