from collections.abc import Iterable

import torch
from torch import Tensor, nn


class Ensemble(nn.Module):
    """Evaluate several models and stack their predictions."""

    def __init__(
        self,
        models: Iterable[nn.Module],
        dim: int = -1,
    ) -> None:
        super().__init__()

        models = list(models)

        if len(models) == 0:
            raise ValueError("`models` must contain at least one model.")

        self.models = nn.ModuleList(models)
        self.dim = dim

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Tensor:
        return torch.stack([
            model(*args, **kwargs) 
            for model in self.models
        ], dim=self.dim)