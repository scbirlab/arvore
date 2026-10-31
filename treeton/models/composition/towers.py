"""Multi-tower architecture."""

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn


def _merge_concat(*inputs) -> Tensor:
    return torch.cat(inputs, dim=-1)


def _merge_sum(*inputs) -> Tensor:
    return torch.stack(inputs, dim=-1).sum(dim=-1)


def _merge_product(*inputs) -> Tensor:
    return torch.stack(inputs, dim=-1).prod(dim=-1)


_MERGE_FUNCTIONS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "concat": _merge_concat,
    "sum": _merge_sum,
    "product": _merge_product,
}

DEFAULT_MERGE = "concat"

def resolve_merge(x: str | Callable):
    if isinstance(x, Callable):
        return x
    elif isinstance(x, str):
        try:
            merge_fn = _MERGE_FUNCTIONS[x.casefold()]
        except KeyError as e:
            choices = ", ".join(sorted(_MERGE_FUNCTIONS))
            raise ValueError(
                f"Unknown merge method {x!r}. "
                f"Available methods: {choices}."
            ) from e
        else:
            return merge_fn
    else:
        raise ValueError(f"Merge function must be string or callable, but was {type(x)}: {x}")


class MultiTower(nn.Module):
    """Compose encoders with a fusion module.

    Each tower maps one input modality to a latent representation. The latent
    representations are merged and passed through ``fusion``.

    Parameters
    ==========
    towers
        Module(s) receiving the inputs.
    fusion
        Module receiving the merged tower representations.
    merge
        How to combine tower outputs. One of ``concat``, ``sum``, or
        ``product``.
    """

    def __init__(
        self,
        towers: nn.Module | Iterable[nn.Module] | dict[str, nn.Module],
        fusion: nn.Module,
        merge: str = DEFAULT_MERGE
    ) -> None:
        super().__init__()
        if isinstance(towers, nn.Module):
            towers = [towers]
        if not isinstance(towers, (list, tuple, dict)):
            raise ValueError(f"`towers` must be a list, tuple, or dict of nn.Module, but was {towers}")
        if not isinstance(towers, dict):
            towers = {f"tower_{i}": tower for i, tower in enumerate(towers)}
        self.towers = nn.ModuleDict(towers)
        self.fusion = fusion
        self.merge = merge.casefold()
        self._merge = resolve_merge(merge)

    def forward(
        self,
        **inputs
    ) -> Tensor:
        if len(inputs) != len(self.towers):
            raise AttributeError(f"Length of inputs ({len(inputs)}) must be the same as number of towers ({len(self.towers)}).")
        if not set(inputs) == set(self.towers):
            raise KeyError(
                "Mismatched names between inputs and towers:\n\t",
                f". {inputs.keys()=}\n\t"
                f". {self.towers.keys()=}\n\t"
                f". {(set(inputs) - set(self.towers))=}\n\t"
                f". {(set(self.towers) - set(inputs))=}\n\t"
            )
        tower_out = []
        for name, tower in self.towers.items():
            tower_out.append(
                tower(inputs[name])
            )
        merged = self._merge(*tower_out)
        return self.fusion(merged)


class TwoTower(MultiTower):
    """Compose two encoders with a fusion module.

    Each tower maps one input modality to a latent representation. The latent
    representations are merged and passed through ``fusion``.

    Parameters
    ==========
    left
        Module receiving the ``left`` input.
    right
        Module receiving the ``right`` input.
    fusion
        Module receiving the merged tower representations.
    merge
        How to combine tower outputs. One of ``concat``, ``sum``, or
        ``product``.
    """

    def __init__(
        self,
        left: nn.Module,
        right: nn.Module,
        fusion: nn.Module,
        merge: str = DEFAULT_MERGE
    ) -> None:
        super().__init__(towers={"left": left, "right": right}, fusion=fusion, merge=merge)
    
    @property
    def left(self):
        return self.towers["left"]

    @property
    def right(self):
        return self.towers["right"]

    def forward(self, left, right) -> Tensor:
        return super().forward(left=left, right=right)
