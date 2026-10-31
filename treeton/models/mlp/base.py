"""Composable PyTorch model primitives."""

from collections.abc import Callable, Sequence

from torch import Tensor, nn


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}
DEFAULT_HIDDEN_DIMS = 16
DEFAULT_ACTIVATION = "silu"
DEFAULT_OUTPUT_ACTIVATION = "linear"


def resolve_activation(x: str | Callable) -> nn.Module:
    if isinstance(x, Callable):
        return x
    elif isinstance(x, str):
        try:
            activation = _ACTIVATIONS[x.casefold()]
        except KeyError as e:
            choices = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(
                f"Unknown activation {x!r}. "
                f"Available activations: {choices}."
            ) from e
        return activation
    else:
        raise ValueError(f"Activation must be string or callable, but was {type(x)}: {x}")


class MLP(nn.Module):
    """A multilayer perceptron."""

    def __init__(
        self,
        out_features: int = 1,
        hidden_dims: int | Sequence[int] = DEFAULT_HIDDEN_DIMS,
        in_features: int | None = None,
        activation: str = DEFAULT_ACTIVATION,
        final_activation: str = DEFAULT_OUTPUT_ACTIVATION,
        dropout: float = 0.,
        batch_norm: bool = False
    ):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        else:
            hidden_dims = tuple(hidden_dims)

        if in_features is not None and in_features <= 0:
            raise ValueError(
                f"`in_features` must be positive when provided, but was {in_features}."
            )

        if out_features <= 0:
            raise ValueError(f"`out_features` must be positive, but was {out_features}.")

        if any(width <= 0 for width in hidden_dims):
            raise ValueError(
                f"All hidden dimensions must be positive, but were {hidden_dims=}."
            )

        if not 0. <= dropout < 1.:
            raise ValueError(
                f"`dropout` must satisfy 0 <= dropout < 1, but was {dropout}."
            )

        self.input_dim = in_features
        self.output_dim = out_features
        self.hidden_dims = hidden_dims
        self.activation = resolve_activation(activation)
        self.final_activation = resolve_activation(final_activation)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layers = None
        self._built = False

        self.build()

    def build(self) -> None:
        if self._built:
            return None
        widths = (*self.hidden_dims, self.output_dim)

        layers: list[nn.Module] = []
        previous_dim = self.input_dim

        end_idx = (len(widths) - 1)
        for i, width in enumerate(widths):
            is_output = i == end_idx
            if previous_dim is None:
                layers.append(nn.LazyLinear(width))
            else:
                layers.append(
                    nn.Linear(previous_dim, width)
                )

            if not is_output:
                if self.batch_norm:
                    layers.append(
                        nn.BatchNorm1d(width)
                    )

                layers.append(
                    self.activation()
                )

                if self.dropout > 0.:
                    layers.append(
                        nn.Dropout(self.dropout)
                    )

            else:
                layers.append(
                    self.final_activation()
                )

            previous_dim = width

        self.layers = nn.Sequential(*layers)
        self._built = True

    def forward(self, x: Tensor) -> Tensor:
        self.build()
        return self.layers(x)
