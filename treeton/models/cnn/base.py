from collections.abc import Callable, Iterable

from torch import Tensor, nn

from ..mlp.base import (
    DEFAULT_ACTIVATION,
    DEFAULT_OUTPUT_ACTIVATION,
    resolve_activation,
)


class CNN2D(nn.Module):
    """Simple composable 2D convolutional network."""

    def __init__(
        self,
        in_channels: int,
        out_features: int = 1,
        channels: int | Iterable[int] = 32,
        hidden_dims: int | Iterable[int] = 64,
        kernel_size: int = 3,
        padding: int | str = "same",
        pool_size: int = 2,
        activation: str | Callable = DEFAULT_ACTIVATION,
        final_activation: str | Callable = DEFAULT_OUTPUT_ACTIVATION,
        dropout: float = 0.,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(channels, int):
            channels = (channels,)
        else:
            channels = tuple(channels)

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        else:
            hidden_dims = tuple(hidden_dims)

        if in_channels <= 0:
            raise ValueError(
                f"`input_channels` must be positive, but was {in_channels}."
            )

        if out_features <= 0:
            raise ValueError(
                f"`output_dim` must be positive, but was {out_features}."
            )

        if any(
            width <= 0
            for width in channels
        ):
            raise ValueError(
                f"`channels` must contain only positive value, but was {channels}."
            )

        if any(
            width <= 0
            for width in hidden_dims
        ):
            raise ValueError(
                f"`hidden_dims` must contain only positive values, but was {hidden_dims}."
            )

        if not 0. <= dropout < 1.:
            raise ValueError(
                f"`dropout` must satisfy 0 <= dropout < 1, but was {dropout}."
            )

        if padding in ("same", "valid"):
            pass
        elif not isinstance(padding, int):
            raise ValueError(
                "`padding` must be an integer, "
                "'same', or 'valid'."
            )

        activation_cls = resolve_activation(activation)
        final_activation_cls = (resolve_activation(final_activation))

        conv_layers = []
        previous_channels = in_channels

        for width in channels:
            conv_layers.append(
                nn.Conv2d(
                    previous_channels,
                    width,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            if batch_norm:
                conv_layers.append(
                    nn.BatchNorm2d(width)
                )

            conv_layers.append(
                activation_cls()
            )

            if pool_size > 1:
                conv_layers.append(
                    nn.MaxPool2d(pool_size)
                )

            if dropout > 0.:
                conv_layers.append(
                    nn.Dropout2d(dropout)
                )

            previous_channels = width

        head_layers = [
            nn.Flatten(),
        ]

        previous_dim = None
        for width in hidden_dims:
            if previous_dim is None:
                head_layers.append(
                    nn.LazyLinear(width)
                )
            else:
                head_layers.append(
                    nn.Linear(
                        previous_dim,
                        width,
                    )
                )
            head_layers.append(
                activation_cls()
            )
            if dropout > 0.:
                head_layers.append(
                    nn.Dropout(dropout)
                )
            previous_dim = width

        if previous_dim is None:
            head_layers.append(
                nn.LazyLinear(output_dim)
            )
        else:
            head_layers.append(
                nn.Linear(
                    previous_dim,
                    out_features,
                )
            )

        head_layers.append(
            final_activation_cls()
        )

        self.features = nn.Sequential(
            *conv_layers
        )
        self.head = nn.Sequential(
            *head_layers
        )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.head(self.features(input))
