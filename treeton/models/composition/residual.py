

from collections.abc import Callable, Mapping

from torch import Tensor, nn

from ..mlp.base import DEFAULT_ACTIVATION, resolve_activation


class ResidualBlock(nn.Module):
    """Residual composition."""

    def __init__(
        self,
        module: nn.Module,
        in_features: int | None = None,
        out_features: int | None = None,
        projection: nn.Module | None = None,
        activation: str | Callable = DEFAULT_ACTIVATION
    ):
        super().__init__()

        self.module = module
        self.activation = resolve_activation(activation)()

        if projection is not None:
            self.projection = projection
        elif (
            in_features is None
            or out_features is None
            or in_features == out_features
        ):
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(
                in_features,
                out_features,
            )

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.activation(self.module(input) + self.projection(input))


class ResidualStack(nn.Module):

    def __init__(
        self,
        module_class: Callable = nn.Linear,
        module_kwargs: Mapping[str, ...] | None = None,
        depth: int = 1,
        in_features: int | None = None,
        out_features: int | None = None,
        projection: nn.Module | None = None,
        activation: str | Callable = DEFAULT_ACTIVATION
    ):
        super().__init__()

        self.activation = resolve_activation(activation)()
        module_kwargs = dict(module_kwargs or {})
        stack = []
        next_in_features = None
        for i in range(depth - 1):
            these_in_features = in_features if i == 0 else next_in_features
            layer = module_class(**(
                module_kwargs | {"in_features": next_in_features}
            ))
            next_in_features = module_kwargs.get("out_features")
            stack.append(layer)
        layer = module_class(**(
            module_kwargs | {
                "in_features": next_in_features, 
                "out_features": out_features,
            }
        ))
        self.block = ResidualBlock(
            module=nn.Sequential(*stack),
            in_features=in_features,
            out_features=out_features,
            projection=projection,
            activation=activation,

        )

    def forward(self, input: Tensor):
        return self.block(input)


