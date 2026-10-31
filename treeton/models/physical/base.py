
from collections.abc import Callable, Iterable, Mapping
import inspect
from inspect import signature

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _param_check(
    params: str | Iterable[str] | None, 
    allowed_params: Iterable[str]
) -> tuple[tuple, list]:
    if params is None:
        params = tuple()
    elif isinstance(params, str):
        params = (params,)
    elif isinstance(params, (dict, list)):
        params = tuple(params)
    if not isinstance(params, (tuple, list)):
        raise ValueError(
            "If provided, params must be string, list or tuple, "
            f"but was {type(params)}: {params}"
        )
    disallowed_params = set(params) - set(allowed_params)
    if disallowed_params:
        raise ValueError(
            "Params named that aren't used by the function:\n\t"
            f"- {disallowed_params=}\n\t"
            f"- {params=}\n\t"
            f"- {allowed_params=}\n\t"
        )
    return params, [p for p in allowed_params if p not in params]


def _init_trainable(
    *shape, 
    initializer: Callable[[Tensor], ...] = nn.init.xavier_normal_,
    **kwargs
):
    x = torch.empty(*shape)
    initializer(x)
    return x


class FixedParams(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._keys = tuple(kwargs)
        for key, value in kwargs.items():
            self.register_buffer(
                key,
                torch.tensor(float(value)),
            )

    def keys(self) -> tuple[str]:
        return self._keys

    def __getitem__(self, key: str):
        if key in self._keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Fixed value `{key}` is not defined")

    def __iter__(self):
        for key in self._keys:
            yield key


class PhysicalModel(nn.Module):

    def __init__(
        self,
        fn: Callable,
        fixed_params: Iterable[str] | Mapping[str, Tensor] | None = None,
        trainable_params: Iterable[str]| Mapping[str, Tensor] | None = None,
        latent_params: str | Iterable[str] | None = None,
        context_params: str | Iterable[str] | None = None,
        transforms: Mapping[str, Callable[[Tensor], Tensor]] | None = None,
        default_init_value: float | int | None = None,
        default_fixed_value: float | int | None = .0001
    ):
        super().__init__()
        if not callable(fn):
            raise ValueError(
                "PhysicalModel function must be callable, "
                f"but was {type(fn)}: {fn}"
            )
        self.fn = fn
        self.all_params = {
            name: param.default
            for name, param in dict(
                signature(self.fn).parameters
            ).items()
        }
        self.required_params = tuple(
            p for p, v in self.all_params.items()
            if v is inspect._empty
        )
        
        self.fixed_params, remaining_params = _param_check(
            fixed_params, 
            list(self.all_params),
        )

        if isinstance(fixed_params, Mapping):
            self.fixed_param_values = dict(fixed_params)
        else:
            self.fixed_param_values = {
                name: self.all_params[name] 
                if self.all_params[name] is not inspect._empty
                else default_fixed_value
                for name in self.fixed_params
            }
        
        self.trainable_params, remaining_params = _param_check(
            trainable_params, 
            remaining_params,
        )
        if isinstance(trainable_params, Mapping):
            self.trainable_params_init = trainable_params
        else:
            self.trainable_params_init = {
                name: default_init_value or _init_trainable(1)
                for name in self.trainable_params
            }

        self.context_params, remaining_params = _param_check(
            context_params, 
            remaining_params,
        )

        if latent_params is None:
            remaining_required_params = set(self.required_params) & set(remaining_params)
            self.latent_params = tuple(remaining_required_params)
            remaining_params = [p for p in remaining_params if p not in self.latent_params]
        else:
            self.latent_params, remaining_params = _param_check(
                latent_params, 
                remaining_params,
            )            
        
        remaining_required_params = set(self.required_params) & set(remaining_params)
        if len(remaining_required_params) > 0:
            raise ValueError(
                f"Function parameters unaccounted for: {remaining_required_params}\n\t"
                f"- {self.fixed_params=}\n\t"
                f"- {self.trainable_params=}\n\t"
                f"- {self.context_params=}\n\t"
                f"- {self.latent_params=}\n\t"
                f"- {self.required_params=}\n\t"
            )

        if transforms is None:
            transforms = {}
        _, _ = _param_check(transforms, self.latent_params + self.trainable_params)
        for name, transform in transforms.items():
            if not callable(transform):
                raise AttributeError(
                    f"transform for {name} must be callable, "
                    f"but was {type(transform)}: {transform}"
                )
        self.transforms = transforms
        self._fixed_params = None
        self._trainable_params = None
        self._built = False

        self.build()

    def build(self) -> None:
        if self._built:
            return None

        self._fixed_params = FixedParams(**self.fixed_param_values)
        self._trainable_params = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(float(value)).clone().detach())
            for key, value in self.trainable_params_init.items()
        })
        self._built = True

    def forward(
        self,
        latent: Tensor | None = None,
        context: Tensor | None = None
    ) -> Tensor:
        self.build()
        if latent is None and len(self.latent_params) > 0:
             raise ValueError(
                "This model requires latent params\n\t"
                f"- {len(self.latent_params)=}"
            )
        if context is None and len(self.context_params) > 0:
             raise ValueError(
                "This model requires context params\n\t"
                f"- {len(self.context_params)=}"
            )

        if latent is not None:
            latent_dim = latent.shape[-1]
            if not latent_dim == len(self.latent_params):
                raise ValueError(
                    "Latent dimension must be the same as the number of latent_params\n\t"
                    f"- {latent.shape=}"
                    f"- {len(self.latent_params)=}"
                )
            latent = dict(zip(self.latent_params, torch.split(latent, 1, dim=-1)))
        else:
            latent = {}
                
        if context is not None:
            context_dim = context.shape[-1]
            if not context_dim == len(self.context_params):
                raise ValueError(
                    "Latent dimension must be the same as the number of context_params\n\t"
                    f"- {context.shape=}"
                    f"- {len(self.context_params)=}"
                )
            context = dict(zip(self.context_params, torch.split(context, 1, dim=-1)))
        else:
            context = {}

        transformed = {
            name: self.transforms[name](val) if name in self.transforms
            else val 
            for name, val in (latent | self._trainable_params).items()
        }
        
        response = self.fn(
            **context,
            **transformed, 
            **self.fixed_param_values,
        )
        if response.ndim < 2:
            response = response.unsqueeze(-1)
        return response
