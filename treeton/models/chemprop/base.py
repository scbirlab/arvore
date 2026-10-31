"""Composable Chemprop model primitives."""

from collections.abc import Mapping

import torch
from chemprop.nn import BondMessagePassing, NormAggregation
from torch import Tensor, nn

from ..mlp.base import MLP, resolve_activation, DEFAULT_ACTIVATION, DEFAULT_HIDDEN_DIMS

DEFAULT_CHEMPROP_MP_DEPTH: int = 3 
DEFAULT_CHEMPROP_MP_HIDDEN_DIM: int = 300
DEFAULT_CHEMPROP_HIDDEN_DIM: int = DEFAULT_HIDDEN_DIMS
DEFAULT_CHEMPROP_ACTIVATION: int = DEFAULT_ACTIVATION
DEFAULT_CHEMPROP_MP_ACTIVATION: str = "elu" 

class ChempropEncoder(nn.Module):
    """Encode molecular graphs into a fixed-width latent representation.

    Expects an already-collated Chemprop-style batch with ``bmg``, ``V_d``
    and ``X_d`` attributes.

    Parameters
    ----------
    output_dim
        Width of the returned molecular representation.
    mp_hidden_dim
        Hidden width of Chemprop message passing.
    mp_depth
        Number of message-passing steps.
    mp_activation
        Chemprop activation name.
    extra_dim
        Width of optional datapoint-level ``X_d`` descriptors.
    hidden_dims
        Hidden dimensions of the projection network after graph aggregation.
    activation
        Activation used by the projection MLP.
    dropout
        Dropout used in both message passing and projection.
    batch_norm
        Apply batch normalization to the aggregated graph representation.
    """

    def __init__(
        self,
        out_features: int = 1,
        mp_hidden_dim: int = DEFAULT_CHEMPROP_MP_HIDDEN_DIM,
        mp_depth: int = DEFAULT_CHEMPROP_MP_DEPTH,
        mp_activation: str = DEFAULT_CHEMPROP_MP_ACTIVATION,
        extra_dim: int = 0,
        hidden_dims: int | tuple[int, ...] = DEFAULT_CHEMPROP_HIDDEN_DIM,
        activation: str = DEFAULT_CHEMPROP_ACTIVATION,
        final_activation: str | None = None,
        dropout: float = 0.,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.output_dim = out_features
        self.mp_hidden_dim = mp_hidden_dim
        self.mp_depth = mp_depth
        self.mp_activation = mp_activation
        if extra_dim < 0:
            raise ValueError(f"`extra_dim` cannot be negative. It was {extra_dim}.")
        self.extra_dim = extra_dim
        self.hidden_dims = hidden_dims
        self.activation = resolve_activation(activation)
        if final_activation is None:
            final_activation = self.activation
        self.final_activation = resolve_activation(final_activation)
        self.dropout = dropout
        self.batch_norm = batch_norm

        self._built = False
        self.message_passing = None
        self.projection = None
        self.aggregation = None
        self.graph_dim = None
        self.build()

    def build(self):
        if self._built:
            return None
        self.message_passing = BondMessagePassing(
            d_h=self.mp_hidden_dim,
            depth=self.mp_depth,
            activation=self.mp_activation,
            dropout=self.dropout,
        )

        self.aggregation = NormAggregation()
        self.graph_dim = self.message_passing.output_dim
        self.batch_normalization = (
            nn.BatchNorm1d(self.graph_dim)
            if self.batch_norm
            else nn.Identity()
        )
        self.projection = MLP(
            input_dim=self.graph_dim + self.extra_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation,
            final_activation=self.final_activation,
            dropout=self.dropout,
        )
        self._built = True

    def forward(
        self, 
        input: Mapping[str, ...]
    ) -> Tensor:
        self.build()
        bmg = input["bmg"]
        V_d = input.get("V_d")
        X_d = input.get("X_d")
        if self.extra_dim == 0 and X_d is not None:
            raise ValueError(
                f"Received X_d descriptors ({len(X_d)=}) but {self.extra_dim=}."
            )
        if self.extra_dim > 0 and X_d is None:
            raise ValueError(
                f"Expected X_d with width {self.extra_dim=}, "
                "but no X_d was provided."
            )
        if X_d is not None and X_d.shape[-1] != self.extra_dim:
            raise ValueError(
                f"Expected X_d width {self.extra_dim=}, "
                f"got {X_d.shape[-1]=}."
            )
        if isinstance(bmg, dict):
            _batch = bmg["batch"]
        else:
            _batch = getattr(bmg, "batch")
        h = self.message_passing(
            bmg,
            V_d,
        )
        h = self.aggregation(
            h,
            _batch,
        )
        h = self.batch_normalization(h)
        if X_d is not None:
            h = torch.cat((h, X_d), dim=-1)
        return self.projection(h)
