"""Feature-wise linear modulation."""

import torch
from torch import Tensor, nn


class FiLM(nn.Module):
    """Modulate features using parameters predicted from context.

    The modulator must produce twice the feature width, interpreted as
    concatenated gamma and beta parameters.

    Parameters
    ----------
    modulator
        Module mapping context to concatenated gamma and beta.
    soft
        Bound gamma to (0, 2) and beta to (-1, 1). Zero-valued raw
        modulation then corresponds to the identity transformation.
    """

    def __init__(
        self,
        modulator: nn.Module,
        soft: bool = False,
    ):
        super().__init__()
        self.modulator = modulator
        self.soft = soft

    def forward(
        self,
        input: Tensor,
        context: Tensor
    ) -> Tensor:
        modulation = self.modulator(context)

        if modulation.shape[-1] % 2:
            raise ValueError(
                "FiLM modulator output width must be even, "
                f"but was {modulation.shape[-1]}."
            )

        gamma, beta = torch.chunk(modulation, 2, dim=-1)
        if gamma.shape[-1] != input.shape[-1]:
            raise ValueError(
                "FiLM modulation width must match input width: "
                f"got {gamma.shape[-1]} and {input.shape[-1]}."
            )

        if self.soft:
            gamma = 2. * torch.sigmoid(gamma)
            beta = torch.tanh(beta)

        return gamma * input + beta
