

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base import PhysicalModel


def _hill_curve(
    conc: Tensor,
    log_ic50: Tensor,
    slope: Tensor = 1.,
    efficacy: Tensor = 1.,
    bottom: Tensor | None = None,
    top: Tensor = 1.,
    is_growth: bool = False
) -> Tensor:
    if bottom is None:
        bottom = top * (1. - efficacy)
    inhibition = bottom + (top - bottom) * torch.sigmoid(
        slope * (torch.log(conc) - log_ic50)
    )
    if is_growth:
        return top - inhibition
    else:
        return inhibition


def hill_curve(
    conc: Tensor,
    log_ic50: Tensor,
    slope: Tensor = 1.,
    efficacy: Tensor = 1.,
    bottom: Tensor | None = None,
    top: Tensor = 1.,
    is_growth: bool = False
) -> Tensor:
    if torch.any(conc <= 0):
        raise ValueError(f"Concentration must be positive, but was {conc}.")
    return _hill_curve(
        conc=conc,
        log_ic50=log_ic50,
        slope=slope,
        efficacy=efficacy,
        bottom=bottom,
        top=top,
        is_growth=is_growth,
    )


class HillCurve(PhysicalModel):
    """Map latent potency parameters and concentration to fractional response.

    Parameters
    ==========
    slope
        Shared Hill slope. If ``None``, the latent representation must contain
        both log-IC50 and an unconstrained slope parameter.
    trainable
        Whether a shared slope should be trainable.

    Notes
    =====
    The first latent value is interpreted as ln(IC50).

    With ``slope=None`` the second latent value is interpreted as an
    unconstrained compound-specific slope and transformed with softplus.

    ``context`` is concentration in the same units as IC50.

    """

    def __init__(
        self,
        slope: float = 1.,
        trainable_slope: bool = False
    ):
        if slope <= 0.:
            raise ValueError(
                f"`slope` must be positive, but was {slope}."
            )
        slope = torch.tensor(float(slope))
        if trainable_slope:
            fixed_params = None
            trainable_params = {"slope": torch.log(torch.expm1(slope))}
            transforms = {"slope": F.softplus}
        else:
            fixed_params = {"slope": slope}
            trainable_params = None
            transforms = None

        super().__init__(
            fn=_hill_curve,
            fixed_params=fixed_params,
            trainable_params=trainable_params,
            latent_params="log_ic50",
            context_params="conc",
            transforms=transforms,
        )
