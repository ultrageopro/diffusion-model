"""Variables class."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Variables:
    """Variables class."""

    def __init__(
        self: Variables,
        alphas: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        betas: torch.Tensor,
    ) -> None:
        """Initialize."""
        self.__alphas = alphas
        self.__betas = betas
        self.__alphas_cumprod = alphas_cumprod

    @property
    def alphas(self: Variables) -> torch.Tensor:
        """Get alphas."""
        return self.__alphas

    @property
    def betas(self: Variables) -> torch.Tensor:
        """Get betas."""
        return self.__betas

    @property
    def alphas_cumprod(self: Variables) -> torch.Tensor:
        """Get alphas cumprod."""
        return self.__alphas_cumprod
