"""Required functions for the training and inference of the model."""

from __future__ import annotations

import torch


def get_x_t(
    x_0: torch.Tensor,
    alphas_cumpod: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get x_t from x_0.

    Returns
    -------
    torch.Tensor x_t
    torch.Tensor noise

    """
    gathered_alphas = alphas_cumpod.gather(-1, t)

    noise = torch.normal(0, 1, x_0.shape)

    cumprod_alphas = gathered_alphas.view(-1, 1, 1, 1)
    return (
        torch.sqrt(cumprod_alphas) * x_0 + torch.sqrt(1 - cumprod_alphas) * noise,
        noise,
    )


def denoise(  # noqa: PLR0913
    predicted_noise: torch.Tensor,
    alphas: torch.Tensor,
    cumprod_alphas: torch.Tensor,
    betas: torch.Tensor,
    t: torch.Tensor,
    x_t: torch.Tensor,
) -> torch.Tensor:
    """Denoise the predicted noise.

    Returns
    -------
    torch.Tensor denoised image (x_t-1)

    """
    gathered_cumprod_alphas = cumprod_alphas.gather(-1, t)
    gathered_cumprod_alphas_t_min_1 = cumprod_alphas.gather(-1, t - 1)
    gathered_alphas = alphas.gather(-1, t)

    gathered_betas = betas.gather(-1, t)
    random_noise = torch.normal(0, 1, x_t.shape)

    # first_prod - the first term in the equation
    first_prod = torch.ones_like(gathered_alphas) / torch.sqrt(gathered_alphas)
    first_prod = first_prod.view(-1, 1, 1, 1)

    # eps_prod - the second term in the equation that is multiplied with predicted noise
    eps_prod = (torch.ones_like(gathered_alphas) - gathered_alphas) / torch.sqrt(
        torch.ones_like(gathered_cumprod_alphas) - gathered_cumprod_alphas,
    )
    eps_prod = eps_prod.view(-1, 1, 1, 1)

    # target_beta - the third term in the equation that adds the predicted image
    target_beta = (
        (1 - gathered_cumprod_alphas_t_min_1)
        / (1 - gathered_cumprod_alphas)
        * gathered_betas
    )
    target_beta = target_beta.view(-1, 1, 1, 1)

    return first_prod * (x_t - eps_prod * predicted_noise) + target_beta * random_noise


def loss_function(model_predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss function."""
    return torch.nn.functional.l1_loss(model_predict, target)
