"""Visualisation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from modules.functions import denoise
from torchvision import transforms
from tqdm import tqdm

if TYPE_CHECKING:
    from modules.unet import SimpleUnet
    from modules.variables import Variables


class Visualisation:
    """Visualisation class."""

    def __init__(
        self: Visualisation,
        model_path: str,
        variables: Variables,
        shape: tuple,
        steps: int,
    ) -> None:
        """Initialize."""
        self.model: SimpleUnet = torch.load(model_path)
        self.alphas = variables.alphas
        self.alphas_cumprod = variables.alphas_cumprod
        self.betas = variables.betas
        self.steps = steps
        self.shape = shape
        self.reverse_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                transforms.Lambda(lambda t: t * 255.0),
                transforms.Lambda(lambda t: t.detach().numpy().astype(np.uint8)),
                transforms.ToPILImage(),
            ],
        )

    def timestep(
        self: Visualisation,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Make one step of denoising process."""
        model_prediction = self.model(x_t, t)
        return denoise(
            model_prediction,
            self.alphas,
            self.alphas_cumprod,
            self.betas,
            t,
            x_t,
        )

    def get_image(self: Visualisation) -> torch.Tensor:
        """Generate image."""
        random_noise = torch.normal(0, 1, self.shape)
        logging.info("Starting denoising process...")

        for i in tqdm(range(1, self.steps)[::-1]):
            t = torch.full((self.shape[0],), i, dtype=torch.long)
            random_noise = self.timestep(random_noise, t)
            random_noise = torch.clamp(random_noise, -1, 1)

        logging.info("Denoising process finished")

        return random_noise

    def tensor_to_image(
        self: Visualisation,
        tensor: torch.Tensor,
        save_path: str,
        text: str | None = None,
    ) -> torch.Tensor:
        """Convert tensor to image and save it."""
        fig, ax = plt.subplots(ncols=tensor.shape[0])
        fig.set_size_inches(20, 5)
        plt.title(text if text else "")

        for i in range(tensor.shape[0]):
            ax[i].imshow(self.reverse_transforms(tensor[i]), cmap="gray")
            ax[i].axis("off")

        plt.savefig(save_path)

        plt.close()
        plt.cla()
        plt.clf()

        return tensor

    def plot_loss(self: Visualisation, loss: list[float], save_path: str) -> None:
        """Plot loss graph."""
        plt.plot(loss)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(save_path)

        plt.close()
        plt.cla()
        plt.clf()

    def save_denoising_process(self: Visualisation, save_path: str) -> None:
        """Save denoising process."""
        animation_shape = (1, self.shape[1], self.shape[2], self.shape[3])
        random_noise = torch.normal(0, 1, animation_shape)
        frames = [random_noise]

        logging.info("Starting denoising process...")
        for i in tqdm(range(1, self.steps)[::-1]):
            t = torch.full((1,), i, dtype=torch.long)
            random_noise = self.timestep(random_noise, t)
            random_noise = torch.clamp(random_noise, -1, 1)

            frames.append(random_noise)

        logging.info("Creating animation...")

        fig, ax = plt.subplots()

        def animate(i: int) -> None:
            """Animate."""
            ax.cla()
            ax.imshow(self.reverse_transforms(frames[i][0]), cmap="gray")
            ax.axis("off")

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(frames),
            interval=1000 / 60,
        )
        anim.save(save_path, writer="imagemagick", fps=30)
        logging.info("Animation created")
