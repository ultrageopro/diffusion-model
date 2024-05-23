"""Train model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import tqdm
from modules.functions import get_x_t, loss_function

if TYPE_CHECKING:
    from modules.variables import Variables


class Trainer:
    """Trainer."""

    def __init__(
        self: Trainer,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        variables: Variables,
        steps: int,
    ) -> None:
        """Initialize."""
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.alphas_cumprod = variables.alphas_cumprod

        self.save_path: str | None

    def set_params(
        self: Trainer,
        save_path: str | None = None,
    ) -> None:
        """Set required params."""
        self.save_path = save_path

    def train(
        self: Trainer,
        epochs: int,
        batch_size: int,
        dataloader: torch.utils.data.DataLoader,
    ) -> list[float]:
        """Train model."""
        loss_history = []
        logging.info("Start training")

        for epoch in range(epochs):
            bar = tqdm.tqdm(dataloader)

            for x, _ in bar:
                self.optimizer.zero_grad()

                try:
                    t = torch.randint(0, self.steps, (batch_size,))
                    x_t, noise = get_x_t(x, self.alphas_cumprod, t)

                    model_pred = self.model(x_t, t)
                    loss = loss_function(model_pred, noise)
                except RuntimeError:
                    continue

                loss.backward()
                self.optimizer.step()

                loss_history.append(loss.item())
                bar.set_description(
                    f"ep {epoch+1}/{epochs} | loss: {loss.item():.4f}",
                )

        logging.info("Finish training\nSaving model to %s", self.save_path)
        torch.save(self.model, self.save_path) if self.save_path else None
        return loss_history
