"""Main code."""

import logging
from pathlib import Path

import torch
import yaml
from modules.dataset import get_dataloader
from modules.train import Trainer
from modules.unet import SimpleUnet
from modules.variables import Variables

# Get the parameters from the config file
with Path("diffusion-model/configs.yml").open() as config_file:
    configs = yaml.safe_load(config_file)

# Get the parameters from the config file
lr = configs["model"]["lr"]
save_path = configs["model"]["save_path"]

epochs = configs["training"]["epochs"]
batch_size = configs["training"]["batch_size"]

alphas_min = configs["diffusion"]["alphas_min"]
alphas_max = configs["diffusion"]["alphas_max"]
steps = configs["diffusion"]["steps"]

image_size = configs["image"]["image_size"]
num_channels = configs["image"]["num_channels"]

# Create the alphas
alphas = torch.linspace(alphas_min, alphas_max, steps)
alphas_cumprod = torch.cumprod(alphas, dim=0)
betas = 1 - alphas

variables = Variables(alphas, alphas_cumprod, betas)

# Create and configure the model
model = SimpleUnet(image_size, num_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
dataloader = get_dataloader(batch_size, image_size, 8)

# Create the trainer
trainer = Trainer(model, optimizer, variables, steps)
trainer.set_params(save_path=save_path)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Train the model
    loss_history = trainer.train(epochs, batch_size, dataloader)
