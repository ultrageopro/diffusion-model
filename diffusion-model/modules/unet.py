"""Unet."""

from __future__ import annotations

import math

import torch
from torch import nn


class Block(nn.Module):
    """Unet block."""

    def __init__(
        self: Block,
        in_ch: int,
        out_ch: int,
        time_emb_dim: torch.Tensor,
        *,
        up: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(
        self: Block,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process."""
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    """Time embedding."""

    def __init__(self: SinusoidalPositionEmbeddings, dim: int) -> None:
        """Initialize."""
        super().__init__()
        self.dim = dim

    def forward(self: SinusoidalPositionEmbeddings, time: torch.Tensor) -> torch.Tensor:
        """Forward process."""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(1000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class SimpleUnet(nn.Module):
    """A simplified variant of the Unet architecture."""

    def __init__(self: SimpleUnet, image_size: int, image_channels: int) -> None:
        """Init."""
        super().__init__()
        out_dim = image_channels
        time_emb_dim = 32
        down_channels = (image_size, 64, 128, 256, 512)
        up_channels = (512, 256, 128, 64, image_size)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim)
                for i in range(len(down_channels) - 1)
            ],
        )
        # Upsample
        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
                for i in range(len(up_channels) - 1)
            ],
        )

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(
        self: SimpleUnet,
        x: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward."""
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
