import torch
import torch.nn as nn


class VQVAEDownBlock(nn.Module):
    """
    Down-sample block for the VQ-VAE model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.down_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.down_block(x)
        return x


class VQVAEUpBlock(nn.Module):
    """
    Up-sample block for the VQ-VAE model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.up_block(x)
        return x


class VQVAEOutBlock(nn.Module):
    """
    Output block for the VQ-VAE model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x
