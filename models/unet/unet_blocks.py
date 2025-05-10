import torch
import torch.nn as nn


class TimeResidual(nn.Module):
    """
    Time residual block for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
        )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_channels),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
        )

    def forward(self, x, t):
        residual = x

        x = self.conv_block1(x)
        time = self.time_emb(t)
        x = x + time.unsqueeze(-1).unsqueeze(-1)
        x = self.conv_block2(x)

        return x + residual


class UNetDownBlock(nn.Module):
    """
    Downsample block for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        downsample: bool = True
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        )
        self.time_res_block = TimeResidual(
            in_channels=out_channels,
            time_emb_dim=time_emb_dim,
        )

        self.down_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        ) if downsample else nn.Identity()

    def forward(self, x, t):
        x = self.conv_block(x)
        x = self.time_res_block(x, t)

        skip = x
        x = self.down_block(x)
        return x, skip


class UNetUpBlock(nn.Module):
    """
    Upsample block for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        upsample: bool = True
    ):
        super().__init__()

        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
        ) if upsample else nn.Identity()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(out_channels // 8, 1), out_channels),
            nn.SiLU(),
        )
        self.time_res_block = TimeResidual(
            in_channels=out_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x, skip, t):
        x = self.up_block(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        x = self.time_res_block(x, t)

        return x


class UNetOutBlock(nn.Module):
    """
    Output block for the UNet model.
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
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Bottleneck block for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(in_channels // 8, 1), in_channels),
            nn.SiLU(),
        )

        self.time_res_block = TimeResidual(
            in_channels=in_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x, t):
        x = self.conv_block(x)
        x = self.time_res_block(x, t)

        return x
