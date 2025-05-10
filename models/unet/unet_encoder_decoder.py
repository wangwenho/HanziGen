import torch
import torch.nn as nn

from .unet_blocks import BottleneckBlock, UNetDownBlock, UNetOutBlock, UNetUpBlock


class UNetEncoder(nn.Module):
    """
    Encoder for the UNet model.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        ch_multipliers = [1, 2, 4, 8]
        channels = [in_channels] + [base_channels * m for m in ch_multipliers]

        self.encoder_blocks = nn.ModuleList(
            [
                UNetDownBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    time_emb_dim=time_emb_dim,
                    downsample=False if i == 0 else True,
                )
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x, t):
        skips = []
        for block in self.encoder_blocks:
            x, skip = block(x, t)
            skips.append(skip)
        return x, skips


class UNetDecoder(nn.Module):
    """
    Decoder for the UNet model.
    """

    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        ch_multipliers = [8, 4, 2, 1]
        channels = [base_channels * m for m in ch_multipliers] + [base_channels]

        self.decoder_blocks = nn.ModuleList(
            [
                UNetUpBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    time_emb_dim=time_emb_dim,
                    upsample=False if i == len(channels) - 2 else True,
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.output_block = UNetOutBlock(
            in_channels=channels[-1],
            out_channels=out_channels,
        )

    def forward(self, x, skips, t):
        for block in self.decoder_blocks:
            x = block(x, skips.pop(), t)
        x = self.output_block(x)
        return x


class UNetBottleneck(nn.Module):
    """
    Bottleneck for the UNet model.
    """

    def __init__(
        self,
        base_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        channels = base_channels * 8

        self.bottleneck_blocks = nn.ModuleList(
            [
                BottleneckBlock(
                    in_channels=channels,
                    time_emb_dim=time_emb_dim,
                )
                for _ in range(3)
            ]
        )

    def forward(self, x, t):
        for block in self.bottleneck_blocks:
            x = block(x, t)
        return x
