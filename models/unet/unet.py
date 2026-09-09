import torch
import torch.nn as nn

from utils.hardware.hardware_utils import select_device

from .unet_encoder_decoder import UNetBottleneck, UNetDecoder, UNetEncoder


class UNet(nn.Module):
    """
    UNet.
    """

    # ===== Initialization =====
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        time_emb_dim: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Initialize the model device
        self.device = select_device(device)

        # Define the encoder, bottleneck, and decoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )
        self.bottleneck = UNetBottleneck(
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )
        self.decoder = UNetDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )

        # Move model to the specified device
        self.to(self.device)

    # ===== Core Operations =====
    def forward(self, x, t):
        x, skips = self.encoder(x, t)
        x = self.bottleneck(x, t)
        x = self.decoder(x, skips, t)
        return x
