import torch
import torch.nn as nn

from utils.hardware.hardware_utils import select_device

from .unet_encoder_decoder import UNetBottleneck, UNetDecoder, UNetEncoder


class UNet(nn.Module):
    """
    UNet model with encoder, bottleneck, and decoder components.

    - Encoder: Downsampling blocks with optional self-attention and cross-attention.
    - Bottleneck: Intermediate processing block.
    - Decoder: Upsampling blocks with optional self-attention and cross-attention.

    Args:
        model_config: Model configuration.
        device: Device to run the model on (e.g., 'mps', 'cuda', 'cpu').
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
