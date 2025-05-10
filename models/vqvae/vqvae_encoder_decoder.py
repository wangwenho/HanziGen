import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .vqvae_blocks import VQVAEDownBlock, VQVAEOutBlock, VQVAEUpBlock


class VQVAEEncoder(nn.Module):
    """
    Encoder for the VQ-VAE model.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
    ):
        super().__init__()
        ch_multipliers = [1, 2, 4]
        channels = [in_channels] + [base_channels * m for m in ch_multipliers]

        self.encoder_blocks = nn.ModuleList(
            [
                VQVAEDownBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.encoder_blocks.append(
            nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


class VQVAEDecoder(nn.Module):
    """
    Decoder for the VQ-VAE model.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
    ):
        super().__init__()

        ch_multipliers = [4, 2, 1]
        channels = [base_channels * m for m in ch_multipliers] + [out_channels]

        self.decoder_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)]
        )
        self.decoder_blocks.extend(
            [
                VQVAEUpBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.decoder_blocks.append(
            VQVAEOutBlock(
                in_channels=channels[-1],
                out_channels=out_channels,
            )
        )

    def forward(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return x


class VQVAEQuantizer(nn.Module):
    """
    Vector Quantizer for the VQ-VAE model.
    """

    def __init__(
        self,
        codebook_size: int,
        latent_dim: int,
        commitment_cost: float,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost

        self.emb = nn.Embedding(codebook_size, latent_dim)
        self.emb.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x):
        x_perm = rearrange(x, "b c h w -> b h w c")
        b, h, w, c = x_perm.shape

        flat_x = rearrange(x_perm, "b h w c -> (b h w) c")

        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.emb.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.emb.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        quantized = self.emb(encoding_indices.squeeze(1))
        quantized = rearrange(quantized, "(b h w) c -> b h w c", b=b, h=h, w=w)

        e_latent_loss = F.mse_loss(quantized.detach(), x_perm)
        q_latent_loss = F.mse_loss(quantized, x_perm.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x_perm + (quantized - x_perm).detach()

        quantized = rearrange(quantized, "b h w c -> b c h w")

        return quantized, loss
