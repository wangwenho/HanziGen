import torch
import torch.nn as nn
from torch import Tensor

from utils.hardware.hardware_utils import select_device


class TimeEmbedding(nn.Module):
    """
    Time step embedding module for diffusion models.
    """

    def __init__(
        self,
        time_pos_dim: int,
        time_emb_dim: int,
        time_steps: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.device = select_device(device)

        self.time_pos_emb = SinusoidalPositionalEncoding(
            time_pos_dim=time_pos_dim,
            time_steps=time_steps,
            device=self.device,
        )
        self.time_ff = TimeFeedForward(time_pos_dim, time_emb_dim)

        self.to(self.device)

    def forward(self, t: Tensor) -> Tensor:
        t = self.time_pos_emb(t)
        out = self.time_ff(t)
        return out


class TimeFeedForward(nn.Module):
    """
    Feed-forward network for time step embeddings.
    """

    def __init__(self, time_pos_dim: int, time_emb_dim: int):
        super().__init__()

        self.time_emb = nn.Sequential(
            nn.Linear(time_pos_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.time_emb(t)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings for time steps.
    """

    def __init__(
        self,
        time_pos_dim: int,
        time_steps: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.device = select_device(device)

        self.pos_emb = torch.zeros(time_steps, time_pos_dim, device=self.device)

        pos = torch.arange(0, time_steps, device=self.device).float().unsqueeze(1)
        div_term = 10000.0 ** (
            torch.arange(0, time_pos_dim, 2, device=self.device).float() / time_pos_dim
        )

        self.pos_emb[:, 0::2] = torch.sin(pos / div_term)
        self.pos_emb[:, 1::2] = torch.cos(pos / div_term)

    def forward(self, t: Tensor) -> Tensor:
        return self.pos_emb[t]
