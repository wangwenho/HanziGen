import math

import torch
from torch import Tensor

from utils.hardware.hardware_utils import select_device


class SigmoidScheduler:
    """
    Sigmoid scheduler for diffusion models.
    """

    def __init__(
        self,
        noise_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        sigmoid_start: int = -6,
        sigmoid_end: int = 6,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.device = select_device(device)
        self.noise_steps = noise_steps

        sigmoid_range = torch.linspace(
            sigmoid_start, sigmoid_end, noise_steps, device=self.device
        )
        self.betas = torch.sigmoid(sigmoid_range) * (beta_end - beta_start) + beta_start

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.timesteps = None

    # ===== Training Setup Operations =====
    def sample_timesteps(
        self,
        batch_size: int,
    ) -> Tensor:
        """
        Sample timesteps for training.
        """
        return torch.randint(1, self.noise_steps, (batch_size,), device=self.device)

    # ===== Training Noise Operations =====
    def add_noise(
        self,
        x: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Add noise to the input x at time t.
        """
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(
            -1, 1, 1, 1
        )

        epsilon = torch.randn_like(x, device=self.device)
        x = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * epsilon

        return x, epsilon

    # ===== Inference Setup Operations =====
    def set_timesteps(
        self,
        inference_steps: int,
    ) -> None:
        """
        Set timesteps for inference.
        """
        step_ratio = self.noise_steps // inference_steps
        self.timesteps = (
            torch.arange(0, inference_steps, device=self.device).flip(0) * step_ratio
        )

    # ===== Denoising Step Operations =====
    def ddpm_step(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Perform a single step for DDPM.

        Args:
            x: Input latents at timestep t.
            t: Current timestep indices.
            y: Predicted noise.

        Returns:
            Tensor: Latents at previous timestep (t-1).
        """
        # Get alpha, alpha_cumprod, and beta values for the current timestep
        alphas = self.alphas[t].view(-1, 1, 1, 1)
        alphas_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        betas = self.betas[t].view(-1, 1, 1, 1)

        # Generate noise
        noise = (
            torch.randn_like(x, device=self.device)
            if t[0] > 0
            else torch.zeros_like(x, device=self.device)
        )

        # Calculate x_prev
        x_prev = (
            1
            / torch.sqrt(alphas)
            * (x - ((1.0 - alphas) / (torch.sqrt(1.0 - alphas_cumprod))) * y)
            + torch.sqrt(betas) * noise
        )

        return x_prev

    def ddim_step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        y: Tensor,
        eta: float = 0.0,
    ) -> Tensor:
        """
        Perform a single step for DDIM.

        Args:
            x: Latents at timestep t.
            t: Current timestep indices.
            t_prev: Previous timestep indices.
            y: Predicted noise.
            eta: Noise multiplier.

        Returns:
            Tensor: Latents at timestep t_prev.
        """
        # Get alpha cumprod values for current and previous timesteps
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)

        # Calculate x0
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)
        x0 = (x - sqrt_one_minus_alphas_cumprod_t * y) / sqrt_alphas_cumprod_t

        # Calculate variance and noise
        variance = eta * torch.sqrt(
            (1.0 - alphas_cumprod_t_prev)
            / (1.0 - alphas_cumprod_t)
            * (1.0 - alphas_cumprod_t / alphas_cumprod_t_prev)
        )
        noise = (
            torch.randn_like(x, device=self.device)
            if t[0] > 0
            else torch.zeros_like(x, device=self.device)
        )

        # Calculate x_prev
        direction = torch.sqrt(1.0 - alphas_cumprod_t_prev - variance**2) * y
        x_prev = torch.sqrt(alphas_cumprod_t_prev) * x0 + direction + variance * noise

        return x_prev
