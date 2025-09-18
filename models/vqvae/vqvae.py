from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm

from configs.vqvae_config import VQVAEModelConfig, VQVAETrainingConfig
from datasets.loader import Loader
from utils.hardware.hardware_utils import select_device

from .vqvae_encoder_decoder import VQVAEDecoder, VQVAEEncoder, VQVAEQuantizer


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE).
    """

    # ===== Initialization & Setup =====
    def __init__(
        self,
        model_config: VQVAEModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Initialize model device
        self.device = select_device(device)

        # Define encoder, vector quantizer, and decoder
        self.encoder = VQVAEEncoder(
            in_channels=model_config.input_img_channels,
            base_channels=model_config.encoder_base_channels,
            out_channels=model_config.latent_dim,
        )
        self.vector_quantizer = VQVAEQuantizer(
            codebook_size=model_config.codebook_size,
            latent_dim=model_config.latent_dim,
            commitment_cost=model_config.commitment_cost,
        )
        self.decoder = VQVAEDecoder(
            in_channels=model_config.latent_dim,
            base_channels=model_config.encoder_base_channels,
            out_channels=model_config.input_img_channels,
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()

        # Move model to device
        self.to(self.device)

    # ===== Core Operations =====
    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the VQ-VAE model.
        """

        encoded_features = self.encoder(x)
        quantized_features, vq_loss = self.vector_quantizer(encoded_features)
        x_recon = self.decoder(quantized_features)

        return x_recon, vq_loss

    def _process_batch(
        self,
        batch: dict[str, Tensor],
        is_training: bool,
        optimizer: Optimizer | None = None,
        scaler: GradScaler | None = None,
    ) -> dict[str, float]:
        """
        Process a single batch of data.
        """
        tgt_imgs = batch["tgt_img"].to(self.device)
        ref_imgs = batch["ref_img"].to(self.device)

        if scaler is not None:
            with autocast(device_type=self.device.type):
                tgt_x_recon, tgt_vq_loss = self(tgt_imgs)
                tgt_recon_loss = self.loss_fn(tgt_x_recon, tgt_imgs)

                ref_x_recon, ref_vq_loss = self(ref_imgs)
                ref_recon_loss = self.loss_fn(ref_x_recon, ref_imgs)

                recon_loss = (tgt_recon_loss + ref_recon_loss) / 2
                vq_loss = (tgt_vq_loss + ref_vq_loss) / 2
                total_loss = recon_loss + vq_loss

            if is_training and optimizer is not None:
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            tgt_x_recon, tgt_vq_loss = self(tgt_imgs)
            tgt_recon_loss = self.loss_fn(tgt_x_recon, tgt_imgs)

            ref_x_recon, ref_vq_loss = self(ref_imgs)
            ref_recon_loss = self.loss_fn(ref_x_recon, ref_imgs)

            recon_loss = (tgt_recon_loss + ref_recon_loss) / 2
            vq_loss = (tgt_vq_loss + ref_vq_loss) / 2
            total_loss = recon_loss + vq_loss

            if is_training and optimizer is not None:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

        losses = {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "vq": vq_loss.item(),
        }

        return losses

    # ===== Training and Validation =====
    def fit(
        self,
        loader: Loader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        training_config: VQVAETrainingConfig,
        scaler: GradScaler | None = None,
    ) -> None:
        """
        Train the VQ-VAE model and save the best model checkpoint.
        """
        log_dir = Path(training_config.tensorboard_log_dir) / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        with SummaryWriter(log_dir) as writer:

            model_save_path = Path(training_config.model_save_path)
            model_save_path.parent.mkdir(parents=True, exist_ok=True)

            min_val_loss = float("inf")

            for epoch in range(training_config.num_epochs):

                # Train and validate for one epoch
                train_losses, val_losses = self._run_epoch(
                    loader=loader,
                    optimizer=optimizer,
                    scaler=scaler,
                )

                # Update learning rate
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                # Log training and validation metrics
                self._log_training_metrics(
                    writer=writer,
                    epoch=epoch,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    learning_rate=current_lr,
                )

                # Save the best model checkpoint
                if val_losses["total"] < min_val_loss:
                    min_val_loss = val_losses["total"]
                    torch.save(self.state_dict(), model_save_path)
                    print(f"✅ Best model saved (val loss: {min_val_loss:.6f})")

                # Print Metrics
                self._print_epoch_status(
                    epoch=epoch + 1,
                    total_epochs=training_config.num_epochs,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    learning_rate=current_lr,
                )

    def _run_epoch(
        self,
        loader: Loader,
        optimizer: Optimizer,
        scaler: GradScaler | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Run one epoch of training and validation.
        """
        train_loader = loader.loader.train
        val_loader = loader.loader.val

        train_losses = self._train_one_epoch(
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_losses = self._validate_one_epoch(
            val_loader=val_loader,
            scaler=scaler,
        )

        return train_losses, val_losses

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scaler: GradScaler | None = None,
    ) -> dict[str, float]:
        """
        Train the model for one epoch.
        """
        self.train()
        epoch_losses = {"total": 0.0, "recon": 0.0, "vq": 0.0}

        for batch in tqdm(train_loader, desc="Training"):
            batch_losses = self._process_batch(
                batch=batch,
                is_training=True,
                optimizer=optimizer,
                scaler=scaler,
            )
            for k in epoch_losses.keys():
                epoch_losses[k] += batch_losses[k]

        num_batches = len(train_loader)
        return {
            loss_type: loss / num_batches for loss_type, loss in epoch_losses.items()
        }

    @torch.no_grad()
    def _validate_one_epoch(
        self,
        val_loader: DataLoader,
        scaler: GradScaler | None = None,
    ) -> dict[str, float]:
        """
        Validate the model for one epoch.
        """
        self.eval()
        epoch_losses = {"total": 0.0, "recon": 0.0, "vq": 0.0}

        for batch in tqdm(val_loader, desc="Validating"):
            batch_losses = self._process_batch(
                batch=batch,
                is_training=False,
                optimizer=None,
                scaler=scaler,
            )
            for k in epoch_losses.keys():
                epoch_losses[k] += batch_losses[k]

        num_batches = len(val_loader)
        return {
            loss_type: loss / num_batches for loss_type, loss in epoch_losses.items()
        }

    # ===== Logging =====
    def _log_training_metrics(
        self,
        writer: SummaryWriter,
        epoch: int,
        train_losses: dict,
        val_losses: dict,
        learning_rate: float,
    ) -> None:
        """
        Log training and validation metrics to TensorBoard.
        """
        for loss_type, loss_value in train_losses.items():
            writer.add_scalar(f"Loss/train/{loss_type}", loss_value, epoch)

        for loss_type, loss_value in val_losses.items():
            writer.add_scalar(f"Loss/val/{loss_type}", loss_value, epoch)

        writer.add_scalar("LR", learning_rate, epoch)

    def _print_epoch_status(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: dict,
        val_losses: dict,
        learning_rate: float,
    ) -> None:
        """
        Print the status of the current epoch using a rich table.
        """
        console = Console()
        table = Table(title=f"Epoch Status [{epoch}/{total_epochs}]")

        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Train Loss", justify="right", style="magenta")
        table.add_column("Val Loss", justify="right", style="green")

        for loss_type in train_losses.keys():
            table.add_row(
                loss_type.capitalize(),
                f"{train_losses[loss_type]:.6f}",
                f"{val_losses[loss_type]:.6f}",
            )

        table.add_row("Learning Rate", f"{learning_rate:.6f}", "-")

        console.print(table)
