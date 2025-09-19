from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm

from configs.font_processing_config import FontProcessingConfig
from configs.ldm_config import LDMInferenceConfig, LDMModelConfig, LDMTrainingConfig
from configs.vqvae_config import VQVAEModelConfig
from datasets.image_dataset import PairedGlyphImageDataset
from datasets.loader import Loader
from models.unet.unet import UNet
from models.vqvae.vqvae import VQVAE
from utils.font.font_utils import read_charset_from_file
from utils.hardware.hardware_utils import select_device
from utils.image.image_generator import GlyphImageGenerator
from utils.image.image_utils import convert_tensor_to_pil_images, save_images
from utils.metrics.lpips import compute_lpips_from_directories
from utils.metrics.psnr import compute_psnr_from_directories
from utils.metrics.ssim import compute_ssim_from_directories

from .scheduler import SigmoidScheduler
from .time_embedding import TimeEmbedding


class LDM(nn.Module):
    """
    Latent Diffusion Model (LDM).
    """

    # ===== Initialization & Setup =====
    def __init__(
        self,
        vqvae_model_config: VQVAEModelConfig,
        ldm_model_config: LDMModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Initialize model device
        self.device = select_device(device)

        # Define time embedding
        self.time_emb = TimeEmbedding(
            time_pos_dim=ldm_model_config.time_pos_dim,
            time_emb_dim=ldm_model_config.time_emb_dim,
            time_steps=ldm_model_config.time_steps,
            device=self.device,
        )

        # Define VQ-VAE model
        self.vqvae = VQVAE(
            model_config=vqvae_model_config,
            device=self.device,
        )
        self.vqvae_encoder = self.vqvae.encoder
        self.vqvae_quantizer = self.vqvae.vector_quantizer
        self.vqvae_decoder = self.vqvae.decoder

        # Define UNet model
        self.unet = UNet(
            in_channels=vqvae_model_config.latent_dim * 2,
            out_channels=vqvae_model_config.latent_dim,
            base_channels=ldm_model_config.unet_base_channels,
            time_emb_dim=ldm_model_config.time_emb_dim,
            device=self.device,
        )

        # Define scheduler
        self.scheduler = SigmoidScheduler(
            noise_steps=ldm_model_config.time_steps,
            device=self.device,
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()

        # Move model to device
        self.to(self.device)

    def _freeze_vqvae(
        self,
    ) -> None:
        """
        Freeze VQ-VAE model parameters.
        """
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def _load_pretrained_vqvae(
        self,
        pretrained_vqvae_path: str | Path,
    ) -> None:
        """
        Load pre-trained VQ-VAE model weights.
        """
        pretrained_vqvae = Path(pretrained_vqvae_path)
        if not pretrained_vqvae.exists():
            raise FileNotFoundError(
                f"Pretrained VQ-VAE checkpoint not found: '{pretrained_vqvae_path}'"
            )
        if not pretrained_vqvae.is_file():
            raise FileNotFoundError(
                f"Invalid pretrained VQ-VAE checkpoint path: '{pretrained_vqvae_path}' is not a file"
            )

        print(f"Loading pretrained VQ-VAE from '{pretrained_vqvae_path}'")
        state_dict = torch.load(
            pretrained_vqvae_path,
            map_location=self.device,
            weights_only=True,
        )
        self.vqvae.load_state_dict(state_dict)

    # ===== Core Operations =====
    def forward(
        self,
        x: Tensor,
        ref: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Forward pass of the LDM.
        """
        t_emb = self.time_emb(t)
        x_cat = torch.cat([x, ref], dim=1)
        noise_pred = self.unet(x_cat, t_emb)

        return noise_pred

    def _process_batch(
        self,
        batch: dict[str, Tensor],
        is_training: bool,
        optimizer: Optimizer | None = None,
        scaler: GradScaler | None = None,
    ) -> float:
        """
        Process a single batch of data.
        """
        tgt_imgs = batch["tgt_img"].to(self.device)
        ref_imgs = batch["ref_img"].to(self.device)

        if scaler is not None:
            with autocast(device_type=self.device.type):
                batch_size = tgt_imgs.shape[0]
                t = self.scheduler.sample_timesteps(batch_size)

                tgt_latents = self._encode_to_latent(tgt_imgs)
                x_t, noise = self.scheduler.add_noise(tgt_latents, t)
                ref_latents = self._encode_to_latent(ref_imgs)

                noise_pred = self(x_t, ref_latents, t)
                loss = self.loss_fn(noise_pred, noise)

            if is_training and optimizer is not None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        else:
            batch_size = tgt_imgs.shape[0]
            t = self.scheduler.sample_timesteps(batch_size)

            tgt_latents = self._encode_to_latent(tgt_imgs)
            x_t, noise = self.scheduler.add_noise(tgt_latents, t)
            ref_latents = self._encode_to_latent(ref_imgs)

            noise_pred = self(x_t, ref_latents, t)
            loss = self.loss_fn(noise_pred, noise)

            if is_training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

        return loss.item()

    def _encode_to_latent(self, images: Tensor) -> Tensor:
        """
        Encode and quantize images.
        """
        latents = self.vqvae_encoder(images)
        quantized_latents, _ = self.vqvae_quantizer(latents)

        return quantized_latents

    def _decode_from_latent(self, latents: Tensor) -> Tensor:
        """
        Decode latent back to images.
        """
        return self.vqvae_decoder(latents)

    # ===== Training and Validation =====
    def fit(
        self,
        loader: Loader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        training_config: LDMTrainingConfig,
        resume: bool = False,
        scaler: GradScaler | None = None,
    ) -> None:
        """
        Train the LDM and save the best model checkpoint.
        """
        if not resume:
            self._load_pretrained_vqvae(training_config.pretrained_vqvae_path)

        # Freeze VQ-VAE parameters
        self._freeze_vqvae()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_sample_root = Path(training_config.sample_root)
        training_config.sample_root = str(
            base_sample_root.parent / f"{base_sample_root.name}_training_{timestamp}"
        )
        print(f"📁 Training samples will be saved to: {training_config.sample_root}")

        log_dir = Path(training_config.tensorboard_log_dir) / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)

        with SummaryWriter(log_dir) as writer:

            model_save_path = Path(training_config.model_save_path)
            model_save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save ground truth images for evaluation
            self._save_evaluation_ground_truth(
                loader=loader,
                split=training_config.gt_split,
                sample_root=training_config.sample_root,
            )

            min_lpips_score = float("inf")

            for epoch in range(training_config.num_epochs):

                # Train and validate for one epoch
                train_loss, val_loss = self._run_epoch(
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
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=current_lr,
                )

                # Generate training and validation sample images
                self._save_epoch_visualization_samples(
                    loader=loader,
                    training_config=training_config,
                    epoch=epoch,
                )

                # Evaluate metrics on the validation dataset
                scores = self._compute_and_log_metrics(
                    loader=loader,
                    training_config=training_config,
                    writer=writer,
                    epoch=epoch,
                )

                # Save the best model based on LPIPS score
                if (
                    scores is not None
                    and scores.get("lpips", float("inf")) < min_lpips_score
                ):

                    min_lpips_score = scores["lpips"]
                    torch.save(self.state_dict(), model_save_path)
                    print(f"✅ Best model saved. (LPIPS score: {min_lpips_score:.6f})")

                # Print Metrics
                self._print_epoch_status(
                    epoch=epoch + 1,
                    total_epochs=training_config.num_epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=current_lr,
                )

    def _run_epoch(
        self,
        loader: Loader,
        optimizer: Optimizer,
        scaler: GradScaler | None = None,
    ) -> tuple[float, float]:
        """
        Run one epoch of training and validation.
        """
        train_loader = loader.loader.train
        val_loader = loader.loader.val

        train_loss = self._train_one_epoch(
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss = self._validate_one_epoch(
            val_loader=val_loader,
            scaler=scaler,
        )

        return train_loss, val_loss

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scaler: GradScaler | None = None,
    ) -> float:
        """
        Train the model for one epoch.
        """
        self.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            batch_loss = self._process_batch(
                batch=batch,
                is_training=True,
                optimizer=optimizer,
                scaler=scaler,
            )
            epoch_loss += batch_loss

        return epoch_loss / len(train_loader)

    @torch.no_grad()
    def _validate_one_epoch(
        self,
        val_loader: DataLoader,
        scaler: GradScaler | None = None,
    ) -> float:
        """
        Validate the model for one epoch.
        """
        self.eval()
        epoch_loss = 0.0

        for batch in tqdm(val_loader, desc="Validating"):
            batch_loss = self._process_batch(
                batch=batch,
                is_training=False,
                optimizer=None,
                scaler=scaler,
            )
            epoch_loss += batch_loss

        return epoch_loss / len(val_loader)

    # ===== Evaluation =====
    @torch.no_grad()
    def _compute_and_log_metrics(
        self,
        loader: Loader,
        training_config: LDMTrainingConfig,
        writer: SummaryWriter,
        epoch: int,
    ) -> dict[str, float] | None:
        """
        Evaluate the model on the validation dataset using various metrics.
        """
        self.eval()

        if (
            epoch % training_config.lpips_eval_interval != 0
            and epoch != training_config.num_epochs - 1
        ):
            return

        self._generate_images_from_loader(
            loader=loader.loader.val, config=training_config
        )

        sample_root = Path(training_config.sample_root)

        scores = {
            "psnr": compute_psnr_from_directories(
                gen_img_dir=sample_root / training_config.gen_split,
                gt_img_dir=sample_root / training_config.gt_split,
                batch_size=training_config.eval_batch_size,
            ),
            "ssim": compute_ssim_from_directories(
                gen_img_dir=sample_root / training_config.gen_split,
                gt_img_dir=sample_root / training_config.gt_split,
                batch_size=training_config.eval_batch_size,
            ),
            "lpips": compute_lpips_from_directories(
                gen_img_dir=sample_root / training_config.gen_split,
                gt_img_dir=sample_root / training_config.gt_split,
                batch_size=training_config.eval_batch_size,
                device=self.device,
            ),
        }

        self._log_evaluation_metrics(
            writer=writer,
            epoch=epoch,
            scores=scores,
        )

        return scores

    @torch.no_grad()
    def _save_epoch_visualization_samples(
        self,
        loader: Loader,
        training_config: LDMTrainingConfig,
        epoch: int,
    ) -> None:
        """
        Generate and save sample images from the training and validation loaders.
        """
        self.eval()

        if (
            epoch % training_config.img_save_interval != 0
            and epoch != training_config.num_epochs - 1
        ):
            return

        train_loader = loader.loader.train
        val_loader = loader.loader.val

        train_split = training_config.train_split
        val_split = training_config.val_split

        for data_split, loader in tqdm(
            [(train_split, train_loader), (val_split, val_loader)],
            desc="Generating sample images",
        ):
            self._generate_triple_comparison_images(
                loader=loader,
                training_config=training_config,
                epoch=epoch,
                split=data_split,
            )

    @torch.no_grad()
    def _save_evaluation_ground_truth(
        self,
        loader: Loader,
        split: str,
        sample_root: str | Path,
    ) -> None:
        """
        Save ground truth images from the validation loader.
        """
        self.eval()

        val_loader = loader.loader.val

        sample_dir = Path(sample_root) / split
        sample_dir.mkdir(parents=True, exist_ok=True)

        for batch in tqdm(val_loader, desc="Saving ground truth images"):
            imgs = batch["tgt_img"].to(self.device)
            img_names = batch["img_name"]

            imgs_pil = convert_tensor_to_pil_images(imgs)
            save_images(imgs_pil, img_names, sample_dir)

    # ===== Diffusion Processing =====
    @torch.no_grad()
    def _denoise_using_ddpm(
        self,
        latents: Tensor,
        ref_latents: Tensor,
        sample_steps: int,
    ) -> Tensor:
        """
        Perform DDPM steps for image generation.
        """
        self.scheduler.set_timesteps(sample_steps)
        timesteps = self.scheduler.timesteps
        for t in timesteps:
            t_tensor = torch.full((latents.shape[0],), t, device=self.device)

            noise_pred = self(latents, ref_latents, t_tensor)
            latents = self.scheduler.ddpm_step(
                x=latents,
                t=t_tensor,
                y=noise_pred,
            )

        return latents

    @torch.no_grad()
    def _denoise_using_ddim(
        self,
        latents: Tensor,
        ref_latents: Tensor,
        sample_steps: int,
    ) -> Tensor:
        """
        Perform DDIM steps for image generation.
        """
        self.scheduler.set_timesteps(sample_steps + 1)
        timesteps = self.scheduler.timesteps

        for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_tensor = torch.full((latents.shape[0],), t, device=self.device)
            t_prev_tensor = torch.full((latents.shape[0],), t_prev, device=self.device)

            noise_pred = self(latents, ref_latents, t_tensor)
            latents = self.scheduler.ddim_step(
                x=latents,
                t=t_tensor,
                t_prev=t_prev_tensor,
                y=noise_pred,
            )

        return latents

    def _sample_random_latents(
        self,
        batch_size: int,
        latent_channels: int,
        latent_size: tuple[int, int],
        device: torch.device | None = None,
    ) -> Tensor:
        """
        Sample random latents for inference.
        """
        return torch.randn(
            batch_size,
            latent_channels,
            latent_size[0],
            latent_size[1],
            device=device,
        )

    # ===== Image Generation =====
    @torch.no_grad()
    def _generate_images_from_loader(
        self,
        loader: DataLoader,
        config: LDMTrainingConfig | LDMInferenceConfig,
    ) -> None:
        """
        Generate and saves images from a DataLoader.
        """
        self.eval()

        sample_dir = Path(config.sample_root) / config.gen_split
        sample_dir.mkdir(parents=True, exist_ok=True)

        for batch in tqdm(loader, desc="Generating inference images"):
            ref_imgs = batch["ref_img"].to(self.device)
            img_names = batch["img_name"]

            generated_pil_imgs = self._synthesize_images_from_references(
                ref_imgs=ref_imgs,
                config=config,
            )
            save_images(generated_pil_imgs, img_names, sample_dir)

    @torch.no_grad()
    def _synthesize_images_from_references(
        self,
        ref_imgs: Tensor,
        config: LDMTrainingConfig | LDMInferenceConfig,
    ) -> list[Image.Image]:
        """
        Generate batch images from reference images.
        """
        self.eval()

        ref_latents = self._encode_to_latent(ref_imgs)
        b, c, h, w = ref_latents.shape

        latents = self._sample_random_latents(
            batch_size=b,
            latent_channels=c,
            latent_size=(h, w),
            device=self.device,
        )

        latents = self._denoise_using_ddim(
            latents=latents,
            ref_latents=ref_latents,
            sample_steps=config.sample_steps,
        )

        generated_imgs = self._decode_from_latent(latents)
        generated_pil_imgs = convert_tensor_to_pil_images(generated_imgs)

        return generated_pil_imgs

    @torch.no_grad()
    def _generate_triple_comparison_images(
        self,
        loader: DataLoader,
        training_config: LDMTrainingConfig,
        epoch: int,
        split: str,
    ) -> None:
        """
        Generate and saves combined images from a DataLoader.
        """
        self.eval()

        batch = next(iter(loader))
        tgt_imgs = batch["tgt_img"].to(self.device)
        ref_imgs = batch["ref_img"].to(self.device)
        img_names = batch["img_name"]

        generated_pil_imgs = self._synthesize_images_from_references(
            ref_imgs=ref_imgs,
            config=training_config,
        )

        sample_dir = Path(training_config.sample_root) / split
        sample_dir.mkdir(parents=True, exist_ok=True)

        for gen_pil_img, tgt_img, ref_img, img_name in zip(
            generated_pil_imgs, tgt_imgs, ref_imgs, img_names
        ):

            tgt_pil_img = convert_tensor_to_pil_images(tgt_img)
            ref_pil_img = convert_tensor_to_pil_images(ref_img)

            width, height = tgt_pil_img.width, tgt_pil_img.height
            combined_img = Image.new("L", (width * 3, height))

            combined_img.paste(ref_pil_img, (0, 0))
            combined_img.paste(tgt_pil_img, (width, 0))
            combined_img.paste(gen_pil_img, (width * 2, 0))

            img_path = sample_dir / f"epoch_{epoch:04d}_{img_name}.png"
            combined_img.save(img_path)

    @torch.no_grad()
    def generate_images_from_charset_file(
        self,
        target_font_path: str | Path,
        reference_fonts_dir: str | Path,
        charset_path: str | Path,
        font_processing_config: FontProcessingConfig,
        inference_config: LDMInferenceConfig,
    ) -> None:
        """
        Generate images from a charset file.
        """
        self.eval()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_sample_root = Path(inference_config.sample_root)
        inference_config.sample_root = str(
            base_sample_root.parent / f"{base_sample_root.name}_inference_{timestamp}"
        )
        print(f"📁 Inference samples will be saved to: {inference_config.sample_root}")

        tgt_generator = GlyphImageGenerator.from_target_font(
            target_font_path=target_font_path,
            font_processing_config=font_processing_config,
        )
        ref_generators = GlyphImageGenerator.from_reference_fonts(
            reference_fonts_dir=reference_fonts_dir,
            font_processing_config=font_processing_config,
        )

        charset = read_charset_from_file(
            charset_path=charset_path,
        )

        sample_dir = Path(inference_config.sample_root)
        tgt_output_dir = sample_dir / inference_config.gt_split
        ref_output_dir = sample_dir / inference_config.ref_split

        for directory in [tgt_output_dir, ref_output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        for char in tqdm(charset, desc="Generating ground truth and reference images"):
            tgt_generator.save_glyph_image(
                char=char,
                output_dir=tgt_output_dir,
                img_size=font_processing_config.img_size,
            )

            for ref_generator in ref_generators:
                covered_charset_path = (
                    Path(font_processing_config.unihan_coverage_charset_dir)
                    / ref_generator.font_name
                    / "covered.txt"
                )
                covered_charset = read_charset_from_file(
                    charset_path=covered_charset_path,
                )

                if char in covered_charset:
                    ref_generator.save_glyph_image(
                        char=char,
                        output_dir=ref_output_dir,
                        img_size=font_processing_config.img_size,
                    )

        dataset = PairedGlyphImageDataset(tgt_output_dir, ref_output_dir)
        loader = DataLoader(
            dataset, batch_size=inference_config.batch_size, shuffle=False
        )
        self._generate_images_from_loader(
            loader=loader,
            config=inference_config,
        )

    # ===== Logging =====
    def _log_training_metrics(
        self,
        writer: SummaryWriter,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
    ) -> None:
        """
        Log training and validation metrics to TensorBoard.
        """
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", learning_rate, epoch)

    def _log_evaluation_metrics(
        self,
        writer: SummaryWriter,
        epoch: int,
        scores: dict[str, float],
    ) -> None:
        """
        Log evaluation metrics to TensorBoard.
        """
        for metric_name, metric_value in scores.items():
            writer.add_scalar(f"Metrics/val/{metric_name}", metric_value, epoch)

    def _print_epoch_status(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
    ) -> None:
        """
        Print the status for the current epoch using a rich table.
        """
        console = Console()
        table = Table(title=f"Epoch Status [{epoch}/{total_epochs}]")

        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Train Loss", justify="right", style="magenta")
        table.add_column("Val Loss", justify="right", style="green")

        table.add_row("Total", f"{train_loss:.6f}", f"{val_loss:.6f}")
        table.add_row("Learning Rate", f"{learning_rate:.6f}", "-")

        console.print(table)
