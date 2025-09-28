import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs import VQVAEDatasetConfig, VQVAEModelConfig, VQVAETrainingConfig
from datasets.loader import Loader
from models import VQVAE
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import print_model_params, select_device
from utils.validation.project_validator import ProjectValidationError, ProjectValidator


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for VQ-VAE training.
    """
    parser = argparse.ArgumentParser(
        description="Train VQVAE model",
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=2,
        help="Train/val split ratios",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        help="Model save path",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device (mps, cpu, cuda)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--use_amp",
        type=str,
        default="false",
        help="Use Automatic Mixed Precision",
    )
    return parser.parse_args()


def load_vqvae_checkpoint(
    vqvae: VQVAE,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """
    Load VQ-VAE model weights from checkpoint.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: '{checkpoint_path}'")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Invalid VQ-VAE checkpoint path: '{checkpoint_path}' is not a file"
        )

    print(f"Loading VQ-VAE checkpoint from '{checkpoint_path}'")
    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    vqvae.load_state_dict(state_dict)


def train_vqvae(
    dataset_config: VQVAEDatasetConfig,
    model_config: VQVAEModelConfig,
    training_config: VQVAETrainingConfig,
    device: torch.device,
    resume: bool = False,
    use_amp: bool = False,
):
    """
    Train VQ-VAE model.
    """
    loader = Loader.from_dataset_config(
        dataset_config=dataset_config,
        device=device,
    )

    vqvae = VQVAE(
        model_config=model_config,
        device=device,
    )

    optimizer = optim.Adam(
        vqvae.parameters(),
        lr=training_config.learning_rate,
    )

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=training_config.num_epochs,
        eta_min=training_config.min_learning_rate,
    )

    scaler = GradScaler(device=device) if use_amp else None

    if resume:
        load_vqvae_checkpoint(
            vqvae=vqvae,
            checkpoint_path=training_config.model_save_path,
            device=device,
        )

    print_model_params(
        model=vqvae,
    )

    vqvae.fit(
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        scaler=scaler,
    )


def main() -> None:
    """
    Main function to run the VQ-VAE training process.
    """
    try:
        args = parse_args()
        resume = args.resume.lower() == "true"
        use_amp = args.use_amp.lower() == "true"

        if resume:
            ProjectValidator.validate_checkpoint_file(
                file_path=args.model_save_path,
                name="VQ-VAE model checkpoint",
            )

        dataset_config = update_config_from_args(
            converting_config=VQVAEDatasetConfig(),
            args=args,
        )
        model_config = update_config_from_args(
            converting_config=VQVAEModelConfig(),
            args=args,
        )
        training_config = update_config_from_args(
            converting_config=VQVAETrainingConfig(),
            args=args,
        )
        device = select_device(args.device)

        train_vqvae(
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
            device=device,
            resume=resume,
            use_amp=use_amp,
        )
    except ProjectValidationError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
