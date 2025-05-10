import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs import (
    LDMDatasetConfig,
    LDMModelConfig,
    LDMTrainingConfig,
    VQVAEModelConfig,
)
from datasets.loader import Loader
from models import LDM
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import print_model_params, select_device


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Train LDM")
    parser.add_argument(
        "--split_ratios", type=float, nargs=2, help="Train/val split ratios"
    )
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "--pretrained_vqvae_path", type=str, help="Pretrained VQVAE model path"
    )
    parser.add_argument("--model_save_path", type=str, help="Model save path")
    parser.add_argument("--sample_root", type=str, help="Sample root directory")
    parser.add_argument("--sample_steps", type=int, help="Number of sampling steps")
    parser.add_argument("--img_save_interval", type=int, help="Image save interval")
    parser.add_argument(
        "--lpips_eval_interval", type=int, help="LPIPS evaluation interval"
    )
    parser.add_argument("--eval_batch_size", type=int, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, help="Training device (mps, cpu, cuda)")

    return parser.parse_args()


def train_ldm(
    dataset_config: LDMDatasetConfig,
    vqvae_model_config: VQVAEModelConfig,
    ldm_model_config: LDMModelConfig,
    training_config: LDMTrainingConfig,
    device: torch.device,
):
    """ """
    loader = Loader.from_dataset_config(
        dataset_config=dataset_config,
        device=device,
    )

    ldm = LDM(
        vqvae_model_config=vqvae_model_config,
        ldm_model_config=ldm_model_config,
        device=device,
    )

    optimizer = optim.Adam(
        ldm.parameters(),
        lr=training_config.learning_rate,
    )

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=training_config.num_epochs,
        eta_min=training_config.min_learning_rate,
    )

    print_model_params(
        model=ldm,
    )

    ldm.fit(
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
    )


def main() -> None:
    """ """
    args = parse_args()
    dataset_config = update_config_from_args(
        converting_config=LDMDatasetConfig(),
        args=args,
    )
    vqvae_model_config = update_config_from_args(
        converting_config=VQVAEModelConfig(),
        args=args,
    )
    ldm_model_config = update_config_from_args(
        converting_config=LDMModelConfig(),
        args=args,
    )
    training_config = update_config_from_args(
        converting_config=LDMTrainingConfig(),
        args=args,
    )
    device = select_device(args.device)

    train_ldm(
        dataset_config=dataset_config,
        vqvae_model_config=vqvae_model_config,
        ldm_model_config=ldm_model_config,
        training_config=training_config,
        device=device,
    )


if __name__ == "__main__":
    main()
