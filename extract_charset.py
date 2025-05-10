import argparse
from pathlib import Path

import torch

from configs import LDMDatasetConfig
from datasets.dataset_utils import save_dataset_charset
from datasets.loader import Loader
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import select_device


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(
        description="Extract train/val charset from dataset"
    )
    parser.add_argument("--target_font_path", type=str, help="Target font path")
    parser.add_argument(
        "--split_ratios", type=float, nargs=2, help="Train/val split ratios"
    )
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, help="Training device (mps, cpu, cuda)")

    return parser.parse_args()


def extract_train_val_charset(
    target_font_path: str,
    dataset_config: LDMDatasetConfig,
    device: torch.device,
) -> None:
    """ """
    loader = Loader.from_dataset_config(
        dataset_config=dataset_config,
        device=device,
    )

    save_dataset_charset(
        train_loader=loader.loader.train,
        val_loader=loader.loader.val,
        target_font_path=target_font_path,
        charset_root=dataset_config.splits_root,
    )


def main() -> None:
    """ """
    args = parse_args()
    dataset_config = update_config_from_args(
        converting_config=LDMDatasetConfig(),
        args=args,
    )
    device = select_device(args.device)

    extract_train_val_charset(
        target_font_path=args.target_font_path,
        dataset_config=dataset_config,
        device=device,
    )


if __name__ == "__main__":
    main()
