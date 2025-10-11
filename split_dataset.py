import argparse
import sys
from pathlib import Path

import torch

from configs import LDMDatasetConfig
from datasets.dataset_utils import save_dataset_charset
from datasets.loader import Loader
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import select_device
from utils.validation.project_validator import ProjectValidationError, ProjectValidator


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for charset extraction.
    """
    parser = argparse.ArgumentParser(
        description="Extract train/val charset from dataset",
    )
    parser.add_argument(
        "--target_font_path",
        type=str,
        help="Target font path",
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=2,
        help="Train/val split ratios",
    )
    parser.add_argument(
        "--split_random_seed",
        type=int,
        help="Split random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device (mps, cpu, cuda)",
    )

    return parser.parse_args()


def split_and_extract_charset(
    target_font_path: str,
    dataset_config: LDMDatasetConfig,
    device: torch.device,
) -> None:
    """
    Extract train/val charset from dataset and save to files.
    """
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
    """
    Main function to run the charset extraction process.
    """
    try:
        args = parse_args()

        ProjectValidator.validate_font_file(
            file_path=args.target_font_path,
            name="Target font",
        )

        dataset_config = update_config_from_args(
            converting_config=LDMDatasetConfig(),
            args=args,
        )
        device = select_device(args.device)

        split_and_extract_charset(
            target_font_path=args.target_font_path,
            dataset_config=dataset_config,
            device=device,
        )
    except ProjectValidationError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
