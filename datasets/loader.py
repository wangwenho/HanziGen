from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from configs.ldm_config import LDMDatasetConfig
from configs.vqvae_config import VQVAEDatasetConfig

from .dataset_utils import split_dataset
from .image_dataset import PairedGlyphImageDataset


@dataclass
class TrainValLoader:
    """
    Contains training and validation dataset splits.
    """

    train: DataLoader
    val: DataLoader

    @classmethod
    def from_dataset(
        cls,
        dataset: PairedGlyphImageDataset,
        dataset_config: VQVAEDatasetConfig | LDMDatasetConfig,
        device: torch.device,
    ) -> "TrainValLoader":
        """
        Creates training and validation data loaders from a dataset.
        """
        train_dataset, val_dataset = split_dataset(
            dataset=dataset,
            split_ratios=dataset_config.split_ratios,
            random_seed=dataset_config.split_random_seed,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=True,
            num_workers=dataset_config.num_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True if device.type == "cuda" else False,
        )

        return cls(train=train_loader, val=val_loader)


@dataclass
class Loader:
    """
    Holds a single data loader for a specific dataset split.
    """

    loader: TrainValLoader

    @classmethod
    def from_dataset_config(
        cls,
        dataset_config: VQVAEDatasetConfig | LDMDatasetConfig,
        device: torch.device,
    ) -> "Loader":
        """
        Creates a single data loader from a dataset directory.
        """
        dataset = PairedGlyphImageDataset(
            target_img_dir=dataset_config.target_img_dir,
            reference_img_dir=dataset_config.reference_img_dir,
        )
        loader = TrainValLoader.from_dataset(
            dataset=dataset,
            dataset_config=dataset_config,
            device=device,
        )

        return cls(loader=loader)
