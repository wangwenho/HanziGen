import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from utils.font.font_utils import write_charset_to_file


def split_dataset(
    dataset: Dataset,
    split_ratios: tuple[float, float],
    random_seed: int,
) -> tuple[Subset]:
    """
    Splits a dataset into multiple subsets based on given ratios.
    """
    if not math.isclose(sum(split_ratios), 1.0, rel_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0.")

    torch.manual_seed(random_seed)

    total_len = len(dataset)
    lengths = [int(round(total_len * ratio)) for ratio in split_ratios]

    while sum(lengths) != total_len:
        diff = total_len - sum(lengths)
        for i in range(abs(diff)):
            lengths[i % len(lengths)] += 1 if diff > 0 else -1

    return random_split(dataset, lengths)


def get_img_names_from_subset(
    subset: Subset,
) -> list[str]:
    """
    Get image names from a Subset.
    """
    dataset = subset.dataset

    if hasattr(dataset, "get_img_name_at_index"):
        return [dataset.get_img_name_at_index(i) for i in subset.indices]

    raise AttributeError("Dataset does not have get_img_name_at_index method.")


def get_img_names_from_dataloader(
    dataloader: DataLoader,
) -> list[str]:
    """
    Get image names from a DataLoader.
    """
    dataset = dataloader.dataset

    if isinstance(dataset, Subset):
        return get_img_names_from_subset(dataset)

    raise AttributeError("Dataset does not have get_img_names method.")


def save_dataset_charset(
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_font_path: str,
    charset_root: str,
) -> None:
    """
    Save the character sets from the training and validation loaders to text files.
    """
    train_img_names = get_img_names_from_dataloader(train_loader)
    val_img_names = get_img_names_from_dataloader(val_loader)

    train_charset = set(chr(int(char, 16)) for char in train_img_names)
    val_charset = set(chr(int(char, 16)) for char in val_img_names)

    charset_dir = Path(charset_root) / "splits"
    charset_dir.mkdir(parents=True, exist_ok=True)

    target_font_name = Path(target_font_path).stem
    train_charset_path = charset_dir / target_font_name / "train.txt"
    val_charset_path = charset_dir / target_font_name / "val.txt"
    (charset_dir / target_font_name).mkdir(parents=True, exist_ok=True)

    write_charset_to_file(train_charset, train_charset_path)
    write_charset_to_file(val_charset, val_charset_path)
