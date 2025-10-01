import os
import shutil
import tempfile
from pathlib import Path

import torch
from cleanfid import fid
from PIL import Image
from tqdm.rich import tqdm

from utils.hardware.hardware_utils import select_device
from utils.image.image_utils import get_image_paths


def compute_fid_score(
    gen_img_dir: str | Path,
    gt_img_dir: str | Path,
    batch_size: int,
    device: torch.device | None = None,
) -> float:
    """
    Computes the Fréchet Inception Distance (FID) score between generated and target images.
    """
    device = select_device(device)
    score = fid.compute_fid(
        fdir1=gen_img_dir,
        fdir2=gt_img_dir,
        num_workers=0,
        batch_size=batch_size,
        device=device,
    )

    return score


def compute_fid_from_directories(
    gen_img_dir: str | Path,
    gt_img_dir: str | Path,
    batch_size: int,
    device: str | torch.device | None = None,
) -> float:
    """ """
    device = select_device(device)
    temp_dir = tempfile.mkdtemp()
    temp_gen_img_dir = os.path.join(temp_dir, "gen_rgb_img")
    temp_gt_img_dir = os.path.join(temp_dir, "gt_rgb_img")

    os.makedirs(temp_gen_img_dir, exist_ok=True)
    os.makedirs(temp_gt_img_dir, exist_ok=True)

    try:
        for img_path in tqdm(
            get_image_paths(gen_img_dir),
            desc="Converting images to RGB",
        ):
            img = Image.open(img_path).convert("L")
            rgb_img = Image.merge("RGB", (img, img, img))
            rgb_img.save(os.path.join(temp_gen_img_dir, img_path.name))

        for img_path in tqdm(
            get_image_paths(gt_img_dir),
            desc="Converting images to RGB",
        ):
            img = Image.open(img_path).convert("L")
            rgb_img = Image.merge("RGB", (img, img, img))
            rgb_img.save(os.path.join(temp_gt_img_dir, img_path.name))

        fid_score = compute_fid_score(
            gen_img_dir=temp_gen_img_dir,
            gt_img_dir=temp_gt_img_dir,
            batch_size=batch_size,
            device=device,
        )

    finally:
        shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} removed.")

    return fid_score
