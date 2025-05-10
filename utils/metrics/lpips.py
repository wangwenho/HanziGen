import warnings
from pathlib import Path

import lpips
import torch
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from datasets.image_dataset import MetricsImageDataset
from utils.hardware.hardware_utils import select_device

warnings.filterwarnings("ignore")


@torch.no_grad()
def compute_lpips_score(
    loader: DataLoader,
    device: torch.device | None = None,
) -> float:
    """
    Computes the LPIPS (Learned Perceptual Image Patch Similarity) score between generated and target images.
    """
    device = select_device(device)
    lpips_model = lpips.LPIPS(net="vgg").eval().to(device)
    scores = []

    for batch in tqdm(loader, desc="Computing LPIPS scores"):
        gen_imgs = batch["gen_img"].to(device)
        gt_imgs = batch["gt_img"].to(device)

        score = lpips_model(gen_imgs, gt_imgs).mean()
        scores.append(score.item())

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score


def compute_lpips_from_directories(
    gen_img_dir: str | Path,
    gt_img_dir: str | Path,
    batch_size: int,
    device: torch.device | None = None,
) -> float:
    """ """
    device = select_device(device)
    dataset = MetricsImageDataset(
        generated_img_dir=gen_img_dir,
        ground_truth_img_dir=gt_img_dir,
        normalize=True,
        convert_to_rgb=True,
    )

    loader = DataLoader(dataset, batch_size=batch_size)
    return compute_lpips_score(loader, device)
