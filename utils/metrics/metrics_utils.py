from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from utils.hardware.hardware_utils import select_device

from .fid import compute_fid_from_directories
from .lpips import compute_lpips_from_directories
from .psnr import compute_psnr_from_directories
from .ssim import compute_ssim_from_directories


def compute_all_metrics(
    generated_img_dir: str,
    ground_truth_img_dir: str,
    eval_batch_size: int,
    device: torch.device | None = None,
) -> None:
    """
    Compute PSNR, SSIM, LPIPS, and FID scores for the generated images compared to the target images.
    """
    device = select_device(device)

    gen_img_dir = Path(generated_img_dir)
    gt_img_dir = Path(ground_truth_img_dir)

    psnr_score = compute_psnr_from_directories(
        gen_img_dir=gen_img_dir,
        gt_img_dir=gt_img_dir,
        batch_size=eval_batch_size,
    )

    ssim_score = compute_ssim_from_directories(
        gen_img_dir=gen_img_dir,
        gt_img_dir=gt_img_dir,
        batch_size=eval_batch_size,
    )

    lpips_score = compute_lpips_from_directories(
        gen_img_dir=gen_img_dir,
        gt_img_dir=gt_img_dir,
        batch_size=eval_batch_size,
        device=device,
    )

    fid_score = compute_fid_from_directories(
        gen_img_dir=gen_img_dir,
        gt_img_dir=gt_img_dir,
        batch_size=eval_batch_size,
        device=device,
    )

    scores = {
        "psnr": psnr_score,
        "ssim": ssim_score,
        "lpips": lpips_score,
        "fid": fid_score,
    }

    console = Console()
    table = Table(title="Metrics Results")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="magenta")

    for metric, score in scores.items():
        table.add_row(metric.upper(), f"{score:.4f}")

    console.print(table)
