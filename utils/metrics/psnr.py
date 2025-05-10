from pathlib import Path

import torch
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from datasets.image_dataset import MetricsImageDataset


@torch.no_grad()
def compute_psnr_score(
    loader: DataLoader,
) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between generated and target images.
    """
    scores = []

    for batch in tqdm(loader, desc="Computing PSNR"):
        gen_imgs = batch["gen_img"].cpu().numpy()
        gt_imgs = batch["gt_img"].cpu().numpy()

        for gen_img, gt_img in zip(gen_imgs, gt_imgs):
            gen_img = gen_img.transpose((1, 2, 0))
            gt_img = gt_img.transpose((1, 2, 0))
            scores.append(compute_psnr(gt_img, gen_img, data_range=1.0))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score


def compute_psnr_from_directories(
    gen_img_dir: str | Path,
    gt_img_dir: str | Path,
    batch_size: int,
) -> float:
    """ """
    dataset = MetricsImageDataset(
        generated_img_dir=gen_img_dir,
        ground_truth_img_dir=gt_img_dir,
        normalize=False,
        convert_to_rgb=True,
    )

    loader = DataLoader(dataset, batch_size=batch_size)
    return compute_psnr_score(loader)
