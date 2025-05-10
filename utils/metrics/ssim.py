from pathlib import Path

import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from datasets.image_dataset import MetricsImageDataset


@torch.no_grad()
def compute_ssim_score(
    loader: DataLoader,
) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between generated and target images.
    """
    scores = []

    for batch in tqdm(loader, desc="Computing SSIM"):
        gen_imgs = batch["gen_img"].cpu().numpy()
        gt_imgs = batch["gt_img"].cpu().numpy()

        for gen_img, gt_img in zip(gen_imgs, gt_imgs):
            gen_img = gen_img.transpose((1, 2, 0))
            gt_img = gt_img.transpose((1, 2, 0))
            scores.append(ssim(gt_img, gen_img, data_range=1, channel_axis=-1))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score


def compute_ssim_from_directories(
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
    return compute_ssim_score(loader)
