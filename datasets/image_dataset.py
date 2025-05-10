from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from utils.image.image_utils import check_names_match, get_image_names, get_image_paths


def create_transform(
    normalize: bool,
) -> T.Compose:
    """
    Create a transform for the dataset.
    """

    transform_list = [T.ToTensor()]

    if normalize:
        transform_list.append(T.Normalize(mean=(0.5,), std=(0.5,)))

    return T.Compose(transform_list)


class PairedGlyphImageDataset(Dataset):
    """
    Paired image dataset for training LDM.
    """

    def __init__(
        self,
        target_img_dir: str | Path,
        reference_img_dir: str | Path,
    ):
        self.tgt_img_paths = get_image_paths(target_img_dir)
        self.ref_img_paths = get_image_paths(reference_img_dir)

        check_names_match(self.tgt_img_paths, self.ref_img_paths)
        self.img_names = get_image_names(self.tgt_img_paths)

        self.transform = create_transform(
            normalize=True,
        )

    def __len__(self):
        return len(self.ref_img_paths)

    def __getitem__(self, idx):
        tgt_img = Image.open(self.tgt_img_paths[idx]).convert("L")
        ref_img = Image.open(self.ref_img_paths[idx]).convert("L")

        tgt_img = self.transform(tgt_img)
        ref_img = self.transform(ref_img)

        img_name = self.img_names[idx]

        return {"tgt_img": tgt_img, "ref_img": ref_img, "img_name": img_name}

    def get_img_names(self) -> list[str]:
        """
        Get image names from the dataset.
        """
        return self.img_names

    def get_img_name_at_index(
        self,
        idx: int,
    ) -> str:
        """
        Get the image name at a specific index.
        """
        return self.img_names[idx]


class MetricsImageDataset(Dataset):
    """
    Paired image dataset for computing metrics.
    """

    def __init__(
        self,
        generated_img_dir: str | Path,
        ground_truth_img_dir: str | Path,
        normalize: bool,
        convert_to_rgb: bool,
    ):
        self.gen_img_paths = get_image_paths(generated_img_dir)
        self.gt_img_paths = get_image_paths(ground_truth_img_dir)

        self.normalize = normalize
        self.convert_to_rgb = convert_to_rgb

        check_names_match(self.gen_img_paths, self.gt_img_paths)
        self.img_names = get_image_names(self.gen_img_paths)

        self.transform = create_transform(
            normalize=normalize,
        )

    def __len__(self):
        return len(self.gen_img_paths)

    def __getitem__(self, idx):
        gen_img = Image.open(self.gen_img_paths[idx]).convert("L")
        gt_img = Image.open(self.gt_img_paths[idx]).convert("L")

        gen_img = self.transform(gen_img)
        gt_img = self.transform(gt_img)

        if self.convert_to_rgb:
            gen_img = gen_img.repeat(3, 1, 1)
            gt_img = gt_img.repeat(3, 1, 1)

        return {
            "gen_img": gen_img,
            "gt_img": gt_img,
        }
