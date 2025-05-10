from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch import Tensor


def get_image_paths(
    img_dir: str | Path,
) -> list[Path]:
    """
    Return all image file paths in the given directory.
    """
    img_dir = Path(img_dir)

    if not img_dir.is_dir():
        raise ValueError(f"Provided path is not a directory: {img_dir}.")

    return sorted(
        [
            *img_dir.glob("*.png"),
            *img_dir.glob("*.jpg"),
            *img_dir.glob("*.jpeg"),
        ]
    )


def get_image_names(
    img_paths: list[Path],
) -> list[str]:
    """
    Return the names of images without their extensions.
    """
    return [img_path.stem for img_path in img_paths]


def check_names_match(
    tgt_img_paths: list[Path],
    ref_img_paths: list[Path],
) -> bool:
    """
    Check if the names of target and reference images match.
    """
    tgt_img_names = get_image_names(tgt_img_paths)
    ref_img_names = get_image_names(ref_img_paths)

    if tgt_img_names != ref_img_names:
        raise ValueError("Target and reference image names do not match.")

    return True


def convert_tensor_to_pil_images(
    images: Tensor,
) -> Image.Image | list[Image.Image]:
    """
    Convert normalized images from [-1,1] to [0,1] and return as PIL format.
    """
    match images.dim():
        case 3:
            return convert_to_pil(images)
        case 4:
            return [convert_to_pil(img) for img in images]
        case _:
            raise ValueError(f"Unsupported tensor shape: {images.shape}")


def convert_to_pil(
    image: Tensor,
) -> Image.Image:
    """
    Convert a normalized tensor to a PIL image.
    """
    image = image.clamp(-1, 1)
    image = image * 0.5 + 0.5
    return T.ToPILImage()(image)


def save_images(
    images: list[Image.Image],
    img_names: list[str],
    output_dir: Path,
) -> None:
    """
    Save a list of PIL images to the specified directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for img, img_name in zip(images, img_names):
        img.save(output_dir / f"{img_name}.png")
