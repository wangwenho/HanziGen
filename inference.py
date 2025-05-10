import argparse

import torch

from configs import (
    FontProcessingConfig,
    LDMInferenceConfig,
    LDMModelConfig,
    VQVAEModelConfig,
)
from models import LDM
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import print_model_params, select_device


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--target_font_path", type=str, help="Target font path")
    parser.add_argument(
        "--reference_fonts_dir", type=str, help="Reference fonts directory"
    )
    parser.add_argument("--charset_path", type=str, help="Path to charset file")
    parser.add_argument(
        "--pretrained_ldm_path", type=str, help="Path to pretrained LDM model"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--sample_root", type=str, help="Sample root directory")
    parser.add_argument("--sample_steps", type=int, help="Number of sampling steps")
    parser.add_argument(
        "--img_size", type=int, nargs=2, help="Image size (width height)"
    )
    parser.add_argument("--device", type=str, help="Training device (mps, cpu, cuda)")

    return parser.parse_args()


def inference(
    target_font_path: str,
    reference_fonts_dir: str,
    charset_path: str,
    vqvae_model_config: VQVAEModelConfig,
    ldm_model_config: LDMModelConfig,
    font_processing_config: FontProcessingConfig,
    ldm_inference_config: LDMInferenceConfig,
    device: str,
) -> None:
    """ """
    device = torch.device(device)
    ldm = LDM(
        vqvae_model_config=vqvae_model_config,
        ldm_model_config=ldm_model_config,
        device=device,
    )

    ckpt = torch.load(
        ldm_inference_config.pretrained_ldm_path,
        map_location="cpu",
        weights_only=True,
    )
    ldm.load_state_dict(ckpt, strict=False)

    print_model_params(
        model=ldm,
    )

    ldm.generate_images_from_charset_file(
        target_font_path=target_font_path,
        reference_fonts_dir=reference_fonts_dir,
        charset_path=charset_path,
        font_processing_config=font_processing_config,
        inference_config=ldm_inference_config,
    )


def main() -> None:
    """ """
    args = parse_args()
    font_processing_config = update_config_from_args(
        converting_config=FontProcessingConfig(),
        args=args,
    )
    vqvae_model_config = update_config_from_args(
        converting_config=VQVAEModelConfig(),
        args=args,
    )
    ldm_model_config = update_config_from_args(
        converting_config=LDMModelConfig(),
        args=args,
    )
    ldm_inference_config = update_config_from_args(
        converting_config=LDMInferenceConfig(),
        args=args,
    )
    device = select_device(args.device)

    inference(
        target_font_path=args.target_font_path,
        reference_fonts_dir=args.reference_fonts_dir,
        vqvae_model_config=vqvae_model_config,
        ldm_model_config=ldm_model_config,
        font_processing_config=font_processing_config,
        ldm_inference_config=ldm_inference_config,
        charset_path=args.charset_path,
        device=device,
    )


if __name__ == "__main__":
    main()
