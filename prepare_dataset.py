import argparse

from configs import FontProcessingConfig
from utils.argparse.argparse_utils import update_config_from_args
from utils.image import GlyphImageGenerator


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Prepare image dataset for training")
    parser.add_argument("--target_font_path", type=str, help="Target font path")
    parser.add_argument(
        "--reference_fonts_dir", type=str, help="Reference fonts directory"
    )
    parser.add_argument("--source_charset_path", type=str, help="Source charset path")
    parser.add_argument(
        "--img_size", type=int, nargs=2, help="Image size (width height)"
    )
    parser.add_argument("--sample_ratio", type=float, help="Sampling ratio (0-1)")

    return parser.parse_args()


def prepare_image_dataset(
    target_font_path: str,
    reference_fonts_dir: str,
    source_charset_path: str,
    font_processing_config: FontProcessingConfig,
) -> None:
    """ """
    tgt_generator = GlyphImageGenerator.from_target_font(
        target_font_path=target_font_path,
        font_processing_config=font_processing_config,
    )
    ref_generators = GlyphImageGenerator.from_reference_fonts(
        reference_fonts_dir=reference_fonts_dir,
        font_processing_config=font_processing_config,
    )

    tgt_generator.generate_glyph_images(
        source_charset_path=source_charset_path,
        font_role="target",
    )
    for ref_generator in ref_generators:
        ref_generator.generate_glyph_images(
            source_charset_path=source_charset_path,
            font_role="reference",
        )


def main() -> None:
    """ """
    args = parse_args()
    font_processing_config = update_config_from_args(
        converting_config=FontProcessingConfig(),
        args=args,
    )

    prepare_image_dataset(
        target_font_path=args.target_font_path,
        reference_fonts_dir=args.reference_fonts_dir,
        source_charset_path=args.source_charset_path,
        font_processing_config=font_processing_config,
    )


if __name__ == "__main__":
    main()
