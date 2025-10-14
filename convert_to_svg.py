import argparse
import sys

from configs import ConvertingConfig
from utils.argparse.argparse_utils import update_config_from_args
from utils.image import GlyphImageConverter
from utils.validation.project_validator import ProjectValidationError, ProjectValidator


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for SVG conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert images to SVG format",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--blacklevel",
        type=float,
        help="Black level",
    )
    parser.add_argument(
        "--turdsize",
        type=int,
        help="Suppress speckles of up to this size",
    )
    parser.add_argument(
        "--alphamax",
        type=float,
        help="Corner threshold parameter",
    )
    parser.add_argument(
        "--opttolerance",
        type=float,
        help="Curve optimization tolerance",
    )

    return parser.parse_args()


def convert_to_svg(
    input_dir: str,
    output_dir: str,
    converting_config: ConvertingConfig,
) -> None:
    """
    Convert images in the input directory to SVG format and save them in the output directory.
    """
    converter = GlyphImageConverter(
        converting_config=converting_config,
    )
    converter.convert_images_to_svgs(
        input_dir=input_dir,
        output_dir=output_dir,
    )


def main() -> None:
    """
    Main function to run the SVG conversion.
    """
    try:
        args = parse_args()

        ProjectValidator.validate_directory(
            dir_path=args.input_dir,
            name="Input directory",
        )

        converting_config = update_config_from_args(
            converting_config=ConvertingConfig(),
            args=args,
        )

        convert_to_svg(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            converting_config=converting_config,
        )

        print("✅ SVG conversion completed successfully")

    except ProjectValidationError as e:
        print(f"❌ {e}")
        print("❌ SVG conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
