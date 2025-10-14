import argparse
import sys

from configs import FontProcessingConfig
from utils.argparse.argparse_utils import update_config_from_args
from utils.font import FontCoverageAnalyzer
from utils.validation.project_validator import ProjectValidationError, ProjectValidator


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for font analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze font hanzi coverage",
    )
    parser.add_argument(
        "--target_font_path",
        type=str,
        help="Target font path",
    )
    parser.add_argument(
        "--reference_fonts_dir",
        type=str,
        help="Reference fonts directory",
    )
    parser.add_argument(
        "--analyze_target_font",
        action="store_true",
        help="Analyze target font",
    )
    parser.add_argument(
        "--analyze_reference_fonts",
        action="store_true",
        help="Analyze reference fonts",
    )

    return parser.parse_args()


def analyze_target_font(
    target_font_path: str,
    font_processing_config: FontProcessingConfig,
) -> None:
    """
    Analyze hanzi coverage of the target font.
    """
    analyzer = FontCoverageAnalyzer.from_target_font(
        target_font_path=target_font_path,
        font_processing_config=font_processing_config,
    )
    analyzer.analyze_coverage()


def analyze_reference_fonts(
    reference_fonts_dir: str,
    font_processing_config: FontProcessingConfig,
) -> None:
    """
    Analyze hanzi coverage of reference fonts in the specified directory.
    """
    analyzers = FontCoverageAnalyzer.from_reference_fonts(
        reference_fonts_dir=reference_fonts_dir,
        font_processing_config=font_processing_config,
    )
    for analyzer in analyzers:
        analyzer.analyze_coverage()


def main() -> None:
    """
    Main function to run the font analysis.
    """
    try:
        args = parse_args()
        font_processing_config = update_config_from_args(
            converting_config=FontProcessingConfig(),
            args=args,
        )

        if args.analyze_target_font:
            ProjectValidator.validate_font_file(
                file_path=args.target_font_path,
                name="Target font",
            )
            analyze_target_font(
                target_font_path=args.target_font_path,
                font_processing_config=font_processing_config,
            )

        if args.analyze_reference_fonts:
            ProjectValidator.validate_fonts_directory(
                dir_path=args.reference_fonts_dir,
                name="Reference fonts directory",
            )
            analyze_reference_fonts(
                reference_fonts_dir=args.reference_fonts_dir,
                font_processing_config=font_processing_config,
            )

        print("✅ Font analysis completed successfully")

    except ProjectValidationError as e:
        print(f"❌ {e}")
        print("❌ Font analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
