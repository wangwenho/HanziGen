import argparse

from configs import FontProcessingConfig
from utils.argparse.argparse_utils import update_config_from_args
from utils.font import FontCoverageAnalyzer


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Analyze font hanzi coverage")
    parser.add_argument("--target_font_path", type=str, help="Target font path")
    parser.add_argument(
        "--reference_fonts_dir", type=str, help="Reference fonts directory"
    )
    parser.add_argument(
        "--analyze_target_font", action="store_true", help="Analyze target font"
    )
    parser.add_argument(
        "--analyze_reference_fonts", action="store_true", help="Analyze reference fonts"
    )

    return parser.parse_args()


def analyze_target_font(
    target_font_path: str,
    font_processing_config: FontProcessingConfig,
) -> None:
    """ """
    analyzer = FontCoverageAnalyzer.from_target_font(
        target_font_path=target_font_path,
        font_processing_config=font_processing_config,
    )
    analyzer.analyze_coverage()


def analyze_reference_fonts(
    reference_fonts_dir: str,
    font_processing_config: FontProcessingConfig,
) -> None:
    """ """
    analyzers = FontCoverageAnalyzer.from_reference_fonts(
        reference_fonts_dir=reference_fonts_dir,
        font_processing_config=font_processing_config,
    )
    for analyzer in analyzers:
        analyzer.analyze_coverage()


def main() -> None:
    """ """
    args = parse_args()
    font_processing_config = update_config_from_args(
        converting_config=FontProcessingConfig(),
        args=args,
    )

    if args.analyze_target_font:
        analyze_target_font(
            target_font_path=args.target_font_path,
            font_processing_config=font_processing_config,
        )

    if args.analyze_reference_fonts:
        analyze_reference_fonts(
            reference_fonts_dir=args.reference_fonts_dir,
            font_processing_config=font_processing_config,
        )


if __name__ == "__main__":
    main()
