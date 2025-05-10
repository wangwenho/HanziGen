import warnings
from pathlib import Path

from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from rich.console import Console
from rich.table import Table
from tqdm.rich import tqdm

from configs.font_processing_config import FontProcessingConfig
from utils.font.font_utils import (
    get_charset_paths,
    get_font_paths,
    read_charset_from_file,
    write_charset_to_file,
)

warnings.filterwarnings("ignore")


class FontCoverageAnalyzer:
    """
    Analyzes how many Hanzi characters a font supports.
    """

    # ===== Initialization =====
    def __init__(
        self,
        font_path: str | Path,
        font_processing_config: FontProcessingConfig,
    ):
        self.font_path = Path(font_path)
        self.font_name = self.font_path.stem
        self.config = font_processing_config
        self._font = None
        self._glyph_set = None
        self._best_cmap = None

    @classmethod
    def from_target_font(
        cls,
        target_font_path: str | Path,
        font_processing_config: FontProcessingConfig,
    ) -> "FontCoverageAnalyzer":
        """
        Create an analyzer for the given target font file.
        """
        return cls(target_font_path, font_processing_config)

    @classmethod
    def from_reference_fonts(
        cls,
        reference_fonts_dir: str | Path,
        font_processing_config: FontProcessingConfig,
    ) -> list["FontCoverageAnalyzer"]:
        """
        Create analyzers for all font files in the given reference fonts directory.
        """
        font_paths = get_font_paths(reference_fonts_dir)
        return [cls(font_path, font_processing_config) for font_path in font_paths]

    def _load_font(
        self,
    ) -> None:
        """
        Load the font file and initialize the glyph set and best cmap.
        """
        if self._font is None:
            self._font = TTFont(self.font_path)
            self._glyph_set = self._font.getGlyphSet()
            self._best_cmap = self._font.getBestCmap()

    # ===== Font Analyzation =====
    def analyze_coverage(
        self,
    ) -> None:
        """
        Analyze the jf7000 and unihan coverage for the font file.
        """
        jf7000_covered_charset, jf7000_missing_charset = self._split_charset_coverage(
            charset_path=self.config.jf7000_charset_path,
        )
        unihan_covered_charset, unihan_missing_charset = self._split_charset_coverage(
            charset_path=self.config.unihan_charset_path,
        )

        self._print_coverage_statistics(
            charset_dir=self.config.jf7000_charset_dir,
        )
        self._print_coverage_statistics(
            charset_dir=self.config.unihan_charset_dir,
        )
        output_dir = Path(self.config.charset_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_mappings = {
            f"jf7000_coverage/{self.font_name}/covered.txt": jf7000_covered_charset,
            f"jf7000_coverage/{self.font_name}/missing.txt": jf7000_missing_charset,
            f"unihan_coverage/{self.font_name}/covered.txt": unihan_covered_charset,
            f"unihan_coverage/{self.font_name}/missing.txt": unihan_missing_charset,
        }

        for rel_path, charset in output_mappings.items():
            output_path = output_dir / rel_path
            write_charset_to_file(charset, output_path)

    def _split_charset_coverage(
        self,
        charset_path: Path,
    ) -> tuple[set[str], set[str]]:
        """
        Split the charset into covered and missing charsets.
        """
        self._load_font()
        charset = read_charset_from_file(charset_path)

        covered_charset = set()
        missing_charset = set()

        for char in charset:
            glyph_name = self._best_cmap.get(ord(char))
            if self._is_glyph_empty_or_missing(self._glyph_set, glyph_name):
                missing_charset.add(char)
            else:
                covered_charset.add(char)

        return covered_charset, missing_charset

    # ===== Font processing =====
    def _is_glyph_empty_or_missing(
        self,
        glyph_set: set,
        glyph_name: str,
    ) -> bool:
        """
        Check if the glyph is empty or missing in the font file.
        """
        if (glyph_name is None) or (glyph_name not in glyph_set):
            return True

        glyph = glyph_set[glyph_name]
        pen = RecordingPen()
        glyph.draw(pen)
        return len(pen.value) == 0

    # ===== Logging =====
    def _print_coverage_statistics(
        self,
        charset_dir: Path,
    ) -> None:
        """
        Show coverage statistics for the given charset directory.
        """
        console = Console()
        table = Table(title=f"{self.font_name} {charset_dir.name} Coverage Statistics")

        table.add_column("Charset", justify="left", style="cyan", no_wrap=True)
        table.add_column("Total", justify="right", style="yellow")
        table.add_column("Covered", justify="right", style="magenta")
        table.add_column("Missing", justify="right", style="green")
        table.add_column("Covered Ratio", justify="right", style="magenta")
        table.add_column("Missing Ratio", justify="right", style="green")

        charset_paths = get_charset_paths(charset_dir)

        for charset_path in tqdm(charset_paths, desc="Analyzing"):
            covered_glyphs, missing_glyphs = self._split_charset_coverage(charset_path)
            total = len(covered_glyphs) + len(missing_glyphs)
            covered_ratio = (
                f"{(len(covered_glyphs) / total * 100):.2f}%" if total > 0 else "N/A"
            )
            missing_ratio = (
                f"{(len(missing_glyphs) / total * 100):.2f}%" if total > 0 else "N/A"
            )
            table.add_row(
                charset_path.name,
                str(total),
                str(len(covered_glyphs)),
                str(len(missing_glyphs)),
                covered_ratio,
                missing_ratio,
            )

        console.print(table)
