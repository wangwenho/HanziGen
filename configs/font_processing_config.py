from dataclasses import dataclass
from pathlib import Path


@dataclass
class FontProcessingConfig:
    """
    Configuration class for font analysis and dataset preparation settings.
    """

    charset_root: str = "charsets"

    data_root: str = "data"
    img_size: tuple[int, int] = (512, 512)
    sample_ratio: float = 1.0
    num_workers: int = 4

    @property
    def jf7000_charset_dir(self) -> Path:
        return Path(self.charset_root) / "jf7000"

    @property
    def unihan_charset_dir(self) -> Path:
        return Path(self.charset_root) / "unihan"

    @property
    def jf7000_charset_path(self) -> Path:
        return self.jf7000_charset_dir / "jf7000_all.txt"

    @property
    def unihan_charset_path(self) -> Path:
        return self.unihan_charset_dir / "unihan_all.txt"

    @property
    def jf7000_coverage_charset_dir(self) -> Path:
        return Path(self.charset_root) / "jf7000_coverage"

    @property
    def unihan_coverage_charset_dir(self) -> Path:
        return Path(self.charset_root) / "unihan_coverage"
