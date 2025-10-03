from pathlib import Path

from tqdm.rich import tqdm


def get_font_paths(
    font_dir: str | Path,
) -> list[Path]:
    """
    Return all TTF and OTF file paths in the given directory.
    """
    font_dir = Path(font_dir)
    return sorted(
        [
            *font_dir.glob("*.ttf"),
            *font_dir.glob("*.otf"),
        ]
    )
