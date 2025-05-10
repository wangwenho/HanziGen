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


def get_charset_paths(
    charset_dir: str | Path,
) -> list[Path]:
    """
    Return all charset file paths in the given directory.
    """
    charset_dir = Path(charset_dir)
    return sorted(
        [
            *charset_dir.glob("*.txt"),
        ]
    )


def read_charset_from_file(
    charset_path: str | Path,
) -> set[str]:
    """
    Read a charset file and return a set of characters.
    """
    charset_path = Path(charset_path)
    with charset_path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file.readlines() if len(line.strip()) == 1}


def write_charset_to_file(
    charset: set[str],
    file_path: str | Path,
) -> None:
    """
    Write a charset to a file, one character per line.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        for char in sorted(charset):
            file.write(f"{char}\n")
