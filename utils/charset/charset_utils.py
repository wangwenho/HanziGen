from pathlib import Path


def generate_unicode_charset(
    start: int,
    end: int,
    file_path: str | Path,
) -> None:
    """
    Create a charset file from a range of Unicode code points.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        for code_point in range(start, end + 1):
            char = chr(code_point)
            file.write(f"{char}\n")


def merge_charsets(
    charset_paths: list[str | Path],
    file_path: str | Path,
) -> None:
    """
    Merge multiple charset files into one, removing duplicates.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    charset = set()
    for charset_path in charset_paths:
        charset.update(read_charset_from_file(charset_path))

    write_charset_to_file(charset, file_path)


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
