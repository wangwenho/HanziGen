from pathlib import Path


class ProjectValidationError(Exception):
    """
    Custom exception for project-specific path validation errors.
    """

    pass


class ProjectValidator:
    """
    Project-specific path validator for HanziGen.
    """

    @staticmethod
    def validate_font_file(
        file_path: str | Path,
        name: str = "Font file",
    ) -> None:
        """
        Validate that a font file exists and has correct format.
        """
        if not file_path:
            return

        file_path = Path(file_path)

        if not file_path.exists():
            raise ProjectValidationError(f"{name} not found: {file_path}")

        if not file_path.is_file():
            raise ProjectValidationError(f"{name} is not a file: {file_path}")

        if file_path.suffix.lower() not in [".ttf", ".otf"]:
            raise ProjectValidationError(
                f"Invalid {name.lower()} format: {file_path} (expected .ttf or .otf)"
            )

    @staticmethod
    def validate_fonts_directory(
        dir_path: str | Path,
        name: str = "Fonts directory",
    ) -> None:
        """
        Validate that a directory exists and contains font files.
        """
        if not dir_path:
            return

        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise ProjectValidationError(f"{name} not found: {dir_path}")

        if not dir_path.is_dir():
            raise ProjectValidationError(f"{name} is not a directory: {dir_path}")

        font_files = list(dir_path.glob("*.ttf")) + list(dir_path.glob("*.otf"))
        if not font_files:
            raise ProjectValidationError(
                f"No font files (.ttf/.otf) found in {name.lower()}: {dir_path}"
            )

    @staticmethod
    def validate_directory(
        dir_path: str | Path,
        name: str = "Directory",
    ) -> None:
        """
        Validate that a directory exists.
        """
        if not dir_path:
            return

        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise ProjectValidationError(f"{name} not found: {dir_path}")

        if not dir_path.is_dir():
            raise ProjectValidationError(f"{name} is not a directory: {dir_path}")

    @staticmethod
    def validate_charset_file(
        file_path: str | Path,
        name: str = "Charset file",
    ) -> None:
        """
        Validate that a charset file exists.
        """
        if not file_path:
            return

        file_path = Path(file_path)

        if not file_path.exists():
            raise ProjectValidationError(f"{name} not found: {file_path}")

        if not file_path.is_file():
            raise ProjectValidationError(f"{name} is not a file: {file_path}")

    @staticmethod
    def validate_checkpoint_file(
        file_path: str | Path,
        name: str = "Checkpoint file",
    ) -> None:
        """
        Validate that a checkpoint file exists.
        """
        if not file_path:
            return

        file_path = Path(file_path)

        if not file_path.exists():
            raise ProjectValidationError(f"{name} not found: {file_path}")

        if not file_path.is_file():
            raise ProjectValidationError(f"{name} is not a file: {file_path}")

        if file_path.suffix.lower() != ".pth":
            raise ProjectValidationError(
                f"Invalid {name.lower()} format: {file_path} (expected .pth)"
            )
