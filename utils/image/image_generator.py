import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from tqdm.rich import tqdm

from configs.font_processing_config import FontProcessingConfig
from utils.charset.charset_utils import read_charset_from_file
from utils.font.font_utils import get_font_paths


class GlyphImageGenerator:
    """
    Generates glyph images from a font file.
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

    @classmethod
    def from_target_font(
        cls,
        target_font_path: str | Path,
        font_processing_config: FontProcessingConfig,
    ) -> "GlyphImageGenerator":
        """
        Create a generator for the given target font file.
        """
        return cls(target_font_path, font_processing_config)

    @classmethod
    def from_reference_fonts(
        cls,
        reference_fonts_dir: str | Path,
        font_processing_config: FontProcessingConfig,
    ) -> list["GlyphImageGenerator"]:
        """
        Create generators for all font files in the given reference fonts directory.
        """
        font_paths = get_font_paths(reference_fonts_dir)
        return [cls(font_path, font_processing_config) for font_path in font_paths]

    # ===== Image generation =====
    def generate_glyph_images(
        self,
        source_charset_path: str | Path,
        font_role: Literal["target", "reference"],
    ) -> None:
        """
        Generate glyph images for the given font role ("target" or "reference").
        """
        num_workers = self.config.num_workers

        source_charset_path = Path(source_charset_path)
        covered_charset_path = (
            Path(self.config.unihan_coverage_charset_dir)
            / self.font_name
            / "covered.txt"
        )

        output_dir = Path(self.config.data_root) / font_role
        output_dir.mkdir(parents=True, exist_ok=True)

        selected_glyphs = self._sample_glyphs_for_charset(
            source_charset_path=source_charset_path,
            covered_charset_path=covered_charset_path,
            sample_ratio=self.config.sample_ratio,
        )

        self._process_glyphs_in_parallel(
            glyphs=selected_glyphs,
            output_dir=output_dir,
            image_size=self.config.img_size,
            num_workers=num_workers,
        )

    def save_glyph_image(
        self,
        char: str,
        output_dir: Path,
        img_size: tuple[int, int],
    ) -> None:
        """
        Save the glyph image to the specified directory.
        """
        image = self._render_glyph_to_image(char, img_size)
        image_path = output_dir / f"{ord(char):05X}.png"
        image.save(image_path)

    def _render_glyph_to_image(
        self,
        char: str,
        img_size: tuple[int, int],
    ) -> Image.Image:
        """
        Render a single glyph character to image.
        """
        image = Image.new("L", img_size, color=255)
        draw = ImageDraw.Draw(image)
        font_size = int(img_size[0] * 0.9)
        font = ImageFont.truetype(self.font_path, size=font_size)

        ascent, descent = font.getmetrics()
        total_font_height = ascent + descent

        bbox = draw.textbbox((0, 0), char, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        pos_x = (img_size[0] - width) / 2 - bbox[0]
        base_pos_y = (img_size[1] - height) / 2 - bbox[1]
        baseline_ratio = ascent / total_font_height

        adjustment = 0
        if baseline_ratio > 0.8:
            adjustment = (baseline_ratio - 0.75) * img_size[1] * 0.1
        elif baseline_ratio < 0.6:
            adjustment = (0.65 - baseline_ratio) * img_size[1] * 0.1

        pos_y = base_pos_y + adjustment
        draw.text((pos_x, pos_y), char, font=font, fill=0)

        return image

    def _process_glyphs_in_parallel(
        self,
        glyphs: list[str],
        output_dir: Path,
        image_size: tuple[int, int],
        num_workers: int,
    ) -> None:
        """
        Process and save glyph images in parallel using multiple threads.
        """
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    self.save_glyph_image,
                    glyph,
                    output_dir,
                    image_size,
                )
                for glyph in glyphs
            ]

            for future in tqdm(futures, desc="Saving images"):
                future.result()

    # ===== Glyph selection =====
    def _sample_glyphs_for_charset(
        self,
        source_charset_path: str | Path,
        covered_charset_path: str | Path,
        sample_ratio: float,
    ) -> set[str]:
        """
        Sample glyphs from the source charset based on the covered charset.
        """
        source_glyphs = read_charset_from_file(source_charset_path)
        covered_glyphs = read_charset_from_file(covered_charset_path)

        selected_glyphs = covered_glyphs.intersection(source_glyphs)

        num_sample = max(0, int(len(selected_glyphs) * sample_ratio))
        selected_glyphs = set(random.sample(list(selected_glyphs), num_sample))

        return selected_glyphs
