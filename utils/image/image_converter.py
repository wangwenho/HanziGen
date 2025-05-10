from pathlib import Path

import potrace
from PIL import Image
from potrace import POTRACE_TURNPOLICY_MINORITY, Bitmap
from tqdm.rich import tqdm

from configs.converting_config import ConvertingConfig

from .image_utils import get_image_paths


class GlyphImageConverter:
    """
    Converts images to SVG format.
    """

    def __init__(
        self,
        converting_config: ConvertingConfig,
    ):
        self.config = converting_config

    def convert_images_to_svgs(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        """
        Convert all images in the input directory to SVG format.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        img_paths = get_image_paths(input_dir)

        for img_path in tqdm(img_paths, desc="Converting images to SVG"):
            try:
                svg_path = output_dir / img_path.with_suffix(".svg").name
                self.convert_to_svg(img_path, svg_path)
            except Exception as e:
                print(f"[WARN] Failed to convert {img_path}: {e}")

    def convert_to_svg(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """
        Convert a single image to SVG format.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        image = Image.open(input_path).convert("L")
        bm = Bitmap(image, blacklevel=self.config.blacklevel)
        plist = bm.trace(
            turdsize=self.config.turdsize,
            turnpolicy=POTRACE_TURNPOLICY_MINORITY,
            alphamax=self.config.alphamax,
            opticurve=True,
            opttolerance=self.config.opttolerance,
        )

        svg_content = self._generate_svg_content(
            plist=plist,
            width=image.width,
            height=image.height,
        )

        with open(output_path, "w") as fp:
            fp.write(svg_content)

    @staticmethod
    def _generate_svg_content(
        plist: list[potrace.Path], width: int, height: int
    ) -> str:
        """
        Generate SVG content from the path list.
        """
        svg_header = (
            f"""<svg version="1.1" xmlns="http://www.w3.org/2000/svg" """
            f"""xmlns:xlink="http://www.w3.org/1999/xlink" """
            f"""width="{width}" height="{height}" viewBox="0 0 {width} {height}">"""
        )
        path_data = []
        for curve in plist:
            start_point = curve.start_point
            path_data.append(f"M{start_point.x},{start_point.y}")
            for segment in curve.segments:
                if segment.is_corner:
                    path_data.append(
                        f"L{segment.c.x},{segment.c.y}L{segment.end_point.x},{segment.end_point.y}"
                    )
                else:
                    path_data.append(
                        f"C{segment.c1.x},{segment.c1.y} {segment.c2.x},{segment.c2.y} {segment.end_point.x},{segment.end_point.y}"
                    )
            path_data.append("z")
        path_element = f"""<path stroke='none' fill='black' fill-rule='evenodd' d='{"".join(path_data)}'/>"""
        return f"{svg_header}{path_element}</svg>"
