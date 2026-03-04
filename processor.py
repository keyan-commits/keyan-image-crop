"""
Batch processing of image files.
"""

import logging
from pathlib import Path
from typing import List

from PIL import Image

from models import CropResult, BatchResult
from helpers import DetectionError
from cropper import ImageCropper

SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes all supported images in a directory."""

    def __init__(self, cropper: ImageCropper, output_suffix: str = "_cropped"):
        self._cropper = cropper
        self._output_suffix = output_suffix

    def process(self, input_dir: Path, output_dir: Path) -> BatchResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        files = self._collect_files(input_dir)

        if not files:
            logger.info(f"No images found in {input_dir}")
            return BatchResult()

        logger.info(f"Processing {len(files)} image(s)...\n")
        result = BatchResult()

        for filepath in files:
            crop_result = self._process_single(filepath, output_dir)
            result.results.append(crop_result)
            self._log_result(crop_result)

        self._log_summary(result, output_dir)
        return result

    def _collect_files(self, input_dir: Path) -> List[Path]:
        return sorted(
            f for f in input_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _process_single(self, filepath: Path, output_dir: Path) -> CropResult:
        try:
            img = Image.open(filepath).convert("RGB")
            cropped = self._cropper.crop(img)

            out_path = output_dir / f"{filepath.stem}{self._output_suffix}{filepath.suffix}"
            cropped.save(out_path, quality=95)

            return CropResult(
                filename=filepath.name,
                success=True,
                original_size=(img.width, img.height),
                cropped_size=(cropped.width, cropped.height),
            )
        except (DetectionError, OSError) as e:
            return CropResult(filename=filepath.name, success=False, error=str(e))

    @staticmethod
    def _log_result(result: CropResult):
        if result.success:
            orig = f"{result.original_size[0]}x{result.original_size[1]}"
            crop = f"{result.cropped_size[0]}x{result.cropped_size[1]}"
            logger.info(f"  ✓ {result.filename}: {orig} → {crop}")
        else:
            logger.info(f"  ✗ {result.filename}: {result.error}")

    @staticmethod
    def _log_summary(result: BatchResult, output_dir: Path):
        logger.info(f"\nDone! {result.success_count} cropped, {result.failure_count} failed.")
        logger.info(f"Output: {output_dir.resolve()}")
