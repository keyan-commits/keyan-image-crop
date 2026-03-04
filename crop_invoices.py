"""
Invoice Screenshot Cropper
===========================
Detects and crops invoice cards from phone screenshots.

Architecture:
- Strategy Pattern: ContentDetector ABC with swappable detection strategies
- Single Responsibility: Each class has one job
- Dependency Injection: Cropper receives its detector
- Value Objects: BoundingBox as immutable dataclass

Usage:
    python crop_invoices.py -i ./invoices -o ./output
    python crop_invoices.py -i ./invoices -o ./output --padding 10
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================

@dataclass(frozen=True)
class BoundingBox:
    """Immutable value object representing a crop region."""
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self):
        if self.left >= self.right:
            raise ValueError(f"left ({self.left}) must be < right ({self.right})")
        if self.top >= self.bottom:
            raise ValueError(f"top ({self.top}) must be < bottom ({self.bottom})")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def with_padding(self, padding: int, max_width: int, max_height: int) -> BoundingBox:
        """Return a new BoundingBox expanded by padding, clamped to image bounds."""
        return BoundingBox(
            left=max(0, self.left - padding),
            top=max(0, self.top - padding),
            right=min(max_width, self.right + padding),
            bottom=min(max_height, self.bottom + padding),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass
class CropResult:
    """Result of a single image crop operation."""
    filename: str
    success: bool
    original_size: tuple[int, int] | None = None
    cropped_size: tuple[int, int] | None = None
    error: str | None = None


@dataclass
class BatchResult:
    """Aggregated result of batch processing."""
    results: List[CropResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def failures(self) -> List[CropResult]:
        return [r for r in self.results if not r.success]


# =============================================================================
# Helpers
# =============================================================================

def find_contiguous_blocks(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find all contiguous blocks of True values in a 1D boolean array.
    Returns list of (start_index, end_index) tuples (inclusive).
    """
    blocks = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(mask) - 1))
    return blocks


def largest_contiguous_block(mask: np.ndarray) -> Tuple[int, int]:
    """
    Find the largest contiguous block of True values in a 1D boolean array.
    Returns (start_index, end_index) inclusive.
    Raises DetectionError if no blocks found.
    """
    blocks = find_contiguous_blocks(mask)
    if not blocks:
        raise DetectionError("No contiguous blocks found")
    return max(blocks, key=lambda b: b[1] - b[0])


# =============================================================================
# Detectors (Strategy Pattern)
# =============================================================================

class ContentDetector(ABC):
    """Abstract strategy for detecting the content region in a screenshot."""

    @abstractmethod
    def detect(self, image: Image.Image) -> BoundingBox:
        """Detect the content bounding box in the given image."""
        ...


class CardBoundaryDetector(ContentDetector):
    """
    Detects a white invoice card on a light-gray app background.

    The challenge: the app header (X / Saving...) is also white and full-width,
    so simple white-pixel detection includes it. The card is narrower than the
    header and sits on a gray background with a subtle shadow/border.

    Strategy:
    1. Strip dark UI bars (status bar, nav bar) by brightness.
    2. In the remaining content zone, analyze per-column white density.
       Card columns have high white density; margin columns don't.
       This separates the card's horizontal span from the full-width header.
    3. Within card columns, find per-row white density and select the
       largest contiguous block of white rows — this is the card, excluding
       the shorter header block above it.
    """

    DEFAULT_BRIGHTNESS_FLOOR = 200
    DEFAULT_WHITE_THRESHOLD = 250
    DEFAULT_COL_WHITE_RATIO = 0.15
    DEFAULT_ROW_WHITE_RATIO = 0.10

    def __init__(
        self,
        brightness_floor: int = DEFAULT_BRIGHTNESS_FLOOR,
        white_threshold: int = DEFAULT_WHITE_THRESHOLD,
        col_white_ratio: float = DEFAULT_COL_WHITE_RATIO,
        row_white_ratio: float = DEFAULT_ROW_WHITE_RATIO,
    ):
        self._brightness_floor = brightness_floor
        self._white_threshold = white_threshold
        self._col_white_ratio = col_white_ratio
        self._row_white_ratio = row_white_ratio

    # Minimum brightness (0-255) for a pixel to count as "content" (text, lines)
    DEFAULT_CONTENT_THRESHOLD = 200
    # Minimum fraction of a row's pixels that must be content to count
    DEFAULT_CONTENT_ROW_RATIO = 0.005
    # Margin (as fraction of card height) to add below last content row
    DEFAULT_BOTTOM_MARGIN_RATIO = 0.03

    def detect(self, image: Image.Image) -> BoundingBox:
        gray = np.array(image.convert("L"))
        h, w = gray.shape

        zone_top, zone_bottom = self._find_content_zone(gray)
        left, right = self._find_card_columns(gray, zone_top, zone_bottom)
        card_top, card_bottom = self._find_card_rows(gray, zone_top, zone_bottom, left, right)

        # Trim bottom whitespace: find last row with actual content
        trimmed_bottom = self._trim_bottom(gray, card_top, card_bottom, left, right)

        return BoundingBox(left=left, top=card_top, right=right, bottom=trimmed_bottom)

    def _find_content_zone(self, gray: np.ndarray) -> Tuple[int, int]:
        """Exclude dark UI bars (status bar, nav bar) by row brightness."""
        h = gray.shape[0]
        row_mean = gray.mean(axis=1)
        content_mask = row_mean > self._brightness_floor
        content_rows = np.where(content_mask)[0]

        if len(content_rows) == 0:
            raise DetectionError("Could not find content zone — image too dark")

        return int(content_rows[0]), int(content_rows[-1])

    def _find_card_columns(
        self, gray: np.ndarray, zone_top: int, zone_bottom: int
    ) -> Tuple[int, int]:
        """
        Find the card's horizontal span by per-column white density.
        Card columns have high white density in the content zone;
        margin columns (gray background only) do not.
        """
        zone = gray[zone_top:zone_bottom + 1, :]
        white = zone > self._white_threshold
        col_white_frac = white.mean(axis=0)

        card_col_mask = col_white_frac > self._col_white_ratio
        card_cols = np.where(card_col_mask)[0]

        if len(card_cols) < 10:
            raise DetectionError("Could not detect card columns")

        return int(card_cols[0]), int(card_cols[-1])

    def _find_card_rows(
        self, gray: np.ndarray, zone_top: int, zone_bottom: int, left: int, right: int
    ) -> Tuple[int, int]:
        """
        Find the card's vertical span within its column range.
        Uses the largest contiguous block of white rows to isolate the card
        from the (shorter) header white area above it.
        """
        zone = gray[zone_top:zone_bottom + 1, left:right + 1]
        white = zone > self._white_threshold
        row_white_frac = white.mean(axis=1)

        card_row_mask = row_white_frac > self._row_white_ratio
        block_start, block_end = largest_contiguous_block(card_row_mask)

        # Map back to full image coordinates
        return zone_top + block_start, zone_top + block_end

    def _trim_bottom(
        self, gray: np.ndarray, card_top: int, card_bottom: int, left: int, right: int
    ) -> int:
        """
        Trim empty whitespace at the bottom of the card.
        Finds the last row with actual content (dark text pixels) and adds a small margin.
        """
        card = gray[card_top:card_bottom + 1, left:right + 1]
        card_width = right - left + 1
        card_height = card_bottom - card_top + 1

        # Content pixels: darker than threshold (text, lines, borders)
        has_content = card < self.DEFAULT_CONTENT_THRESHOLD
        row_content_frac = has_content.sum(axis=1) / card_width

        content_rows = np.where(row_content_frac > self.DEFAULT_CONTENT_ROW_RATIO)[0]

        if len(content_rows) == 0:
            return card_bottom

        last_content_row = int(content_rows[-1])
        margin = int(card_height * self.DEFAULT_BOTTOM_MARGIN_RATIO)

        return min(card_bottom, card_top + last_content_row + margin)


class DetectionError(Exception):
    """Raised when content detection fails."""
    pass


# =============================================================================
# Cropper
# =============================================================================

class ImageCropper:
    """Crops an image using a pluggable ContentDetector strategy."""

    def __init__(self, detector: ContentDetector, padding: int = 5):
        self._detector = detector
        self._padding = padding

    def crop(self, image: Image.Image) -> Image.Image:
        bounds = self._detector.detect(image)
        padded = bounds.with_padding(self._padding, image.width, image.height)
        return image.crop(padded.as_tuple())


# =============================================================================
# Batch Processor
# =============================================================================

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


# =============================================================================
# CLI
# =============================================================================

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-crop invoice screenshots")
    parser.add_argument("-i", "--input", default="./input", help="Input folder")
    parser.add_argument("-o", "--output", default="./output", help="Output folder")
    parser.add_argument("-p", "--padding", type=int, default=5, help="Padding in px (default: 5)")
    parser.add_argument(
        "-t", "--threshold", type=int, default=CardBoundaryDetector.DEFAULT_WHITE_THRESHOLD,
        help=f"White threshold 0-255 (default: {CardBoundaryDetector.DEFAULT_WHITE_THRESHOLD})",
    )
    return parser.parse_args(args)


def main(args=None):
    config = parse_args(args)

    detector = CardBoundaryDetector(white_threshold=config.threshold)
    cropper = ImageCropper(detector=detector, padding=config.padding)
    processor = BatchProcessor(cropper=cropper)

    processor.process(
        input_dir=Path(config.input),
        output_dir=Path(config.output),
    )


if __name__ == "__main__":
    main()
