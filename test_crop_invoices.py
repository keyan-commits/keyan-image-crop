"""
Tests for Invoice Screenshot Cropper
Run: python3 -m pytest test_crop_invoices.py -v
Install: pip3 install pytest Pillow numpy
"""

import pytest
import numpy as np
from PIL import Image

from models import BoundingBox, BatchResult, CropResult
from helpers import DetectionError, find_contiguous_blocks, largest_contiguous_block
from detectors import ContentDetector, CardBoundaryDetector
from cropper import ImageCropper
from processor import BatchProcessor


# =============================================================================
# BoundingBox Tests
# =============================================================================

class TestBoundingBox:
    def test_valid_creation(self):
        box = BoundingBox(left=10, top=20, right=100, bottom=200)
        assert box.left == 10
        assert box.width == 90
        assert box.height == 180

    def test_invalid_left_right_raises(self):
        with pytest.raises(ValueError, match="left"):
            BoundingBox(left=100, top=0, right=50, bottom=100)

    def test_invalid_top_bottom_raises(self):
        with pytest.raises(ValueError, match="top"):
            BoundingBox(left=0, top=100, right=50, bottom=50)

    def test_as_tuple(self):
        box = BoundingBox(left=10, top=20, right=30, bottom=40)
        assert box.as_tuple() == (10, 20, 30, 40)

    def test_with_padding(self):
        box = BoundingBox(left=50, top=50, right=200, bottom=200)
        padded = box.with_padding(10, max_width=300, max_height=300)
        assert padded == BoundingBox(left=40, top=40, right=210, bottom=210)

    def test_with_padding_clamped_to_bounds(self):
        box = BoundingBox(left=5, top=5, right=295, bottom=295)
        padded = box.with_padding(10, max_width=300, max_height=300)
        assert padded.left == 0
        assert padded.top == 0
        assert padded.right == 300
        assert padded.bottom == 300

    def test_is_immutable(self):
        box = BoundingBox(left=0, top=0, right=10, bottom=10)
        with pytest.raises(AttributeError):
            box.left = 5


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestContiguousBlocks:
    def test_single_block(self):
        mask = np.array([False, True, True, True, False])
        assert find_contiguous_blocks(mask) == [(1, 3)]

    def test_multiple_blocks(self):
        mask = np.array([True, True, False, True, False, True, True, True])
        blocks = find_contiguous_blocks(mask)
        assert blocks == [(0, 1), (3, 3), (5, 7)]

    def test_no_blocks(self):
        mask = np.array([False, False, False])
        assert find_contiguous_blocks(mask) == []

    def test_all_true(self):
        mask = np.array([True, True, True])
        assert find_contiguous_blocks(mask) == [(0, 2)]

    def test_block_at_end(self):
        mask = np.array([False, True, True])
        assert find_contiguous_blocks(mask) == [(1, 2)]

    def test_largest_block(self):
        mask = np.array([True, True, False, True, True, True, False, True])
        start, end = largest_contiguous_block(mask)
        assert (start, end) == (3, 5)

    def test_largest_block_raises_on_empty(self):
        mask = np.array([False, False])
        with pytest.raises(DetectionError):
            largest_contiguous_block(mask)


# =============================================================================
# Result Model Tests
# =============================================================================

class TestBatchResult:
    def test_empty_result(self):
        result = BatchResult()
        assert result.success_count == 0
        assert result.failure_count == 0

    def test_mixed_results(self):
        result = BatchResult(results=[
            CropResult(filename="a.png", success=True),
            CropResult(filename="b.png", success=False, error="fail"),
            CropResult(filename="c.png", success=True),
        ])
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.failures[0].filename == "b.png"


# =============================================================================
# Synthetic Screenshot Builder
# =============================================================================

def _make_screenshot(
    width=400, height=1200,
    card_rect=(50, 300, 350, 900),
    bg_color=240, card_color=255,
    bar_height=80, header_height=60,
    text_end_ratio=0.6,
    shadow_width=8, shadow_color=235,
) -> Image.Image:
    """
    Create a synthetic phone screenshot with realistic card shadow:
    - Dark status bar at top
    - White full-width header
    - Gray divider
    - Gray background with a white card (with shadow borders)
    - Dark nav bar at bottom
    """
    arr = np.full((height, width), bg_color, dtype=np.uint8)

    # Dark status bar
    arr[:bar_height, :] = 50

    # White full-width header
    arr[bar_height:bar_height + header_height, :] = 255

    # Gray divider
    arr[bar_height + header_height:bar_height + header_height + 4, :] = 200

    # Dark nav bar at bottom
    arr[height - bar_height:, :] = 80

    # White card
    l, t, r, b = card_rect
    arr[t:b, l:r] = card_color

    # Add shadow on card edges (left, right, top)
    arr[t:t + shadow_width, l:r] = shadow_color          # top shadow
    arr[t:b, l:l + shadow_width] = shadow_color           # left shadow
    arr[t:b, r - shadow_width:r] = shadow_color           # right shadow

    # Add text inside card
    card_height = b - t
    card_w = r - l
    text_width = int(card_w * 0.4)
    text_start = l + shadow_width + 20
    text_bottom = t + shadow_width + int(card_height * text_end_ratio)

    for row in range(t + shadow_width + 30, text_bottom, 50):
        arr[row, text_start:text_start + text_width] = 60

    return Image.fromarray(arr).convert("RGB")


# =============================================================================
# Detector Tests
# =============================================================================

class TestCardBoundaryDetector:
    def test_detects_card_in_synthetic_screenshot(self):
        img = _make_screenshot(card_rect=(50, 300, 350, 900))
        detector = CardBoundaryDetector()
        box = detector.detect(img)

        # Should be inside the shadow, not on the outer edge
        assert box.left > 55
        assert box.right < 345
        assert box.top > 305
        assert box.bottom >= 600

    def test_excludes_status_bar(self):
        img = _make_screenshot()
        detector = CardBoundaryDetector()
        box = detector.detect(img)
        assert box.top > 80

    def test_excludes_nav_bar(self):
        img = _make_screenshot()
        detector = CardBoundaryDetector()
        box = detector.detect(img)
        assert box.bottom < 1120

    def test_excludes_full_width_header(self):
        img = _make_screenshot()
        detector = CardBoundaryDetector()
        box = detector.detect(img)
        assert box.top > 200

    def test_smaller_card(self):
        img = _make_screenshot(card_rect=(80, 350, 320, 600))
        detector = CardBoundaryDetector()
        box = detector.detect(img)
        assert box.width < 300
        assert box.height < 350

    def test_raises_on_all_gray_image(self):
        arr = np.full((800, 400), 200, dtype=np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        detector = CardBoundaryDetector()
        with pytest.raises(DetectionError):
            detector.detect(img)

    def test_custom_white_threshold(self):
        img = _make_screenshot(card_color=248)
        strict = CardBoundaryDetector(white_threshold=252)
        relaxed = CardBoundaryDetector(white_threshold=245)

        with pytest.raises(DetectionError):
            strict.detect(img)

        box = relaxed.detect(img)
        assert box.width > 100


# =============================================================================
# Side Trim Tests
# =============================================================================

class TestSideTrim:
    """Tests for left/right shadow trimming."""

    def test_trims_left_shadow(self):
        img = _make_screenshot(card_rect=(50, 300, 350, 900), shadow_width=10)
        detector = CardBoundaryDetector()
        box = detector.detect(img)
        assert box.left >= 60

    def test_trims_right_shadow(self):
        img = _make_screenshot(card_rect=(50, 300, 350, 900), shadow_width=10)
        detector = CardBoundaryDetector()
        box = detector.detect(img)
        assert box.right <= 340

    def test_wider_shadow_trimmed(self):
        narrow = _make_screenshot(shadow_width=5)
        wide = _make_screenshot(shadow_width=15)
        detector = CardBoundaryDetector()

        box_narrow = detector.detect(narrow)
        box_wide = detector.detect(wide)

        assert box_wide.width < box_narrow.width

    def test_symmetric_trim(self):
        img = _make_screenshot(card_rect=(100, 300, 300, 900), shadow_width=10)
        detector = CardBoundaryDetector()
        box = detector.detect(img)

        left_margin = box.left - 100
        right_margin = 300 - box.right
        assert abs(left_margin - right_margin) < 5


# =============================================================================
# Top Trim Tests
# =============================================================================

class TestTopTrim:
    """Tests for top shadow trimming."""

    def test_trims_top_shadow(self):
        img = _make_screenshot(card_rect=(50, 300, 350, 900), shadow_width=10)
        detector = CardBoundaryDetector()
        box = detector.detect(img)
        assert box.top >= 310

    def test_wider_top_shadow_trimmed_more(self):
        narrow = _make_screenshot(shadow_width=5)
        wide = _make_screenshot(shadow_width=15)
        detector = CardBoundaryDetector()

        box_narrow = detector.detect(narrow)
        box_wide = detector.detect(wide)

        assert box_wide.top > box_narrow.top


# =============================================================================
# Bottom Trim Tests
# =============================================================================

class TestBottomTrim:
    """Tests for the bottom whitespace trimming behavior."""

    def test_trims_empty_bottom_whitespace(self):
        card_rect = (50, 300, 350, 900)
        img = _make_screenshot(card_rect=card_rect, text_end_ratio=0.5)
        detector = CardBoundaryDetector()
        box = detector.detect(img)

        card_top, card_bottom_full = card_rect[1], card_rect[3]
        card_height = card_bottom_full - card_top
        assert box.bottom < card_top + int(card_height * 0.7)

    def test_more_text_means_taller_crop(self):
        card_rect = (50, 300, 350, 900)
        img_short = _make_screenshot(card_rect=card_rect, text_end_ratio=0.4)
        img_tall = _make_screenshot(card_rect=card_rect, text_end_ratio=0.8)

        detector = CardBoundaryDetector()
        box_short = detector.detect(img_short)
        box_tall = detector.detect(img_tall)

        assert box_tall.height > box_short.height

    def test_full_text_card_keeps_most_of_card(self):
        card_rect = (50, 300, 350, 900)
        img = _make_screenshot(card_rect=card_rect, text_end_ratio=0.95)
        detector = CardBoundaryDetector()
        box = detector.detect(img)

        card_height = card_rect[3] - card_rect[1]
        assert box.height >= card_height * 0.75

    def test_trim_preserves_margin_below_content(self):
        card_rect = (50, 300, 350, 900)
        img = _make_screenshot(card_rect=card_rect, text_end_ratio=0.5)
        detector = CardBoundaryDetector()
        box = detector.detect(img)

        # Bottom should be near where text ends, not at the card's full extent
        card_height = card_rect[3] - card_rect[1]
        assert box.bottom < card_rect[1] + int(card_height * 0.7)
        assert box.bottom > card_rect[1] + int(card_height * 0.4)


# =============================================================================
# ImageCropper Tests
# =============================================================================

class TestImageCropper:
    def test_crop_reduces_size(self):
        img = _make_screenshot()
        cropper = ImageCropper(detector=CardBoundaryDetector(), padding=0)
        cropped = cropper.crop(img)
        assert cropped.width < img.width
        assert cropped.height < img.height

    def test_crop_with_padding_is_larger(self):
        img = _make_screenshot()
        no_pad = ImageCropper(detector=CardBoundaryDetector(), padding=0).crop(img)
        padded = ImageCropper(detector=CardBoundaryDetector(), padding=20).crop(img)
        assert padded.width >= no_pad.width
        assert padded.height >= no_pad.height

    def test_accepts_custom_detector(self):
        class FixedDetector(ContentDetector):
            def detect(self, image):
                return BoundingBox(left=10, top=10, right=100, bottom=100)

        img = _make_screenshot()
        cropped = ImageCropper(detector=FixedDetector(), padding=0).crop(img)
        assert cropped.width == 90
        assert cropped.height == 90


# =============================================================================
# BatchProcessor Tests
# =============================================================================

class TestBatchProcessor:
    def _make_processor(self):
        return BatchProcessor(
            cropper=ImageCropper(detector=CardBoundaryDetector())
        )

    def test_process_empty_directory(self, tmp_path):
        (tmp_path / "input").mkdir()
        result = self._make_processor().process(tmp_path / "input", tmp_path / "output")
        assert result.success_count == 0

    def test_process_valid_images(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["inv1.png", "inv2.jpg"]:
            _make_screenshot().save(input_dir / name)

        result = self._make_processor().process(input_dir, tmp_path / "output")
        assert result.success_count == 2
        assert result.failure_count == 0
        assert len(list((tmp_path / "output").iterdir())) == 2

    def test_skips_non_image_files(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "notes.txt").write_text("not an image")
        _make_screenshot().save(input_dir / "inv.png")

        result = self._make_processor().process(input_dir, tmp_path / "output")
        assert result.success_count == 1

    def test_output_suffix(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _make_screenshot().save(input_dir / "test.png")

        processor = BatchProcessor(
            cropper=ImageCropper(detector=CardBoundaryDetector()),
            output_suffix="_trimmed",
        )
        processor.process(input_dir, tmp_path / "output")
        assert (tmp_path / "output" / "test_trimmed.png").exists()

    def test_creates_output_directory(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _make_screenshot().save(input_dir / "test.png")

        output_dir = tmp_path / "nested" / "output"
        self._make_processor().process(input_dir, output_dir)
        assert output_dir.exists()
