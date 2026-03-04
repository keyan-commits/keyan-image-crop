"""
Tests for Invoice Screenshot Cropper
Run: python3 -m pytest test_crop_invoices.py -v
Install: pip3 install pytest Pillow numpy
"""

import pytest
import numpy as np
from PIL import Image

from crop_invoices import (
    BoundingBox,
    BatchResult,
    CropResult,
    ContentDetector,
    CardBoundaryDetector,
    ImageCropper,
    BatchProcessor,
    DetectionError,
    find_contiguous_blocks,
    largest_contiguous_block,
)


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
) -> Image.Image:
    """
    Create a synthetic phone screenshot:
    - Dark status bar at top
    - White full-width header (simulates X / Saving...)
    - Gray divider line
    - Gray background with a white card (narrower than full width)
    - Gray space below card
    - Dark nav bar at bottom

    Args:
        text_end_ratio: How far down the card text extends (0.0-1.0).
                        e.g. 0.6 means text fills the top 60%, bottom 40% is empty white.
    """
    arr = np.full((height, width), bg_color, dtype=np.uint8)

    # Dark status bar
    arr[:bar_height, :] = 50

    # White full-width header (below status bar)
    arr[bar_height:bar_height + header_height, :] = 255

    # Gray divider
    arr[bar_height + header_height:bar_height + header_height + 4, :] = 200

    # Dark nav bar at bottom
    arr[height - bar_height:, :] = 80

    # White card (narrower than full width, on gray background)
    l, t, r, b = card_rect
    arr[t:b, l:r] = card_color

    # Add realistic text: short dark segments in the top portion of the card
    card_height = b - t
    card_width = r - l
    text_width = int(card_width * 0.4)
    text_start = l + 20
    text_bottom = t
