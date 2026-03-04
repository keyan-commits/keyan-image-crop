"""
Content detection strategies for finding invoice cards in screenshots.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from PIL import Image

from models import BoundingBox
from helpers import DetectionError, largest_contiguous_block


class ContentDetector(ABC):
    """Abstract strategy for detecting the content region in a screenshot."""

    @abstractmethod
    def detect(self, image: Image.Image) -> BoundingBox:
        ...


class CardBoundaryDetector(ContentDetector):
    """
    Detects a white invoice card on a light-gray app background.
    """

    DEFAULT_BRIGHTNESS_FLOOR = 200
    DEFAULT_WHITE_THRESHOLD = 250
    DEFAULT_COL_WHITE_RATIO = 0.15
    DEFAULT_ROW_WHITE_RATIO = 0.10
    DEFAULT_CONTENT_THRESHOLD = 200
    DEFAULT_CONTENT_ROW_RATIO = 0.005
    DEFAULT_BOTTOM_MARGIN_RATIO = 0.03

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

    def detect(self, image: Image.Image) -> BoundingBox:
        gray = np.array(image.convert("L"))
        h, w = gray.shape

        zone_top, zone_bottom = self._find_content_zone(gray)
        left, right = self._find_card_columns(gray, zone_top, zone_bottom)
        card_top, card_bottom = self._find_card_rows(gray, zone_top, zone_bottom, left, right)

        # Trim top shadow
        trimmed_top = self._trim_top(gray, card_top, card_bottom, left, right)

        # Trim bottom whitespace
        trimmed_bottom = self._trim_bottom(gray, card_top, card_bottom, left, right)

        # Trim side shadows using the trimmed_top row (guaranteed pure white, no text)
        trimmed_left, trimmed_right = self._trim_sides(gray, trimmed_top, left, right)

        return BoundingBox(left=trimmed_left, top=trimmed_top, right=trimmed_right, bottom=trimmed_bottom)

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
        Find the card's horizontal span.
        """
        zone = gray[zone_top:zone_bottom + 1, :]
        white = zone > self._white_threshold
        col_white_frac = white.mean(axis=0)

        card_col_mask = col_white_frac > self._col_white_ratio
        card_cols = np.where(card_col_mask)[0]

        if len(card_cols) < 10:
            raise DetectionError("Could not detect card columns")

        block_start, block_end = largest_contiguous_block(card_col_mask)

        return block_start, block_end

    def _find_card_rows(
        self, gray: np.ndarray, zone_top: int, zone_bottom: int, left: int, right: int
    ) -> Tuple[int, int]:
        """
        Find the card's vertical span using the largest contiguous white block.
        """
        zone = gray[zone_top:zone_bottom + 1, left:right + 1]
        white = zone > self._white_threshold
        row_white_frac = white.mean(axis=1)

        card_row_mask = row_white_frac > self._row_white_ratio
        block_start, block_end = largest_contiguous_block(card_row_mask)

        return zone_top + block_start, zone_top + block_end

    def _trim_top(
        self, gray: np.ndarray, card_top: int, card_bottom: int, left: int, right: int
    ) -> int:
        """Find the first row inside the card that is pure white (past the shadow)."""
        pure_white = 253
        for row in range(card_top, card_top + (card_bottom - card_top) // 4):
            row_min = np.percentile(gray[row, left:right + 1], 5)
            if row_min >= pure_white:
                return row
        return card_top

    def _trim_sides(
        self, gray: np.ndarray, trimmed_top: int, left: int, right: int
    ) -> Tuple[int, int]:
        """
        Trim side shadows using a known pure-white row (trimmed_top).

        The pixel pattern from left to right is:
        outer margin (255) → shadow (232-254) → card interior (255) → shadow → outer margin (255)

        Scan from each side: skip initial white (outer margin), pass through shadow,
        then stop when we hit white again (card interior).
        """
        row = gray[trimmed_top, left:right + 1]
        pure_white = 253

        # Scan from left: skip outer margin, pass shadow, find card interior
        in_shadow = False
        trimmed_left = left
        for i, val in enumerate(row):
            if not in_shadow and val < pure_white:
                in_shadow = True
            elif in_shadow and val >= pure_white:
                trimmed_left = left + i
                break

        # Scan from right: skip outer margin, pass shadow, find card interior
        in_shadow = False
        trimmed_right = right
        for i in range(len(row) - 1, -1, -1):
            if not in_shadow and row[i] < pure_white:
                in_shadow = True
            elif in_shadow and row[i] >= pure_white:
                trimmed_right = left + i
                break

        return trimmed_left, trimmed_right

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

        has_content = card < self.DEFAULT_CONTENT_THRESHOLD
        row_content_frac = has_content.sum(axis=1) / card_width

        content_rows = np.where(row_content_frac > self.DEFAULT_CONTENT_ROW_RATIO)[0]

        if len(content_rows) == 0:
            return card_bottom

        last_content_row = int(content_rows[-1])
        margin = int(card_height * self.DEFAULT_BOTTOM_MARGIN_RATIO)

        return min(card_bottom, card_top + last_content_row + margin)
