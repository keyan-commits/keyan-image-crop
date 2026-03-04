"""
Utility functions for contiguous block detection.
"""

from typing import List, Tuple

import numpy as np


class DetectionError(Exception):
    """Raised when content detection fails."""
    pass


def find_contiguous_blocks(mask: np.ndarray) -> List[Tuple[int, int]]:
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
    blocks = find_contiguous_blocks(mask)
    if not blocks:
        raise DetectionError("No contiguous blocks found")
    return max(blocks, key=lambda b: b[1] - b[0])
