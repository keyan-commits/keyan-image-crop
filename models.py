"""
Value objects and result types for the invoice cropper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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
