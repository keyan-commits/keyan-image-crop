"""
Image cropper with pluggable detection strategy.
"""

from PIL import Image

from models import BoundingBox
from detectors import ContentDetector


class ImageCropper:
    """Crops an image using a pluggable ContentDetector strategy."""

    def __init__(self, detector: ContentDetector, padding: int = 0):
        self._detector = detector
        self._padding = padding

    def crop(self, image: Image.Image) -> Image.Image:
        bounds = self._detector.detect(image)
        padded = bounds.with_padding(self._padding, image.width, image.height)
        return image.crop(padded.as_tuple())
