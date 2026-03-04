"""
CLI entry point for invoice screenshot cropper.
"""

import argparse
import logging
from pathlib import Path

from detectors import CardBoundaryDetector
from cropper import ImageCropper
from processor import BatchProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-crop invoice screenshots")
    parser.add_argument("-i", "--input", default="./input", help="Input folder")
    parser.add_argument("-o", "--output", default="./output", help="Output folder")
    parser.add_argument("-p", "--padding", type=int, default=0, help="Padding in px (default: 0)")
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
