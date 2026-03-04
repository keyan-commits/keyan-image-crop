# Invoice Screenshot Cropper

Auto-detect and crop invoice cards from mobile app screenshots. Strips status bars, navigation bars, and empty whitespace — leaving just the clean invoice.

## Before / After

| Input (screenshot) | Output (cropped) |
|---|---|
| Status bar, header, card, nav bar | Just the invoice card |

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Put screenshots in ./invoices, crop to ./output
python3 crop_invoices.py -i ./invoices -o ./output
```

## Options

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | `./input` | Input folder with screenshots |
| `-o`, `--output` | `./output` | Output folder for cropped images |
| `-p`, `--padding` | `5` | Padding around detected card (px) |
| `-t`, `--threshold` | `250` | White threshold 0-255 (lower = more tolerant) |

## How It Works

1. **Strip UI chrome** — removes dark status bar and navigation bar by row brightness
2. **Detect card columns** — finds the narrower white card vs full-width header using per-column white density
3. **Detect card rows** — isolates the largest contiguous white block (the card) from shorter white areas (header)
4. **Trim bottom whitespace** — finds the last row with actual text and adds a small margin

## Architecture

- **Strategy Pattern** — `ContentDetector` ABC with swappable detection strategies (`CardBoundaryDetector`)
- **Dependency Injection** — `ImageCropper` receives its detector, `BatchProcessor` receives its cropper
- **Value Objects** — `BoundingBox` is immutable with validation
- **Result Objects** — `CropResult` and `BatchResult` for inspectable outcomes

## Tests

```bash
pip3 install pytest
python3 -m pytest test_crop_invoices.py -v
```

## Supported Formats

PNG, JPG, JPEG
