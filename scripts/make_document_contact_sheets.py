from __future__ import annotations

from pathlib import Path
import argparse
import math
import re

from PIL import Image, ImageDraw, ImageFont


def page_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def build(source: Path, output: Path, pages_per_sheet: int = 8) -> None:
    pages = sorted(source.glob("*.png"), key=page_number)
    output.mkdir(parents=True, exist_ok=True)
    columns = 2
    rows = math.ceil(pages_per_sheet / columns)
    thumb_width = 520
    label_height = 28
    for sheet_index in range(0, len(pages), pages_per_sheet):
        current = pages[sheet_index:sheet_index + pages_per_sheet]
        first = Image.open(current[0]).convert("RGB")
        thumb_height = round(first.height * thumb_width / first.width)
        canvas = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#d9e0e6")
        draw = ImageDraw.Draw(canvas)
        for index, path in enumerate(current):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            column = index % columns
            row = index // columns
            x = column * thumb_width + (thumb_width - image.width) // 2
            y = row * (thumb_height + label_height) + label_height
            canvas.paste(image, (x, y))
            draw.text((column * thumb_width + 8, row * (thumb_height + label_height) + 5), f"Page {page_number(path)}", fill="#21394d")
        destination = output / f"sheet-{sheet_index // pages_per_sheet + 1:02d}.png"
        canvas.save(destination, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--pages-per-sheet", type=int, default=8)
    args = parser.parse_args()
    build(Path(args.source), Path(args.output), args.pages_per_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
