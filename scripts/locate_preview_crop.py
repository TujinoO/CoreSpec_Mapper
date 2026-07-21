from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _orange_mask(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image.astype(np.int16))
    return (red >= 120) & (red - green >= 15) & (green - blue >= 15)


def locate_crop(screenshot_path: Path, preview_dir: Path) -> list[tuple[object, ...]]:
    screenshot = cv2.imread(str(screenshot_path))
    if screenshot is None:
        raise FileNotFoundError(screenshot_path)
    screenshot_edges = cv2.Canny(
        cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY),
        30,
        100,
    )
    paths = sorted(preview_dir.glob("group_final_*_*.png"))
    background = preview_dir / "swir_background_1600nm.png"
    if background.exists():
        paths.append(background)
    results: list[tuple[object, ...]] = []
    for path in paths:
        source = cv2.imread(str(path))
        if source is None:
            continue
        source_edges = cv2.Canny(
            cv2.cvtColor(source, cv2.COLOR_BGR2GRAY),
            30,
            100,
        )
        best: tuple[float, float, tuple[int, int], tuple[int, int]] | None = None
        maximum_scale = min(3.0, source.shape[1] / screenshot.shape[1])
        for scale in np.arange(0.50, maximum_scale + 0.001, 0.025):
            width = int(round(screenshot.shape[1] * scale))
            height = int(round(screenshot.shape[0] * scale))
            if width > source.shape[1] or height > source.shape[0]:
                continue
            template = cv2.resize(
                screenshot_edges,
                (width, height),
                interpolation=(cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC),
            )
            _, score, _, location = cv2.minMaxLoc(
                cv2.matchTemplate(source_edges, template, cv2.TM_CCOEFF_NORMED)
            )
            if best is None or score > best[0]:
                best = (float(score), float(scale), location, (width, height))
        if best is not None:
            x, y = best[2]
            width, height = best[3]
            crop = source[y : y + height, x : x + width]
            resized_crop = cv2.resize(
                crop,
                (screenshot.shape[1], screenshot.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            screenshot_orange = _orange_mask(screenshot)
            crop_orange = _orange_mask(resized_crop)
            orange_union = int(np.count_nonzero(screenshot_orange | crop_orange))
            orange_iou = (
                float(np.count_nonzero(screenshot_orange & crop_orange) / orange_union)
                if orange_union
                else 1.0
            )
            color_mae = float(
                np.mean(
                    np.abs(
                        screenshot.astype(np.float32)
                        - resized_crop.astype(np.float32)
                    )
                )
            )
            results.append(
                (
                    best[0],
                    orange_iou,
                    -color_mae,
                    path.name,
                    best[1],
                    best[2],
                    best[3],
                )
            )
    return sorted(results, key=lambda item: (item[1], item[0], item[2]), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locate a cropped screenshot inside generated CoreSpec previews"
    )
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("preview_dir", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    for result in locate_crop(args.screenshot, args.preview_dir)[: args.limit]:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
