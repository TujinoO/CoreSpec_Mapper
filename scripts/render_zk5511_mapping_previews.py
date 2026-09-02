from __future__ import annotations

import argparse
import base64
import csv
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MINERALS = (
    ("方解石", "carbonate_2300", 1, "#d1495b"),
    ("白云石", "carbonate_2300", 2, "#7b2cbf"),
    ("硬石膏", "calcium_sulfates", 1, "#0077b6"),
    ("石膏", "calcium_sulfates", 2, "#48cae4"),
    ("伊利石", "white_mica_illite", 1, "#f4a261"),
    ("蒙脱石", "smectites", 1, "#2a9d8f"),
    ("高岭石", "kaolin_2170_2205", 1, "#e9c46a"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render compact ZK5511 mapping previews")
    parser.add_argument("--swir-run", type=Path, required=True)
    parser.add_argument("--ree-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--depth-start", type=float, default=0.0)
    parser.add_argument("--depth-stop", type=float, default=196.3)
    return parser.parse_args()


def chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def configure_matplotlib_font() -> None:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def render_swir_montage(source: Path, output: Path, depth: tuple[float, float]) -> None:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(source).convert("RGB")
    segments, columns = 12, 3
    rows = (segments + columns - 1) // columns
    tile_width, tile_image_height = 570, 740
    tile_header = 66
    gap, margin = 20, 28
    title_height, footer_height = 118, 70
    canvas_width = margin * 2 + columns * tile_width + (columns - 1) * gap
    canvas_height = title_height + rows * (tile_header + tile_image_height + gap) + footer_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (16, 19, 26))
    draw = ImageDraw.Draw(canvas)
    title_font = chinese_font(34)
    label_font = chinese_font(20)
    small_font = chinese_font(16)
    draw.text((margin, 22), "ZK5511 SWIR balanced 整孔矿物填图", font=title_font, fill=(245, 247, 250))
    draw.text(
        (margin, 70),
        "每格为一个连续深度段；从左到右：1600 nm、碳酸盐、钙硫酸盐、伊利石、蒙脱石、高岭石",
        font=label_font,
        fill=(190, 198, 212),
    )
    track_labels = ("1600", "碳酸盐", "硫酸盐", "伊利石", "蒙脱石", "高岭石")
    for index in range(segments):
        row, column = divmod(index, columns)
        x = margin + column * (tile_width + gap)
        y = title_height + row * (tile_header + tile_image_height + gap)
        line0 = round(index / segments * image.height)
        line1 = round((index + 1) / segments * image.height)
        crop = image.crop((0, line0, image.width, line1)).resize(
            (tile_width, tile_image_height), Image.Resampling.LANCZOS
        )
        start = depth[0] + index / segments * (depth[1] - depth[0])
        stop = depth[0] + (index + 1) / segments * (depth[1] - depth[0])
        draw.text((x, y), f"{start:05.1f}–{stop:05.1f} m", font=label_font, fill=(245, 247, 250))
        for track, text in enumerate(track_labels):
            center = x + (track + 0.5) * tile_width / 6
            bbox = draw.textbbox((0, 0), text, font=small_font)
            draw.text((center - (bbox[2] - bbox[0]) / 2, y + 32), text, font=small_font, fill=(160, 170, 188))
        canvas.paste(crop, (x, y + tile_header))
        draw.rectangle((x, y + tile_header, x + tile_width - 1, y + tile_header + tile_image_height - 1), outline=(80, 88, 104), width=1)
    draw.text(
        (margin, canvas_height - 48),
        "QC2：质量等级 B，自动发布门通过；高岭石 balanced 未可靠检出。",
        font=label_font,
        fill=(205, 211, 221),
    )
    canvas.save(output, quality=95)


def read_counts(path: Path) -> dict[tuple[str, str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            (row["policy"], row["mineral"]): int(row["pixels"])
            for row in csv.DictReader(stream)
        }


def binned_class_counts(path: Path, class_id: int, lines: int, samples: int, bins: int) -> np.ndarray:
    array = np.memmap(path, dtype=np.uint8, mode="r", shape=(lines, samples))
    row_counts = np.count_nonzero(array == class_id, axis=1)
    edges = np.linspace(0, lines, bins + 1).round().astype(int)
    result = np.asarray([np.sum(row_counts[edges[i] : edges[i + 1]]) for i in range(bins)], dtype=float)
    del array
    return result


def binned_binary_counts(path: Path, lines: int, samples: int, bins: int) -> np.ndarray:
    array = np.memmap(path, dtype=np.uint8, mode="r", shape=(lines, samples))
    row_counts = np.count_nonzero(array > 0, axis=1)
    edges = np.linspace(0, lines, bins + 1).round().astype(int)
    result = np.asarray([np.sum(row_counts[edges[i] : edges[i + 1]]) for i in range(bins)], dtype=float)
    del array
    return result


def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.asarray([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def render_depth_summary(swir_run: Path, ree_run: Path, output: Path, depth: tuple[float, float]) -> None:
    configure_matplotlib_font()
    counts = read_counts(swir_run / "tables" / "mineral_counts.csv")
    labels = [item[0] for item in MINERALS] + ["稀土经验异常"]
    colors = [item[3] for item in MINERALS] + ["#e76f51"]
    balanced = [counts[("balanced", name)] for name in ("Calcite", "Dolomite", "Anhydrite", "Gypsum", "Illite", "Montmorillonite", "Kaolinite")] + [147]
    bins = 99
    profiles = []
    for _, group, class_id, _ in MINERALS:
        profiles.append(
            binned_class_counts(
                swir_run / "groups" / group / "final_balanced.dat",
                class_id,
                29869,
                320,
                bins,
            )
        )
    profiles.append(
        binned_binary_counts(
            ree_run / "ZK5511_VNIR_REE_family_balanced.dat", 30118, 320, bins
        )
    )
    normalized = []
    for values in profiles:
        maximum = float(np.max(values))
        normalized.append(np.log1p(values) / np.log1p(maximum) if maximum > 0 else values)
    rgba = np.ones((len(labels), bins, 4), dtype=float)
    for row, (values, color) in enumerate(zip(normalized, colors)):
        rgb = hex_rgb(color)
        rgba[row, :, :3] = 1.0 - values[:, None] * (1.0 - rgb[None, :])
        rgba[row, :, 3] = 1.0

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(1.05, 1.0))
    axis_bar = figure.add_subplot(grid[0])
    y = np.arange(len(labels))
    values = np.log10(np.asarray(balanced, dtype=float) + 1.0)
    axis_bar.barh(y, values, color=colors, height=0.64)
    axis_bar.set_yticks(y, labels)
    axis_bar.invert_yaxis()
    axis_bar.set_xlabel("log10（balanced 像元数 + 1）")
    axis_bar.set_title("ZK5511 当前正式填图：balanced 结果规模")
    axis_bar.grid(axis="x", alpha=0.22)
    for index, (value, count) in enumerate(zip(values, balanced)):
        axis_bar.text(value + 0.035, index, f"{count:,}", va="center", fontsize=10)
    axis_bar.set_xlim(0, max(values) + 0.55)

    axis_heat = figure.add_subplot(grid[1])
    axis_heat.imshow(rgba, aspect="auto", interpolation="nearest", extent=(depth[0], depth[1], len(labels) - 0.5, -0.5))
    axis_heat.set_yticks(np.arange(len(labels)), labels)
    axis_heat.set_xlabel("深度（m）")
    axis_heat.set_title("沿深度分布（每个矿物按自身最大值对数归一化；颜色越深代表该深度段越集中）")
    axis_heat.set_xticks(list(np.arange(0, 181, 20)) + [196.3])
    axis_heat.grid(axis="x", color="white", alpha=0.55, linewidth=0.7)
    figure.text(
        0.5,
        0.01,
        "稀土行表示与两条用户经验谱相似的 VNIR 候选异常，不代表已确认矿物相或品位。",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def render_ree_preview(ree_run: Path, output: Path) -> None:
    Image.MAX_IMAGE_PIXELS = None
    candidate = Image.open(ree_run / "ZK5511_VNIR_REE_balanced_candidate_QA.png").convert("RGB")
    reference = Image.open(ree_run / "ZK5511_VNIR_REE_reference_QA.png").convert("RGB")
    policy = Image.open(ree_run / "ZK5511_VNIR_REE_policy_contact.png").convert("RGB")
    canvas = Image.new("RGB", (1840, 1440), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = chinese_font(34)
    label_font = chinese_font(21)
    small_font = chinese_font(18)
    draw.text((34, 22), "ZK5511 VNIR 稀土经验谱异常预览", font=title_font, fill=(28, 32, 40))
    draw.text(
        (34, 72),
        "balanced：147 像元 / 14 个斑块；以下为光谱相似性候选，不是矿物相或品位确认。",
        font=label_font,
        fill=(78, 85, 98),
    )
    candidate.thumbnail((1120, 1240), Image.Resampling.LANCZOS)
    canvas.paste(candidate, (30, 130))
    draw.rectangle((29, 129, 31 + candidate.width, 131 + candidate.height), outline=(175, 180, 190), width=2)
    reference.thumbnail((650, 420), Image.Resampling.LANCZOS)
    canvas.paste(reference, (1170, 145))
    draw.text((1170, 115), "两条用户经验参考谱及窄吸收特征", font=small_font, fill=(45, 50, 60))
    # Policy overview is extremely tall; compress it into a compact full-depth strip.
    policy = policy.resize((650, 760), Image.Resampling.LANCZOS)
    canvas.paste(policy, (1170, 625))
    draw.text((1170, 590), "全深度三档策略与参考谱竞争", font=small_font, fill=(45, 50, 60))
    canvas.save(output, quality=95)


def jpeg_data_uri(path: Path, maximum: tuple[int, int]) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail(maximum, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=58, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_html(path: Path, images: list[tuple[str, Path]]) -> None:
    encoded = [(label, jpeg_data_uri(image, (920, 1200))) for label, image in images]
    buttons = "\n".join(
        f'<button type="button" class="btn" aria-pressed="{str(index == 0).lower()}" data-index="{index}">{label}</button>'
        for index, (label, _) in enumerate(encoded)
    )
    data = ",\n".join(
        "{" + f'label:{label!r},src:{uri!r}' + "}" for label, uri in encoded
    )
    fragment = f'''<div id="zk5511-preview">
  <h2>ZK5511 当前填图结果预览</h2>
  <div class="viz-controls" role="group" aria-label="选择预览图">
{buttons}
  </div>
  <figure>
    <img id="zk5511-preview-image" src="{encoded[0][1]}" alt="{encoded[0][0]}" />
    <figcaption id="zk5511-preview-caption" class="text-muted">{encoded[0][0]}</figcaption>
  </figure>
</div>
<style>
#zk5511-preview {{ color: var(--foreground); }}
#zk5511-preview figure {{ margin: 12px 0 0; }}
#zk5511-preview img {{ display: block; max-width: 100%; height: auto; border: 1px solid var(--border); }}
#zk5511-preview figcaption {{ margin-top: 8px; }}
</style>
<script>
(() => {{
  const root = document.getElementById('zk5511-preview');
  const items = [{data}];
  const image = root.querySelector('#zk5511-preview-image');
  const caption = root.querySelector('#zk5511-preview-caption');
  root.querySelectorAll('button[data-index]').forEach((button) => {{
    button.addEventListener('click', () => {{
      const index = Number(button.dataset.index);
      image.src = items[index].src;
      image.alt = items[index].label;
      caption.textContent = items[index].label;
      root.querySelectorAll('button[data-index]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
    }});
  }});
}})();
</script>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def write_single_image_html(path: Path, label: str, image_path: Path) -> None:
    uri = jpeg_data_uri(image_path, (980, 1400))
    fragment = f'''<div id="zk5511-ree-preview">
  <h2>VNIR 稀土经验谱异常</h2>
  <figure>
    <img src="{uri}" alt="{label}" />
    <figcaption class="text-muted">{label}</figcaption>
  </figure>
</div>
<style>
#zk5511-ree-preview {{ color: var(--foreground); }}
#zk5511-ree-preview figure {{ margin: 12px 0 0; }}
#zk5511-ree-preview img {{ display: block; max-width: 100%; height: auto; border: 1px solid var(--border); }}
#zk5511-ree-preview figcaption {{ margin-top: 8px; }}
</style>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth = (args.depth_start, args.depth_stop)
    swir = args.output_dir / "ZK5511_SWIR_balanced_montage.png"
    depth_summary = args.output_dir / "ZK5511_integrated_depth_summary.png"
    ree = args.output_dir / "ZK5511_VNIR_REE_candidate_preview.png"
    for path in (swir, depth_summary, ree, args.html):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing preview: {path}")
    render_swir_montage(args.swir_run / "previews" / "comparison_balanced.png", swir, depth)
    render_depth_summary(args.swir_run, args.ree_run, depth_summary, depth)
    render_ree_preview(args.ree_run, ree)
    write_html(
        args.html,
        [
            ("SWIR balanced 整孔分段图", swir),
            ("矿物与稀土沿深度分布", depth_summary),
            ("VNIR 稀土候选与参考谱", ree),
        ],
    )
    print(swir)
    print(depth_summary)
    print(ree)
    print(args.html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
