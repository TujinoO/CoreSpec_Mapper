from __future__ import annotations

import base64
from io import BytesIO
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
STRICT_SWIR = ROOT / "output/ZK5511_mineral_mapping_20260816/ZK5511_20260816/ZK5511_SWIR_V5_20260816_QC2"
RELAXED_SWIR = ROOT / "output/ZK5511_mineral_mapping_20260816/ZK5511_20260816_RELAXED_GEO/ZK5511_SWIR_V5_20260816_RELAXED_GEO"
RELAXED_REE = ROOT / "output/ZK5511_mineral_mapping_20260816/vnir_ree_empirical_v7_relaxed_geo"
OUTPUT = ROOT / "output/ZK5511_mineral_mapping_20260816/preview_relaxed_geo_20260816"
DEPTH = (0.0, 196.3)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("msyhbd.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simhei.ttf")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def counts(path: Path, policy: str) -> dict[str, int]:
    with (path / "tables/mineral_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
        return {
            row["mineral"]: int(row["pixels"])
            for row in csv.DictReader(stream)
            if row["policy"] == policy
        }


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, width: int) -> None:
    draw.text((48, 28), title, font=font(42, True), fill=(245, 247, 252))
    draw.text((50, 88), subtitle, font=font(22), fill=(182, 194, 214))
    draw.line((48, 132, width - 48, 132), fill=(68, 83, 110), width=2)


def render_swir() -> Path:
    Image.MAX_IMAGE_PIXELS = None
    source = Image.open(RELAXED_SWIR / "previews/comparison_sensitive.png").convert("RGB")
    strict = counts(STRICT_SWIR, "sensitive")
    relaxed = counts(RELAXED_SWIR, "sensitive")
    minerals = [
        ("Calcite", "方解石", (255, 66, 62)),
        ("Dolomite", "白云石", (168, 86, 238)),
        ("Anhydrite", "硬石膏", (255, 169, 45)),
        ("Gypsum", "石膏", (55, 135, 255)),
        ("Illite", "伊利石", (48, 190, 92)),
        ("Montmorillonite", "蒙脱石", (255, 126, 32)),
        ("Kaolinite", "高岭石", (255, 226, 65)),
    ]
    width, height = 2100, 3480
    canvas = Image.new("RGB", (width, height), (17, 22, 32))
    draw = ImageDraw.Draw(canvas)
    draw_header(
        draw,
        "ZK5511  SWIR 常规蚀变矿物宽松版预览",
        "展示 sensitive 复核层｜0.0–196.3 m｜阈值放宽，固定列、边缘和矿物竞争约束保持启用",
        width,
    )

    box_y, box_h = 160, 250
    draw.rounded_rectangle((48, box_y, width - 48, box_y + box_h), radius=18, fill=(27, 35, 50), outline=(70, 87, 116), width=2)
    draw.text((75, box_y + 18), "严格 sensitive → 宽松 sensitive（像元数）", font=font(24, True), fill=(245, 247, 252))
    column_w = (width - 140) / len(minerals)
    for index, (key, label, color) in enumerate(minerals):
        x = 70 + index * column_w
        draw.ellipse((x, box_y + 63, x + 16, box_y + 79), fill=color)
        draw.text((x + 23, box_y + 56), label, font=font(19), fill=(225, 230, 240))
        before, after = strict.get(key, 0), relaxed.get(key, 0)
        draw.text((x, box_y + 95), f"{before:,}", font=font(18), fill=(155, 165, 184))
        draw.text((x, box_y + 128), "↓", font=font(18), fill=(112, 128, 155))
        draw.text((x, box_y + 157), f"{after:,}", font=font(23, True), fill=color)
        ratio = after / max(before, 1)
        draw.text((x, box_y + 200), f"×{ratio:.1f}", font=font(17), fill=(184, 195, 215))

    labels = ("1600 nm", "碳酸盐", "钙硫酸盐", "伊利石", "蒙脱石", "高岭石")
    segments, columns = 12, 3
    tile_w, image_h, header_h = 638, 642, 64
    gap_x, gap_y = 22, 18
    start_y = 440
    for index in range(segments):
        row, column = divmod(index, columns)
        x = 48 + column * (tile_w + gap_x)
        y = start_y + row * (header_h + image_h + gap_y)
        line0 = round(index / segments * source.height)
        line1 = round((index + 1) / segments * source.height)
        crop = source.crop((0, line0, source.width, line1)).resize((tile_w, image_h), Image.Resampling.LANCZOS)
        start_depth = DEPTH[0] + index / segments * (DEPTH[1] - DEPTH[0])
        stop_depth = DEPTH[0] + (index + 1) / segments * (DEPTH[1] - DEPTH[0])
        draw.text((x, y), f"{start_depth:05.1f}–{stop_depth:05.1f} m", font=font(21, True), fill=(243, 246, 251))
        for track, label in enumerate(labels):
            center = x + (track + 0.5) * tile_w / 6
            bbox = draw.textbbox((0, 0), label, font=font(14))
            draw.text((center - (bbox[2] - bbox[0]) / 2, y + 35), label, font=font(14), fill=(155, 167, 188))
        canvas.paste(crop, (x, y + header_h))
        draw.rectangle((x, y + header_h, x + tile_w - 1, y + header_h + image_h - 1), outline=(73, 85, 105), width=1)

    footer_y = height - 120
    draw.text((50, footer_y), "质量门：Grade B / publishable；重复窄列 0，探测器条带清理保持有效。", font=font(22, True), fill=(201, 225, 207))
    draw.text((50, footer_y + 42), "说明：sensitive 用于扩大召回和人工复核；矿物学确认仍需 XRD、拉曼或点位光谱验证。", font=font(20), fill=(183, 193, 210))
    path = OUTPUT / "ZK5511_SWIR_relaxed_geo_preview.png"
    canvas.save(path, quality=95)
    return path


def render_ree() -> Path:
    Image.MAX_IMAGE_PIXELS = None
    audit = json.loads((RELAXED_REE / "ZK5511_VNIR_REE_RELAXED_GEO_audit.json").read_text(encoding="utf-8"))
    contact = Image.open(RELAXED_REE / "ZK5511_VNIR_REE_RELAXED_GEO_policy_contact.png").convert("RGB")
    candidate = Image.open(RELAXED_REE / "ZK5511_VNIR_REE_RELAXED_GEO_balanced_candidate_QA.png").convert("RGB")
    reference = Image.open(RELAXED_REE / "ZK5511_VNIR_REE_RELAXED_GEO_reference_QA.png").convert("RGB")
    width, height = 2100, 3530
    canvas = Image.new("RGB", (width, height), (246, 248, 252))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 145), fill=(22, 29, 42))
    draw_header(
        draw,
        "ZK5511  VNIR 稀土经验谱相似性异常——宽松地质约束版",
        "两条用户经验谱｜675–925 nm 特征匹配｜仅表示候选异常，不代表已确认矿物相、含量或品位",
        width,
    )
    policies = audit["policies"]
    summary_y = 165
    cards = [
        ("conservative", "保守", (202, 55, 50)),
        ("balanced", "平衡", (241, 135, 35)),
        ("sensitive", "敏感", (219, 180, 28)),
    ]
    for index, (key, label, color) in enumerate(cards):
        x0 = 48 + index * 670
        x1 = x0 + 640
        draw.rounded_rectangle((x0, summary_y, x1, summary_y + 170), radius=18, fill=(255, 255, 255), outline=(204, 211, 222), width=2)
        record = policies[key]
        draw.text((x0 + 24, summary_y + 18), f"{label}档", font=font(24, True), fill=color)
        draw.text((x0 + 24, summary_y + 60), f"{record['family_pixels']:,} 像元", font=font(31, True), fill=(35, 42, 55))
        draw.text((x0 + 24, summary_y + 108), f"{record['kept_component_count']} 个斑块｜前景 {record['fraction_of_foreground'] * 100:.4f}%", font=font(18), fill=(91, 101, 119))
        draw.text((x0 + 24, summary_y + 137), f"最大单列 {record['maximum_column_fraction'] * 100:.3f}%", font=font(17), fill=(112, 121, 137))

    draw.text((50, 370), "全孔分段分布（VNIR 850 nm / 保守 / 平衡 / 敏感 / 两参考谱竞争）", font=font(25, True), fill=(35, 42, 55))
    segments, columns = 6, 2
    tile_w, image_h, header_h = 984, 420, 58
    for index in range(segments):
        row, column = divmod(index, columns)
        x = 48 + column * (tile_w + 34)
        y = 415 + row * (header_h + image_h + 18)
        line0 = round(index / segments * contact.height)
        line1 = round((index + 1) / segments * contact.height)
        crop = contact.crop((0, line0, contact.width, line1)).resize((tile_w, image_h), Image.Resampling.LANCZOS)
        start_depth = index / segments * DEPTH[1]
        stop_depth = (index + 1) / segments * DEPTH[1]
        draw.text((x, y), f"{start_depth:05.1f}–{stop_depth:05.1f} m", font=font(20, True), fill=(47, 54, 67))
        labels = ("850 nm", "保守", "平衡", "敏感", "参考谱竞争")
        for track, label in enumerate(labels):
            center = x + (track + 0.5) * tile_w / 5
            bbox = draw.textbbox((0, 0), label, font=font(15))
            draw.text((center - (bbox[2] - bbox[0]) / 2, y + 31), label, font=font(15), fill=(105, 114, 130))
        canvas.paste(crop, (x, y + header_h))
        draw.rectangle((x, y + header_h, x + tile_w - 1, y + header_h + image_h - 1), outline=(178, 186, 199), width=1)

    lower_y = 1940
    draw.text((50, lower_y), "balanced 最大候选斑块局部复核（左：原始；右：候选叠加）", font=font(25, True), fill=(35, 42, 55))
    candidate.thumbnail((1350, 1400), Image.Resampling.LANCZOS)
    canvas.paste(candidate, (48, lower_y + 48))
    draw.rectangle((47, lower_y + 47, 49 + candidate.width, lower_y + 49 + candidate.height), outline=(178, 186, 199), width=2)
    reference.thumbnail((650, 440), Image.Resampling.LANCZOS)
    canvas.paste(reference, (1420, lower_y + 48))
    draw.text((1420, lower_y + 500), "空间与深度检查", font=font(23, True), fill=(35, 42, 55))
    largest = audit["balanced_components"]["largest"][:6]
    for index, record in enumerate(largest):
        draw.text(
            (1420, lower_y + 545 + index * 48),
            f"#{index + 1}  {record['pixels']:>3} px   {record['centroid_depth_m']:.2f} m",
            font=font(19),
            fill=(66, 75, 91),
        )
    draw.text((1420, lower_y + 855), "质量门：ready / 无警告", font=font(22, True), fill=(35, 139, 84))
    draw.text((1420, lower_y + 900), "balanced：1,380 像元", font=font(20), fill=(72, 81, 97))
    draw.text((1420, lower_y + 937), "严格版 balanced：147 像元", font=font(20), fill=(72, 81, 97))
    draw.text((1420, lower_y + 974), "召回规模约 ×9.4", font=font(22, True), fill=(227, 115, 35))
    draw.text((50, height - 105), "解释边界：图中红/橙/黄区域是与两条 VNIR 经验谱相似的空间连续异常，需结合地质编录与实验分析复核。", font=font(21), fill=(78, 87, 103))
    path = OUTPUT / "ZK5511_VNIR_REE_relaxed_geo_preview.png"
    canvas.save(path, quality=95)
    return path


def data_uri(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((1150, 1900), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_html(path: Path, title: str, image_path: Path) -> None:
    uri = data_uri(image_path)
    path.write_text(
        f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>body{{margin:0;background:#111827;color:#f8fafc;font-family:"Microsoft YaHei",sans-serif}}main{{max-width:1220px;margin:auto;padding:18px}}h1{{font-size:24px;margin:0 0 14px}}img{{display:block;width:100%;height:auto;background:white;border-radius:10px;box-shadow:0 8px 28px #0008}}</style></head><body><main><h1>{title}</h1><img src="{uri}" alt="{title}"></main></body></html>''',
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    swir = render_swir()
    ree = render_ree()
    print(swir)
    print(ree)


if __name__ == "__main__":
    main()
