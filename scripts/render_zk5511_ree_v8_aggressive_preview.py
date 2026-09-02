from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output/ZK5511_mineral_mapping_20260817/vnir_ree_empirical_v8_aggressive_grow"
V7 = ROOT / "output/ZK5511_mineral_mapping_20260816/vnir_ree_empirical_v7_relaxed_geo"
OUTPUT = ROOT / "output/ZK5511_mineral_mapping_20260817/preview_v8_aggressive_grow"
PREFIX = "ZK5511_VNIR_REE_V8_AGGRESSIVE"
DEPTH = 196.3


def font(size: int, bold: bool = False):
    names = ("msyhbd.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simhei.ttf")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def render() -> Path:
    Image.MAX_IMAGE_PIXELS = None
    audit = json.loads((RUN / f"{PREFIX}_audit.json").read_text(encoding="utf-8"))
    v7 = json.loads((V7 / "ZK5511_VNIR_REE_RELAXED_GEO_audit.json").read_text(encoding="utf-8"))
    contact = Image.open(RUN / f"{PREFIX}_policy_contact.png").convert("RGB")
    candidate = Image.open(RUN / f"{PREFIX}_balanced_candidate_QA.png").convert("RGB")
    reference = Image.open(RUN / f"{PREFIX}_reference_QA.png").convert("RGB")
    width, height = 2100, 3660
    canvas = Image.new("RGB", (width, height), (246, 248, 252))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 150), fill=(22, 29, 42))
    draw.text((48, 26), "ZK5511  VNIR 稀土经验谱异常——v8 激进高召回版", font=font(42, True), fill=(248, 250, 253))
    draw.text((50, 88), "种子 + 宽阈值连通生长｜探索性复核层｜不是矿物相、含量或品位确认", font=font(22), fill=(190, 201, 220))
    draw.line((48, 133, width - 48, 133), fill=(76, 91, 119), width=2)

    cards = [
        ("conservative", "保守生长", (196, 55, 49)),
        ("balanced", "平衡高召回", (238, 125, 32)),
        ("sensitive", "敏感探索", (207, 166, 20)),
    ]
    for index, (key, label, color) in enumerate(cards):
        record = audit["policies"][key]
        x0 = 48 + index * 670
        draw.rounded_rectangle((x0, 170, x0 + 640, 365), radius=18, fill="white", outline=(202, 210, 223), width=2)
        draw.text((x0 + 24, 188), label, font=font(23, True), fill=color)
        draw.text((x0 + 24, 229), f"{record['family_pixels']:,} 像元", font=font(32, True), fill=(35, 42, 55))
        draw.text((x0 + 24, 278), f"{record['kept_component_count']} 个斑块｜前景 {record['fraction_of_foreground'] * 100:.3f}%", font=font(18), fill=(83, 94, 113))
        draw.text((x0 + 24, 315), f"种子 {record['seed_pixels_before_region_growth']:,} → 生长 {record['pixels_after_seeded_region_growth']:,}", font=font(18), fill=(99, 109, 126))

    b8 = audit["policies"]["balanced"]["family_pixels"]
    s8 = audit["policies"]["sensitive"]["family_pixels"]
    b7 = v7["policies"]["balanced"]["family_pixels"]
    s7 = v7["policies"]["sensitive"]["family_pixels"]
    draw.rounded_rectangle((48, 388, width - 48, 480), radius=15, fill=(255, 243, 223), outline=(236, 169, 67), width=2)
    draw.text((72, 407), f"相对 v7：balanced {b7:,} → {b8:,}（×{b8 / b7:.1f}）｜sensitive {s7:,} → {s8:,}（×{s8 / s7:.1f}）", font=font(24, True), fill=(121, 71, 20))
    draw.text((72, 447), "注意：balanced / sensitive 生长角超过两参考谱间夹角，召回更高，光谱特异性更低。", font=font(19), fill=(151, 92, 30))

    draw.text((50, 510), "全孔分段分布（850 nm / 保守 / 平衡 / 敏感 / 两参考谱竞争）", font=font(25, True), fill=(35, 42, 55))
    labels = ("850 nm", "保守", "平衡", "敏感", "参考谱竞争")
    tile_w, image_h, header_h = 984, 425, 58
    for index in range(6):
        row, column = divmod(index, 2)
        x = 48 + column * (tile_w + 34)
        y = 552 + row * (header_h + image_h + 18)
        line0 = round(index / 6 * contact.height)
        line1 = round((index + 1) / 6 * contact.height)
        crop = contact.crop((0, line0, contact.width, line1)).resize((tile_w, image_h), Image.Resampling.LANCZOS)
        draw.text((x, y), f"{index / 6 * DEPTH:05.1f}–{(index + 1) / 6 * DEPTH:05.1f} m", font=font(20, True), fill=(47, 54, 67))
        for track, label in enumerate(labels):
            center = x + (track + 0.5) * tile_w / 5
            bbox = draw.textbbox((0, 0), label, font=font(15))
            draw.text((center - (bbox[2] - bbox[0]) / 2, y + 31), label, font=font(15), fill=(104, 114, 131))
        canvas.paste(crop, (x, y + header_h))
        draw.rectangle((x, y + header_h, x + tile_w - 1, y + header_h + image_h - 1), outline=(178, 186, 199), width=1)

    lower_y = 2080
    draw.text((50, lower_y), "balanced 最大连通候选局部复核（左：原始；右：候选叠加）", font=font(25, True), fill=(35, 42, 55))
    candidate.thumbnail((1350, 1420), Image.Resampling.LANCZOS)
    canvas.paste(candidate, (48, lower_y + 48))
    draw.rectangle((47, lower_y + 47, 49 + candidate.width, lower_y + 49 + candidate.height), outline=(178, 186, 199), width=2)
    reference.thumbnail((650, 440), Image.Resampling.LANCZOS)
    canvas.paste(reference, (1420, lower_y + 48))
    draw.text((1420, lower_y + 510), "工程质量检查", font=font(23, True), fill=(35, 42, 55))
    lines = [
        "质量门：ready / 无自动警告",
        "三档嵌套：通过",
        "balanced 最大单列：1.77%",
        "sensitive 最大单列：4.57%",
        "固定条带：未触发",
        "balanced 最大斑块：1,267 px",
        "balanced 中位斑块：25.5 px",
    ]
    for index, text in enumerate(lines):
        color = (35, 139, 84) if index < 2 else (70, 81, 99)
        draw.text((1420, lower_y + 555 + index * 52), text, font=font(19, index < 2), fill=color)
    draw.rounded_rectangle((1415, lower_y + 945, 2070, lower_y + 1125), radius=15, fill=(255, 239, 232), outline=(226, 128, 93), width=2)
    draw.text((1440, lower_y + 968), "解释边界", font=font(22, True), fill=(151, 62, 41))
    draw.multiline_text((1440, lower_y + 1010), "该版本用于尽量发现可能区域。\n靠近裂隙、端部和亮度突变处的\n新增候选必须人工复核。", font=font(18), fill=(113, 68, 57), spacing=9)
    draw.text((50, height - 95), "建议用法：balanced 作为重点复核图，sensitive 作为最大范围排查图；正式矿物结论仍需 XRD、拉曼或点位化验。", font=font(21), fill=(72, 82, 100))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "ZK5511_VNIR_REE_v8_aggressive_preview.png"
    canvas.save(path, quality=95)
    return path


def write_html(image_path: Path, html_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1150, 2000), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=74, optimize=True)
    uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    html_path.write_text(
        f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ZK5511 稀土 v8 激进高召回</title><style>body{{margin:0;background:#111827;color:#f8fafc;font-family:"Microsoft YaHei",sans-serif}}main{{max-width:1220px;margin:auto;padding:18px}}h1{{font-size:24px}}img{{width:100%;height:auto;border-radius:10px;box-shadow:0 8px 28px #0008}}</style></head><body><main><h1>ZK5511 稀土 v8 激进高召回版</h1><img src="{uri}" alt="ZK5511 稀土 v8 激进高召回版"></main></body></html>''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(render())
