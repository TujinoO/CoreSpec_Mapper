from __future__ import annotations

import base64
import csv
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output/ZK5511_mineral_mapping_20260817/vnir_ree_empirical_v8_aggressive_grow"
OUTPUT = ROOT / "output/ZK5511_mineral_mapping_20260817/preview_v8_aggressive_grow_v2"
VIZ = OUTPUT
PREFIX = "ZK5511_VNIR_REE_V8_AGGRESSIVE"
DEPTH_MAX = 196.3


def font(size: int, bold: bool = False):
    candidates = ("msyhbd.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simhei.ttf")
    for name in candidates:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def audit() -> dict:
    return json.loads((RUN / f"{PREFIX}_audit.json").read_text(encoding="utf-8"))


def header(canvas: Image.Image, title: str, subtitle: str) -> ImageDraw.ImageDraw:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 128), fill=(20, 28, 42))
    draw.text((42, 22), title, font=font(36, True), fill=(248, 250, 253))
    draw.text((44, 76), subtitle, font=font(20), fill=(190, 202, 221))
    return draw


def render_overview(data: dict) -> Path:
    canvas = Image.new("RGB", (1920, 1120), (247, 249, 252))
    draw = header(
        canvas,
        "ZK5511 稀土经验谱异常 v8｜激进高召回总览",
        "balanced 用于重点复核；sensitive 用于最大范围排查｜0.0–196.3 m",
    )
    cards = [
        ("conservative", "保守生长", (194, 58, 52)),
        ("balanced", "平衡高召回", (234, 119, 31)),
        ("sensitive", "敏感探索", (199, 160, 25)),
    ]
    for index, (key, label, color) in enumerate(cards):
        record = data["policies"][key]
        x0 = 42 + index * 620
        draw.rounded_rectangle((x0, 152, x0 + 590, 312), radius=16, fill="white", outline=(202, 211, 224), width=2)
        draw.text((x0 + 24, 172), label, font=font(22, True), fill=color)
        draw.text((x0 + 24, 214), f"{record['family_pixels']:,} 像元", font=font(32, True), fill=(35, 43, 57))
        draw.text((x0 + 24, 262), f"{record['kept_component_count']} 个斑块｜前景 {record['fraction_of_foreground'] * 100:.3f}%", font=font(18), fill=(87, 98, 116))

    rows = []
    with (RUN / f"{PREFIX}_depth_blocks.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    centers = np.asarray([(float(r["start_depth_m"]) + float(r["stop_depth_m"])) / 2 for r in rows])
    fractions = np.asarray([float(r["balanced_fraction_of_foreground"]) * 100 for r in rows])
    counts = np.asarray([int(r["balanced_REE_family_pixels"]) for r in rows])

    chart = (70, 390, 1330, 995)
    left, top, right, bottom = chart
    draw.text((left, 342), "balanced 沿深度候选密度", font=font(25, True), fill=(35, 43, 57))
    draw.rectangle(chart, fill="white", outline=(185, 194, 207), width=2)
    pad_l, pad_r, pad_t, pad_b = 90, 32, 35, 62
    px0, px1 = left + pad_l, right - pad_r
    py0, py1 = top + pad_t, bottom - pad_b
    maximum = max(float(np.max(fractions)), 0.001)
    for tick in np.linspace(0, maximum, 5):
        y = py1 - tick / maximum * (py1 - py0)
        draw.line((px0, y, px1, y), fill=(225, 230, 237), width=1)
        draw.text((left + 10, y - 11), f"{tick:.2f}%", font=font(15), fill=(100, 111, 130))
    bar_width = max(4, int((px1 - px0) / len(rows) * 0.68))
    for center, value in zip(centers, fractions):
        x = px0 + center / DEPTH_MAX * (px1 - px0)
        y = py1 - value / maximum * (py1 - py0)
        draw.rectangle((x - bar_width / 2, y, x + bar_width / 2, py1), fill=(239, 132, 50))
    for depth in range(0, 181, 20):
        x = px0 + depth / DEPTH_MAX * (px1 - px0)
        draw.line((x, py1, x, py1 + 7), fill=(82, 92, 108), width=2)
        draw.text((x - 16, py1 + 14), str(depth), font=font(15), fill=(78, 88, 105))
    draw.text((px1 - 10, py1 + 14), "196.3 m", font=font(15), fill=(78, 88, 105), anchor="ra")
    draw.text((left + 10, bottom - 30), "纵轴：每个约3.3 m深度块内候选占岩心前景比例", font=font(15), fill=(96, 107, 125))

    side_x = 1380
    draw.text((side_x, 342), "高密度深度段", font=font(25, True), fill=(35, 43, 57))
    order = np.argsort(counts)[::-1][:10]
    for rank, index in enumerate(order, start=1):
        y = 400 + (rank - 1) * 52
        start = float(rows[index]["start_depth_m"])
        stop = float(rows[index]["stop_depth_m"])
        draw.text((side_x, y), f"{rank:02d}", font=font(18, True), fill=(235, 121, 35))
        draw.text((side_x + 48, y), f"{start:05.1f}–{stop:05.1f} m", font=font(18), fill=(48, 57, 72))
        draw.text((side_x + 278, y), f"{counts[index]:,} px", font=font(18, True), fill=(48, 57, 72))
    draw.rounded_rectangle((1365, 930, 1878, 1020), radius=14, fill=(255, 241, 230), outline=(227, 142, 99), width=2)
    draw.text((1390, 948), "高召回解释边界", font=font(19, True), fill=(150, 65, 43))
    draw.text((1390, 982), "裂隙、岩块端部与亮度突变需重点复核", font=font(16), fill=(112, 73, 63))
    path = OUTPUT / "ZK5511_REE_v8_overview.png"
    canvas.save(path, quality=95)
    return path


def render_depth_atlas() -> Path:
    Image.MAX_IMAGE_PIXELS = None
    contact = Image.open(RUN / f"{PREFIX}_policy_contact.png").convert("RGB")
    canvas = Image.new("RGB", (2400, 2850), (247, 249, 252))
    draw = header(
        canvas,
        "ZK5511 稀土经验谱异常 v8｜全孔分段图",
        "每个深度段按原始宽高比展示：VNIR 850 nm｜balanced｜sensitive",
    )
    legend = [("850 nm 原始影像", (78, 88, 104)), ("balanced 重点复核", (237, 124, 38)), ("sensitive 最大排查", (204, 169, 27))]
    lx = 50
    for text, color in legend:
        draw.ellipse((lx, 151, lx + 16, 167), fill=color)
        draw.text((lx + 24, 143), text, font=font(18), fill=(60, 70, 87))
        lx += 360

    track_x = {"source": (0, 320), "balanced": (640, 960), "sensitive": (960, 1280)}
    tile_w = 1128
    track_w = 350
    gap = 18
    start_y = 205
    for index in range(6):
        row, column = divmod(index, 2)
        tile_x = 50 + column * 1175
        tile_y = start_y + row * 870
        y0 = round(index / 6 * contact.height)
        y1 = round((index + 1) / 6 * contact.height)
        native_h = y1 - y0
        track_h = round(native_h * track_w / 320)
        draw.text((tile_x, tile_y), f"{index / 6 * DEPTH_MAX:05.1f}–{(index + 1) / 6 * DEPTH_MAX:05.1f} m", font=font(22, True), fill=(40, 49, 63))
        for t_index, (key, label) in enumerate((("source", "850 nm"), ("balanced", "balanced"), ("sensitive", "sensitive"))):
            x = tile_x + t_index * (track_w + gap)
            x0, x1 = track_x[key]
            crop = contact.crop((x0, y0, x1, y1)).resize((track_w, track_h), Image.Resampling.LANCZOS)
            draw.text((x, tile_y + 38), label, font=font(17, key != "source"), fill=(84, 95, 113))
            canvas.paste(crop, (x, tile_y + 70))
            draw.rectangle((x, tile_y + 70, x + track_w - 1, tile_y + 70 + track_h - 1), outline=(178, 187, 200), width=1)
    draw.text((50, 2795), "橙/黄候选仅表示与两条用户经验谱相似；岩心影像未做非等比例拉伸。", font=font(18), fill=(85, 96, 114))
    path = OUTPUT / "ZK5511_REE_v8_depth_atlas.png"
    canvas.save(path, quality=95)
    return path


def render_candidate_details() -> Path:
    Image.MAX_IMAGE_PIXELS = None
    source = Image.open(RUN / f"{PREFIX}_balanced_candidate_QA.png").convert("RGB")
    canvas = Image.new("RGB", (2100, 2290), (247, 249, 252))
    draw = header(
        canvas,
        "ZK5511 稀土经验谱异常 v8｜balanced 候选细节",
        "每组左侧为原始 VNIR，右侧为候选叠加；保持原始局部影像比例",
    )
    target_w = 2010
    target_h = round(source.height * target_w / source.width)
    resized = source.resize((target_w, target_h), Image.Resampling.LANCZOS)
    canvas.paste(resized, (45, 158))
    draw.rectangle((44, 157, 46 + target_w, 159 + target_h), outline=(173, 182, 196), width=2)
    draw.text((48, 2228), "优先复核：裂隙边缘、岩块端部、强反射颗粒及托盘边界附近的新增候选。", font=font(19), fill=(125, 71, 55))
    path = OUTPUT / "ZK5511_REE_v8_candidate_details.png"
    canvas.save(path, quality=95)
    return path


def jpg_uri(path: Path, maximum: tuple[int, int], quality: int) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail(maximum, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_fragment(paths: list[Path]) -> Path:
    uris = [jpg_uri(paths[0], (1050, 800), 76), jpg_uri(paths[1], (1000, 1500), 70), jpg_uri(paths[2], (1000, 1150), 72)]
    labels = ["总览", "全孔分段", "候选细节"]
    alts = ["ZK5511 稀土 v8 总览", "ZK5511 稀土 v8 全孔分段图", "ZK5511 稀土 v8 候选细节图"]
    fragment = f'''<div id="zk5511-ree-v8-clear">
  <h2>ZK5511 稀土 v8 清晰预览</h2>
  <div class="viz-controls" role="group" aria-label="选择预览内容">
    <button type="button" class="btn" aria-pressed="true" data-view="0">总览</button>
    <button type="button" class="btn" aria-pressed="false" data-view="1">全孔分段</button>
    <button type="button" class="btn" aria-pressed="false" data-view="2">候选细节</button>
  </div>
  <figure>
    <img data-preview-image src="{uris[0]}" alt="{alts[0]}">
    <figcaption class="text-small text-muted" data-preview-caption>总览：像元规模、深度密度与重点区段</figcaption>
  </figure>
  <style>
    #zk5511-ree-v8-clear figure {{ margin: 12px 0 0; }}
    #zk5511-ree-v8-clear [data-preview-image] {{ display: block; width: 100%; height: auto; object-fit: contain; border: 1px solid var(--border); }}
    #zk5511-ree-v8-clear figcaption {{ margin-top: 8px; }}
  </style>
  <script>
    (() => {{
      const root = document.getElementById('zk5511-ree-v8-clear');
      const images = {json.dumps(uris)};
      const alts = {json.dumps(alts, ensure_ascii=False)};
      const captions = ['总览：像元规模、深度密度与重点区段', '全孔分段：850 nm、balanced 与 sensitive 等比例对照', '候选细节：原始影像与 balanced 叠加对照'];
      const image = root.querySelector('[data-preview-image]');
      const caption = root.querySelector('[data-preview-caption]');
      root.querySelectorAll('button[data-view]').forEach((button) => {{
        button.addEventListener('click', () => {{
          const index = Number(button.dataset.view);
          root.querySelectorAll('button[data-view]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
          image.src = images[index];
          image.alt = alts[index];
          caption.textContent = captions[index];
        }});
      }});
    }})();
  </script>
</div>
'''
    VIZ.mkdir(parents=True, exist_ok=True)
    path = VIZ / "zk5511-ree-v8-clear-preview.html"
    path.write_text(fragment, encoding="utf-8")
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [render_overview(audit()), render_depth_atlas(), render_candidate_details()]
    for path in paths:
        print(path)
    print(write_fragment(paths))


if __name__ == "__main__":
    main()
