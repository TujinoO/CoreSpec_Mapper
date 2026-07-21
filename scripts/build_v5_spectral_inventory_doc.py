from __future__ import annotations

"""Generate the V5 private spectral database inventory from shipped resources."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = ROOT / "src" / "corespec_mapper" / "resources"
DATABASE = RESOURCE_DIR / "corespec_spectral_v5.sqlite3"
KNOWLEDGE = RESOURCE_DIR / "mineral_catalog_v5.json"
RUNTIME = RESOURCE_DIR / "mineral_catalog_v5_runtime.json"
RESERVE = RESOURCE_DIR / "mineral_recognition_reserve_v5.json"
OUTPUT = ROOT / "docs" / "CoreSpec_Mapper_V5_私有光谱数据库清单与识别能力_2026-07-20.md"


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _windows(values: list[list[float]]) -> str:
    return "；".join(f"{float(low):g}–{float(high):g}" for low, high in values)


def _features(values: list[dict]) -> str:
    result = []
    for item in values:
        if "center" in item:
            text = f"{float(item['center']):g}"
            if item.get("kind"):
                text += f" {item['kind']}"
            result.append(text)
        elif "range" in item:
            result.append(_windows([item["range"]]))
    return "、".join(result)


def main() -> int:
    knowledge = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    reserve = json.loads(RESERVE.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{DATABASE.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        summary = json.loads(connection.execute("SELECT value FROM metadata WHERE key='summary_json'").fetchone()[0])
        candidates = {
            row["primary_phase_id"]: int(row["count"])
            for row in connection.execute(
                """SELECT m.primary_phase_id, COUNT(*) AS count
                   FROM measurement m JOIN qc_result q ON q.measurement_id=m.measurement_id
                   WHERE q.qc_status='eligible' GROUP BY m.primary_phase_id"""
            )
        }
        candidate_samples = {
            row["primary_phase_id"]: int(row["count"])
            for row in connection.execute(
                """SELECT m.primary_phase_id, COUNT(DISTINCT m.sample_id) AS count
                   FROM measurement m JOIN qc_result q ON q.measurement_id=m.measurement_id
                   WHERE q.qc_status='eligible' GROUP BY m.primary_phase_id"""
            )
        }
        sources = list(connection.execute(
            """SELECT s.source_id, s.source_family, s.source_role, s.measurement_geometry,
                      s.wavelength_min_nm, s.wavelength_max_nm, COUNT(m.measurement_id) AS measurement_count
               FROM source s LEFT JOIN measurement m ON m.source_id=s.source_id
               GROUP BY s.source_id ORDER BY s.source_id"""
        ))
        role_rows = list(connection.execute(
            """SELECT s.source_role, COUNT(DISTINCT s.source_id) AS source_count, COUNT(m.measurement_id) AS measurement_count
               FROM source s LEFT JOIN measurement m ON m.source_id=s.source_id
               GROUP BY s.source_role ORDER BY s.source_role"""
        ))
        purity_rows = list(connection.execute("SELECT purity_status, COUNT(*) AS count FROM measurement GROUP BY purity_status ORDER BY purity_status"))
        qc_rows = list(connection.execute("SELECT qc_status, COUNT(*) AS count FROM qc_result GROUP BY qc_status ORDER BY qc_status"))
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()

    enabled = [key for key, value in knowledge["minerals"].items() if value["recognition_enabled"]]
    validated = [key for key in enabled if knowledge["minerals"][key]["support_level"] == "validated_swir"]
    experimental = [key for key in enabled if knowledge["minerals"][key]["support_level"] == "experimental_swir"]
    disabled = [key for key, value in knowledge["minerals"].items() if not value["recognition_enabled"]]
    runnable_candidate_count = sum(candidates.get(key, 0) for key in enabled)

    lines = [
        "# CoreSpec Mapper V5 私有光谱数据库清单与识别能力",
        "",
        f"- 数据库 release：{summary['database_release']}",
        f"- schema：{summary['schema_version']}",
        f"- Catalog：{summary['catalog_version']}",
        "- 核对日期：2026-07-20",
        "- 数据文件：`src/corespec_mapper/resources/corespec_spectral_v5.sqlite3`",
        "- 预留名单：`src/corespec_mapper/resources/mineral_recognition_reserve_v5.json`",
        "",
        "## 1. 结论先行",
        "",
        f"V5.1 内置数据库仍完整保存 27 个源光谱库、{summary['measurement_count']:,} 条测量和 {summary['sample_count']:,} 个归并样品。知识 Catalog 已由 16 项扩展到 {summary['mineral_count']} 项，其中 {len(enabled)} 种矿物具备可运行的 SWIR 专家配置：{len(validated)} 种为已验证默认目标，{len(experimental)} 种为实验级可选目标；赤铁矿、针铁矿和石英共 {len(disabled)} 项因当前传感器/专家物理条件不足而保持关闭。",
        "",
        f"本次加入黄钾铁矾、绿脱石、滑石、透闪石、阳起石、黑云母、金云母、菱铁矿、海泡石、蛭石和水铵长石后，eligible 纯相测量由 288 条增加到 {summary['eligible_measurements']} 条；其中当前 24 个可运行 SWIR 目标合计 {runnable_candidate_count} 条。新增目标均标记为 `experimental_swir`，代表技术链路已经可运行，不代表矿物学精度已经通过真值验证。",
        "",
        f"其余 {reserve['summary']['reserved_phase_count']} 个自动解析矿相标签、{reserve['summary']['reserved_measurement_count']} 条纯相测量保存在禁用预留注册表中。它们不会出现在当前识别树或运行时输出中，必须补齐专家知识和验收记录后才能逐项启用。",
        "",
        "## 2. 数据库核心统计与完整性",
        "",
        "| 项目 | 当前值 |",
        "|---|---:|",
        f"| SQLite 文件大小 | {DATABASE.stat().st_size:,} bytes（约 {DATABASE.stat().st_size / 1024 / 1024:.2f} MiB） |",
        f"| SHA-256 | `{_hash(DATABASE)}` |",
        f"| 源 ENVI 光谱库 | {summary['source_count']} |",
        f"| 完整测量曲线 | {summary['measurement_count']:,} |",
        f"| 归并样品 | {summary['sample_count']:,} |",
        f"| Catalog 目标矿物 | {summary['mineral_count']} |",
        f"| 可运行 SWIR 目标 | {len(enabled)} |",
        f"| 已验证 / 实验级 SWIR | {len(validated)} / {len(experimental)} |",
        f"| Catalog 纯相且 eligible | {summary['eligible_measurements']} |",
        f"| 标签冲突测量 | {summary['label_conflict_measurements']}（均排除） |",
        f"| `PRAGMA integrity_check` | `{integrity}` |",
        "",
        "eligible 只表示：来源角色为矿物标准、名称判为单一矿相、已进入 Catalog、数值 QC 合格且无标签冲突。每次运行仍会按传感器波长/FWHM/坏波段做兼容性过滤，按物理样品去重，再以 robust medoid 和谱形/来源/测量几何多样性选择少量代表谱。",
        "",
        "## 3. 数据来源与质量分布",
        "",
        "### 3.1 来源角色",
        "",
        "| 来源角色 | 光谱库数 | 测量数 |",
        "|---|---:|---:|",
    ]
    for row in role_rows:
        lines.append(f"| `{row['source_role']}` | {row['source_count']} | {row['measurement_count']:,} |")
    lines.extend(["", "### 3.2 纯度/材料判定", "", "| 状态 | 测量数 |", "|---|---:|"])
    for row in purity_rows:
        lines.append(f"| `{row['purity_status']}` | {row['count']:,} |")
    lines.extend(["", "### 3.3 QC 状态", "", "| 状态 | 测量数 |", "|---|---:|"])
    for row in qc_rows:
        lines.append(f"| `{row['qc_status']}` | {row['count']:,} |")

    lines.extend([
        "",
        "## 4. 27 个 Catalog 目标与实际候选数量",
        "",
        "| 一级分类 | 矿物 | 竞争族 | eligible 测量 / 独立样品 | 必需窗口（nm） | 关键特征（nm） | 状态 |",
        "|---|---|---|---:|---|---|---|",
    ])
    taxonomy_names = {key: value["display_name_zh"] for key, value in knowledge["taxonomy_groups"].items()}
    for mineral_id, mineral in knowledge["minerals"].items():
        status = "已验证 SWIR；默认勾选" if mineral["support_level"] == "validated_swir" else (
            "实验级 SWIR；可选、默认不勾选" if mineral["support_level"] == "experimental_swir" else "关闭；仅保留资源/定义"
        )
        lines.append(
            f"| {taxonomy_names[mineral['taxonomy_group']]} | {mineral['display_name_zh']} `{mineral_id}` | `{mineral['spectral_family']}` | "
            f"{candidates.get(mineral_id, 0)} / {candidate_samples.get(mineral_id, 0)} | {_windows(mineral['required_windows_nm'])} | {_features(mineral['key_features_nm'])} | {status} |"
        )
    lines.extend([
        "",
        "透闪石/阳起石和黑云母/金云母在前端分别作为独立矿物选择，在后端分别进入同一竞争族。菱铁矿与方解石在约 2.33 µm 高度混淆，透闪石与阳起石也存在连续固溶体特征；这些输出必须结合完整谱形、参考谱共识、竞争间隔和地质背景解释，不能只凭单个吸收中心硬判。",
        "",
        "## 5. 两级分类与后端竞争族",
        "",
        "### 5.1 前端一级分类",
        "",
        "| 一级分类 | 二级矿物成员 |",
        "|---|---|",
    ])
    for taxonomy_id, taxonomy in knowledge["taxonomy_groups"].items():
        members = "、".join(knowledge["minerals"][key]["display_name_zh"] for key in taxonomy["members"])
        lines.append(f"| {taxonomy['display_name_zh']} / `{taxonomy_id}` | {members} |")
    lines.extend(["", "### 5.2 后端光谱竞争族", "", "| 竞争族 | 成员 | 检测窗口（nm） | 关键运行时特征 |", "|---|---|---|---|"])
    for family_id, family in knowledge["spectral_families"].items():
        members = "、".join(knowledge["minerals"][key]["display_name_zh"] for key in family["competition_members"])
        group = runtime["groups"].get(family_id)
        if group is None:
            lines.append(f"| `{family_id}` | {members} | — | 专家未实现 |")
            continue
        windows = _windows(group["detection_windows_nm"])
        features = "、".join(item["id"] for item in group["features"])
        lines.append(f"| `{family_id}` | {members} | {windows} | {features} |")

    lines.extend([
        "",
        "## 6. 本次新增 11 种矿物的识别约束",
        "",
        "- 黄钾铁矾：使用约 1.47、1.855 和 2.265 µm 多窗口证据，并与明矾石同族竞争。",
        "- 绿脱石：使用 1.4/1.9 µm 水—OH 和约 2.29/2.40 µm Fe-OH 组合，与蒙脱石、海泡石、蛭石竞争。",
        "- 滑石：使用约 2.29、2.31、2.39、2.465 µm Mg-OH 谱形，并与绿泥石、云母类共同竞争。",
        "- 透闪石/阳起石：使用约 2.11、2.315、2.385 µm 组合；两者分相为实验级条件性输出，要求较高竞争间隔。",
        "- 黑云母/金云母：黑云母 SWIR 特征偏弱，必须依赖参考谱共识和竞争间隔；两者分别输出、同族竞争。",
        "- 菱铁矿：以 2.335 µm 碳酸根为主，并辅助使用约 1.945 µm；与方解石高度混淆，必须结合完整谱形。",
        "- 海泡石/蛭石：联合使用 1.4、1.9、2.31–2.32 和约 2.39 µm 水—OH/Mg-Fe-OH 证据。",
        "- 水铵长石：使用约 1.56、2.035–2.055 和 2.12 µm NH4 特征，独立进入铵长石专家族。",
        "",
        "### 6.1 3DSSZ 实测数据链路审计",
        "",
        "使用 `configs/v5_3dssz_validation.json` 对新增 11 种矿物执行 V5.1 完整审计，结果为 `Ready`：11/11 目标的必需窗口覆盖率均为 1.0，每种均选出 3–4 条当前传感器可用代表谱；连同内部混淆矿物共选出 51 条标准谱。该审计确认数据读取、掩膜、传感器门禁、参考谱优选和竞争族配置能够跑通；输入仍缺 FWHM，且没有像元级矿物真值，因此不构成准确率结论。",
        "",
        "## 7. 标准谱自动优选链路",
        "",
        "1. 只查询同一矿相、`declared_pure`、`eligible`、矿物标准来源的测量；",
        "2. 检查数据物理属性和全部必需窗口至少 97% 覆盖；",
        "3. 已知 FWHM 时做高斯卷积，未知时做跨间隙受限插值；",
        "4. 计算波长覆盖、吸收深度、粗糙度和谱形质量；",
        "5. 先按 `sample_id` 去重，避免重复测量放大先验权重；",
        "6. 以 robust medoid 起步，再按谱形差异、来源族和测量几何多样性补充；",
        "7. 运行时输出每条标准谱的选择/拒绝原因、曲线预览和来源审计。",
        "",
        "## 8. 其他矿物的禁用预留名单",
        "",
        f"预留注册表含 {reserve['summary']['reserved_phase_count']} 个名称级矿相、{reserve['summary']['reserved_measurement_count']} 条纯相测量。以下名称来自源库主标签自动解析，只是待专家复核的入口，不代表已经确认矿相、建立了可识别专家或允许程序输出。",
        "",
        "| 预留 phase_id | 测量数 | 独立样品数 | 来源数 | 状态 |",
        "|---|---:|---:|---:|---|",
    ])
    for item in reserve["entries"]:
        lines.append(f"| `{item['phase_id']}` | {item['measurement_count']} | {item['independent_sample_count']} | {item['source_count']} | `reserved_expert_required` / 禁用 |")
    lines.extend([
        "",
        "逐项启用至少需要：确认矿物身份和别名、确定一级分类与竞争族、定义必需波段和诊断特征、列出主要混淆矿物、设置三档特征门禁、完成参考谱 QC、传感器覆盖测试、混淆竞争测试和真值/地质验收。仅增加名称或库中存在曲线不能自动获得识别权限。",
        "",
        "## 9. 27 个源光谱库清单",
        "",
        "| # | 源库 ID | 来源族 | 角色 | 测量数 | 波长范围（nm） | 测量几何 |",
        "|---:|---|---|---|---:|---|---|",
    ])
    for index, row in enumerate(sources, 1):
        lines.append(
            f"| {index} | `{row['source_id']}` | `{row['source_family']}` | `{row['source_role']}` | {row['measurement_count']} | "
            f"{row['wavelength_min_nm']:.1f}–{row['wavelength_max_nm']:.1f} | `{row['measurement_geometry']}` |"
        )
    lines.extend([
        "",
        "全部 27 个来源的许可状态仍为 `unverified`；本地技术使用不改变这一事实，对外再分发数据库或制作商业安装包前必须逐源完成许可核查。",
        "",
        "## 10. 能力边界与发布判据",
        "",
        "1. `experimental_swir` 表示曲线、窗口、特征、门禁、竞争族和运行链路已经接通，不等于准确率已验证。",
        "2. 当前四组历史 V5.0 实测运行没有像元级 XRD、拉曼、薄片或点光谱真值，不能据此给出矿物识别准确率。",
        "3. 赤铁矿/针铁矿仍缺完整校准 VNIR 专家条件；石英仍缺 TIR 传感器和兼容发射率参考谱，因此保持关闭。",
        "4. 影像坏条带、列噪声、边缘、孤立小斑块和材料掩膜质量仍会直接控制结果可靠性；空间聚集性只能作为地质合理性复核，不能替代光谱证据。",
        "5. 数据库曲线数量多不等于识别更准确；独立样品数、谱形代表性、混淆矿物覆盖、传感器适配和真值验收同等重要。",
        "6. 新增光谱资源应先进入隔离 staging，完成来源/许可/物理属性登记、名称解析、纯度与数值 QC、样品去重和标签冲突检查，再由专家审核进入 Catalog。",
        "",
        "## 11. 数据库表和可追溯字段",
        "",
        "| 表 | 主要内容 |",
        "|---|---|",
        "| `metadata` | release、schema、Catalog 版本和汇总 |",
        "| `taxonomy_group` | 前端一级矿物分类 |",
        "| `spectral_family` | 后端光谱竞争族和专家 |",
        "| `mineral` | 27 个目标、成熟度、窗口和启用状态 |",
        "| `source` | 27 个源库、角色、物理属性、几何、波长轴和文件哈希 |",
        "| `sample` | 归并样品、主相、组成相、纯度、样品码和解析原因 |",
        "| `measurement` | 原始名称、源索引、数值统计、压缩曲线和哈希 |",
        "| `qc_result` | eligible/review/excluded、数值状态、冲突和原因 |",
        "",
        "数据库以只读 URI 打开；运行 Manifest 同时记录数据库 release、Catalog 版本和 SHA-256，保证同名资源内容发生变化时仍可审计。",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.resolve()),
        "database_sha256": _hash(DATABASE),
        "catalog_targets": len(knowledge["minerals"]),
        "runnable_swir_targets": len(enabled),
        "eligible_measurements": summary["eligible_measurements"],
        "reserve_phase_count": reserve["summary"]["reserved_phase_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
