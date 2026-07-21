# CoreSpec Mapper V5 桌面版使用与 V4 迁移指南

- 适用版本：5.0.0
- 日期：2026-07-20
- 默认桌面入口：`corespec-desktop`
- 旧版桌面入口：`corespec-desktop-v4`

## 1. V5 使用要点

1. 只选择主分析 ENVI 立方体、材料掩膜方式和目标矿物；不再选择光谱库目录。
2. 内置私有光谱数据库随软件包安装并自动以只读方式打开。
3. RGB、NIR、SWIR 可以作为伴随数据登记，但当前不会自动配准或逐像元融合。
4. 真正参与矿物识别的是 `inputs.analysis_image`。联合分析必须先生成已配准的连续立方体，再把它设为主分析数据。
5. 默认使用自动材料掩膜；存在可靠同网格人工掩膜时可切换为外部模式。
6. 默认勾选 7 个已验证 SWIR 目标；实验级目标应在有针对性真值复核时使用。
7. 后端固定写出 Conservative、Balanced、Sensitive 三档；Balanced 是默认解释图。
8. 工程质量等级不是矿物学准确率。

## 2. 环境安装

在 PowerShell 中：

```powershell
cd E:\Code\CoreSpec_Mapper
python -m pip install -e ".[desktop,test]"
```

不安装 editable package 时，可临时设置：

```powershell
$env:PYTHONPATH = "src"
python -m corespec_mapper --help
```

核心运行依赖 NumPy；桌面端需要 PySide6。内置 SQLite 光谱数据库作为 package data 安装，无需单独复制 `spec_lib` 或配置环境变量。

## 3. 启动桌面版

安装后：

```powershell
corespec-desktop
```

或：

```powershell
corespec desktop
```

未安装包时：

```powershell
$env:PYTHONPATH = "src"
python -m corespec_mapper desktop
```

V4 兼容桌面仍可启动：

```powershell
corespec-desktop-v4
corespec desktop-v4
```

## 4. 七步桌面流程

### 4.1 项目

填写项目名称、可选说明和输出根目录。每次运行保存到：

`<输出根目录>/<项目安全名称>/<run_id>/`

已有目录不会覆盖。建议项目名称表达钻孔或数据批次，`run_id` 表达试验目的。

### 4.2 数据

选择主分析立方体。它必须：

- 是可由 ENVI 头文件描述的 `.dat/.hdr` 或可解析等价路径；
- 含严格递增的真实波长；
- 数据物理与专家兼容，当前生产路径使用 reflectance；
- 覆盖所选矿物的诊断窗口。

可选 RGB/NIR/SWIR 伴随文件用于记录来源、尺寸和头文件指纹。桌面可以按同目录文件名尝试发现伴随数据；CLI 不会自动发现，必须显式填写或留空。

页面提示“不会在未配准时强行逐像元融合”是硬边界，不是建议。

### 4.3 掩膜

自动模式适合没有同网格掩膜的数据。审计会检查：

- 有效材料像元数和覆盖比例；
- 自动阈值方法；
- 连通域数与清除的小组件；
- 是否过小、近全幅或高度碎片化。

正式材料掩膜写入 `capability/material_mask.dat`，并在 `previews/material_mask.png` 生成快速预览。

外部模式要求掩膜与主立方体行列完全一致。不要把 RGB 尺寸掩膜直接交给 SWIR 主立方体。

### 4.4 矿物

矿物树按“一级矿物族 → 二级矿物”组织。每个目标显示：

- Catalog 成熟度；
- 当前传感器审计支持级别；
- 必需窗口和关键特征；
- 内置 eligible 候选数；
- 不可用原因。

默认 7 个已验证 SWIR 目标：

`calcite, dolomite, anhydrite, gypsum, illite, montmorillonite, kaolinite`

可选实验级 SWIR 目标：

`muscovite, dickite, pyrophyllite, chlorite, alunite, epidote`

赤铁矿、针铁矿和石英在当前专家条件下锁定。数据库中有参考曲线不等于当前传感器可识别。

### 4.5 模式

Balanced 是默认查看与解释档；Conservative 用于高可信核心，Sensitive 用于漏识别检查。三档共用同一套参考谱和光谱证据，最终保持嵌套。`mode.profile` 只改变结果页默认显示哪一档，不改变科学运行。

高级参数通常保留默认值：

| 参数 | 默认 |
|---|---:|
| SG 窗口 / 阶数 | 11 / 2 |
| 全深度样本块 | 12 |
| 每块行数 | 64 |
| 科学样本最大行数 | 128 |
| 推理分块行数 | 64 |
| 每矿物最大参考谱 | 服务默认 6；示例配置 4 |
| 列校正强度 / 邻域 | 0.70 / 2 |
| 边缘风险宽度 | 2 |

Quick Calibration 控件当前被禁用，并明确提示缺少像元级真值；桌面保存配置时强制 `quick_calibration=false`。“同时输出严格、平衡、敏感三档”固定勾选且禁用，保存时强制 `write_all_profiles=true`。这两个控件都不是可操作的科学开关。

### 4.6 证据审查

先执行“项目审计”。页面会显示：

- 自动入选的标准谱曲线；
- 每条参考谱的来源、样品、选择顺序和原因；
- 每个矿物的可用性和参考数；
- 各组阈值安全候选范围；
- FWHM、掩膜、固定列和真值风险。

Audit 还未运行完整阈值目标搜索，因此此时显示候选范围，不显示伪造的“最终阈值”。完整运行后，页面从 `thresholds.json`/Manifest 加载最终值和候选数。

### 4.7 运行结果

运行完成后优先查看：

1. `previews/comparison_balanced.png`；
2. 各组 `final_balanced.dat`；
3. `confidence/confidence.dat`；
4. 各组 `stripe_noise_mask.dat` 和 `column_risk_score.dat`；
5. `reports/quality_report.md`；
6. `thresholds.json`；
7. `run_manifest.json`。

伊利石组还应查看 `aloh_wavelength_subtype.dat`。它描述短/中/长波 Al-OH，不是确定的伊利石—白云母相位分离。

## 5. V5 最小配置示例

此示例没有任何光谱库路径：

```json
{
  "schema_version": 1,
  "application_version": "5.0.0",
  "project": {
    "name": "DrillCore_V5",
    "description": "SWIR mineral evidence mapping",
    "non_destructive": true
  },
  "inputs": {
    "analysis_image": "D:/Project/Data/SWIR.dat",
    "analysis_domain": "swir",
    "data_physics": "reflectance",
    "input_is_smoothed": false,
    "rgb": null,
    "nir": null,
    "swir": "D:/Project/Data/SWIR.dat"
  },
  "mask": {
    "mode": "automatic",
    "path": null,
    "minimum_component_pixels": 64,
    "edge_guard_pixels": 2
  },
  "minerals": {
    "requested": [
      "calcite",
      "dolomite",
      "anhydrite",
      "gypsum",
      "illite",
      "montmorillonite",
      "kaolinite"
    ],
    "include_internal_confusers": true
  },
  "mode": {
    "profile": "balanced",
    "automatic": true,
    "quick_calibration": false,
    "write_all_profiles": true
  },
  "advanced": {
    "preprocessing": {
      "sg_window": 11,
      "sg_polyorder": 2
    },
    "sampling": {
      "blocks": 12,
      "block_rows": 64,
      "maximum_sample_rows": 128
    },
    "execution": {
      "chunk_rows": 64
    },
    "library_ensemble": {
      "maximum_representatives_per_mineral": 4
    },
    "artifact_control": {
      "edge_width": 2
    }
  }
}
```

`profile` 仅是默认查看偏好；`quick_calibration=false` 和 `write_all_profiles=true` 是 5.0.0 桌面的固定契约。

仓库内可直接复制 `configs/v5_default.json`。

## 6. 外部同网格掩膜示例

只需更改掩膜段：

```json
{
  "mask": {
    "mode": "external",
    "path": "D:/Project/Data/core_mask.dat",
    "minimum_component_pixels": 64,
    "edge_guard_pixels": 2
  }
}
```

外部掩膜模式默认不删除人工小连通域。`minimum_component_pixels` 仍会进入配置和审计，但现行服务对外部掩膜使用“保留组件”的安全策略。

## 7. 多源/连续立方体示例

### 7.1 原始未配准 RGB/NIR/SWIR

以 SWIR 为主分析数据，其他文件仅登记：

```json
{
  "inputs": {
    "analysis_image": "D:/Batch/SWIR.dat",
    "analysis_domain": "swir",
    "data_physics": "reflectance",
    "input_is_smoothed": false,
    "rgb": "D:/Batch/RGB.dat",
    "nir": "D:/Batch/NIR.dat",
    "swir": "D:/Batch/SWIR.dat"
  }
}
```

### 7.2 已配准 691–2518 nm 连续立方体

把协调结果设为主分析数据：

```json
{
  "inputs": {
    "analysis_image": "D:/Fusion/analysis/harmonized_lowres.dat",
    "analysis_domain": "auto",
    "data_physics": "reflectance",
    "input_is_smoothed": false,
    "rgb": null,
    "nir": null,
    "swir": null
  }
}
```

即使连续立方体覆盖到 691 nm，当前铁氧化物矿物级专家仍未实现，且缺失 450–691 nm；不要手工解锁赤铁矿/针铁矿。

## 8. CLI 命令

### 8.1 查看 V5 Catalog 和数据库统计

```powershell
corespec v5-catalog
corespec v5-catalog --output output/v5_catalog.json
```

输出包括知识 Catalog、Runtime Catalog、数据库 27/1783/1143/16/288 统计和每矿物候选数。

### 8.2 输入、掩膜、能力与标准谱审计

```powershell
corespec v5-audit --config configs/v5_default.json --output output/v5_audit.json
```

### 8.3 单独导出本次传感器的标准谱选择

```powershell
corespec v5-library --config configs/v5_default.json --output output/v5_library_selection.json
```

### 8.4 完整运行

```powershell
corespec v5-run `
  --config configs/v5_default.json `
  --output-root output `
  --run-id drillcore_v5_balanced
```

可选输出行范围：

```powershell
corespec v5-run `
  --config configs/v5_default.json `
  --output-root output `
  --run-id drillcore_v5_pilot `
  --start-line 0 `
  --stop-line 800
```

`run_id` 只能包含字母、数字、连字符和下划线，且不能使用 Windows 保留名。

`--start-line/--stop-line` 当前只裁剪最终写出范围；组级 SAM、列风险与阈值标定仍扫描并分配全幅。因此它适合检查局部输出合同，不应被当作节省大幅运行时间或内存的 ROI 模式。

### 8.5 审计快照复用说明

- 桌面中先 Audit、再 Run：同一进程且配置/输入/数据库/Catalog 未变时，复用 `V5AuditSnapshot`。
- 直接执行一次 `v5-run`：运行内部只准备一次审计快照，再进入映射。
- 分别启动 `v5-audit` CLI 和 `v5-run` CLI：这是两个进程，内存快照不能跨进程复用；`v5-run` 会重新审计一次。这不是同一次 Run 内的隐藏重复审计。

## 9. 输出目录

典型运行：

```text
<run>/
  project.csmproj
  config.resolved.json
  audit.json
  summary.json
  thresholds.json
  run_manifest.json
  capability/
  library_ensemble/
  groups/
  confidence/
  previews/
  reports/
  tables/
```

每个处理族目录包含：

- `final_conservative.dat`、`final_balanced.dat`、`final_sensitive.dat`；
- `group_sam_score.dat`、`group_mineral_sam_scores.dat`；
- `column_threshold.dat`、`column_candidates.dat`；
- `raw_mineral_scores.dat`、`calibrated_mineral_scores.dat`；
- `absorption_depth.dat`、`classification_margin.dat`；
- 诊断 feature 栅格；
- `confidence.dat`、`stability.dat`、`rejection_reason.dat`；
- `column_risk_score.dat`、`stripe_noise_mask.dat`、`edge_risk_score.dat`；
- `summary.json`。

`run_manifest.json` 不把自身列入自身输出清单，以避免递归/陈旧哈希。

## 10. V4 配置迁移

### 10.1 字段映射

| V4 | V5 | 迁移说明 |
|---|---|---|
| 顶层 `analysis_image` | `inputs.analysis_image` | 必需 |
| 顶层 `analysis_mask` | `mask.mode=external` + `mask.path` | 没有可靠掩膜时改用 automatic |
| 顶层 `analysis_input_is_smoothed` | `inputs.input_is_smoothed` | 保持真实预处理状态 |
| `v4.project_name` | `project.name` | 输出根目录改由桌面或 CLI 参数提供 |
| `v4.data_physics` | `inputs.data_physics` | 当前通常为 reflectance |
| `v4.requested_minerals` | `minerals.requested` | ID 保持小写英文 |
| `v4.preprocessing` | `advanced.preprocessing` | 默认 11/2 |
| `v4.sampling` | `advanced.sampling` | 增加 `maximum_sample_rows` |
| 顶层/`v4.chunk_rows` | `advanced.execution.chunk_rows` | 默认 64 |
| `v4.library_ensemble.maximum_representatives_per_mineral` | `advanced.library_ensemble.maximum_representatives_per_mineral` | 仍可设置 |
| `v4.artifact_control` | `advanced.artifact_control` | 列校正有安全默认值 |
| `v4.spectral_library_root` | 删除 | V5 禁止要求用户提供内置光谱库路径 |
| `v4.policy_overrides` | 通常删除 | V5 默认走 Catalog 有界自适应搜索 |

### 10.2 V4 示例迁移

V4：

```json
{
  "analysis_image": "D:/Core/SWIR_SG.dat",
  "analysis_mask": "D:/Core/mask.dat",
  "analysis_input_is_smoothed": true,
  "v4": {
    "project_name": "Core_A",
    "data_physics": "reflectance",
    "spectral_library_root": "E:/Code/CoreSpec_Mapper/spec_lib",
    "requested_minerals": ["calcite", "dolomite", "illite"]
  }
}
```

迁移为：

```json
{
  "schema_version": 1,
  "application_version": "5.0.0",
  "project": {
    "name": "Core_A",
    "non_destructive": true
  },
  "inputs": {
    "analysis_image": "D:/Core/SWIR_SG.dat",
    "analysis_domain": "swir",
    "data_physics": "reflectance",
    "input_is_smoothed": true,
    "rgb": null,
    "nir": null,
    "swir": "D:/Core/SWIR_SG.dat"
  },
  "mask": {
    "mode": "external",
    "path": "D:/Core/mask.dat",
    "minimum_component_pixels": 64,
    "edge_guard_pixels": 2
  },
  "minerals": {
    "requested": ["calcite", "dolomite", "illite"],
    "include_internal_confusers": true
  }
}
```

不要把 `spectral_library_root` 复制到 V5 配置。桌面保存时会主动删除这个字段。

## 11. V4 兼容策略

V3/V4 CLI、配置和桌面仍保留，便于历史成果复算：

```powershell
corespec v4-audit --config configs/nc1_v4.json --output output/v4_audit.json
corespec v4-run --config configs/nc1_v4.json --output-root output --run-id legacy_v4
corespec desktop-v4
```

新项目应使用 V5；不要用 V4 的输出目录覆盖 V5 成果。V4 和 V5 结果比较时，应同时记录：

- 软件版本和 Catalog；
- 光谱数据库/库目录身份；
- 主立方体和掩膜；
- 目标矿物与内部竞争者；
- 预处理和阈值来源；
- 三档定义；
- 是否有真值。

## 12. 常见问题

### 12.1 提示材料掩膜过大

自动掩膜覆盖超过 85% 会阻断，通常表示托盘/背景进入材料。检查：

- 主立方体是否是反射率而不是未校准辐亮度；
- 是否选错数据文件；
- 是否为裁剪过小、边界全是岩心的 ROI；
- 是否应提供可靠同网格外部掩膜。

不要为跑通而简单把最大覆盖率改成 100%。

### 12.2 提示材料掩膜高度碎片化

这是 warning，不一定阻断。查看运行后的 `material_mask.png` 和组件统计。最终 3DSSZ 原始外部掩膜保留 159 个组件；最终 ZKH3 原始自动掩膜清除 9,338 个小组件像元后仍保留 263 个组件，两者都被明确警告。

### 12.3 提示 FWHM_UNKNOWN

参考谱改用受限插值。结果可以用于工程复核，但必须保留该限制。优先从传感器标定文件补回每波段 FWHM。

### 12.4 某矿物零检出

零检出不是程序失败。依次查看：

1. `mineral_observability.json`；
2. `column_candidates.dat`；
3. `absorption_depth.dat` 和 feature 栅格；
4. `classification_margin.dat`；
5. `rejection_reason.dat`；
6. `thresholds.json`；
7. 原始光谱和影像质量。

禁止用固定配额强制生成矿物。

### 12.5 结果呈固定列或细直条

联合查看 `column_risk_score.dat`、`stripe_noise_mask.dat`、`classes_before_artifact_filter.dat` 和最终分类。若强证据仍被保留，应在地质图像上确认它是否为真实裂隙/脉体。

### 12.6 为什么有 RGB/NIR 却不能选赤铁矿和针铁矿

当前 NIR 从 691 nm 开始，缺失 450–691 nm；RGB 不是校准高光谱 VNIR。矿物级分相专家未实现。数据库候选只表示资源已整理，不表示专家已具备可用输入。

### 12.7 为什么不能识别石英

当前无 TIR 发射率数据和兼容发射率库。VNIR/SWIR 反射率不能稳定替代石英的 TIR Reststrahlen 证据。

## 13. 验收建议

每次正式运行至少检查：

- Manifest 状态和资源哈希；
- 材料掩膜覆盖及背景泄漏；
- FWHM 和坏波段；
- 入选参考谱的来源与谱形；
- 阈值候选目标和最终值；
- 三档嵌套；
- 同类空间支持和固定列密度；
- 零检出警告；
- 地质编录/PPT 的定性一致性；
- 是否存在独立矿物学真值。

仓库中四组 V5 运行证明原始和协调数据可端到端完成，但全部缺少像元真值。因此正式结论应写“工程流程通过、矿物学准确率待真值验证”，不写百分比准确率。
