# CoreSpec Mapper

**面向岩心高光谱影像的可配置、可复现、可审计矿物证据填图软件。**

CoreSpec Mapper 提供 SWIR 矿物候选识别、材料掩膜、标准谱优选、自适应阈值、空间伪影控制、三档证据输出和中文桌面工作流，同时保留配置、阈值、输入指纹、质量报告与运行清单，适合科研验证、方法复现和受控工程应用。

> 科学边界：工程质量等级、自动化测试或历史弱标签回归均不等同于矿物学准确率。没有独立 XRD、拉曼、薄片或点光谱真值时，输出应表述为矿物光谱证据或经验异常候选。

> 当前稳定版本：**V5.3.0 Stable**（2026-07-21）
> Windows 10/11 x64 · 当前用户安装 · CPU 独立运行 · 当前源码 155 项测试通过、3 项条件性跳过（2026-09-02）

## 数据与成果存储

源码仓库不作为原始高光谱数据、中间栅格、实验迭代、运行成果、渲染 QA、构建树或本地打包环境的长期存储位置。新运行应使用仓库外的专用数据目录，例如 `D:\CoreSpec_Mapper_Data\runs`；分类、保留边界、清理审计与恢复方法见 [数据与成果外置存储规范](docs/CoreSpec_Mapper_数据与成果外置存储规范_2026-09-02.md)。

## 版本与下载

| 项目 | 内容 |
|---|---|
| 当前版本 | `5.3.0 Stable` |
| Git 标签 | [`v5.3.0`](https://github.com/TujinoO/CoreSpec_Mapper/releases/tag/v5.3.0) |
| Windows 安装包 | [`CoreSpec_Mapper_V5.3.0_Stable_Setup.exe`](https://github.com/TujinoO/CoreSpec_Mapper/releases/download/v5.3.0/CoreSpec_Mapper_V5.3.0_Stable_Setup.exe) |
| SHA256 文件 | [`CoreSpec_Mapper_V5.3.0_Stable_Setup.sha256`](https://github.com/TujinoO/CoreSpec_Mapper/releases/download/v5.3.0/CoreSpec_Mapper_V5.3.0_Stable_Setup.sha256) |
| 安装包大小 | 约 206.96 MiB |
| SHA256 | `DC4E190B6285A5A715D1FF4CE2641BF3365A587BAF0E9F9852115A3428EBB3A2` |

普通用户请从 [GitHub Releases](https://github.com/TujinoO/CoreSpec_Mapper/releases) 下载安装包，不需要克隆源码，也不需要安装 Python、Conda、PySide6 或 Torch。当前安装包尚未进行商业代码签名，Windows SmartScreen 可能显示“未知发布者”；请先核对 SHA256，再选择“更多信息 → 仍要运行”。

V5.3 当前自适应技术路线采用“三段式阈值职责”：大类 SAM 仅负责宽松、高召回地发现候选域；SFF、矿物诊断特征、参考谱一致性和相似矿物竞争负责后续严格细分；末端再按多深度段重复、固定探测器列风险与横向/斜向地质支撑清理条带噪声。Conservative、Balanced、Sensitive 使用同一连续证据排序，但分别解析独立且严格嵌套的项目证据门，不再共用一张布尔 QDA 掩膜。

NC-1 全景回归 `project_calibrated_v14_dominant_corridor_20260721` 使用 332,814 像元人工批准材料掩膜，质量 B、`publishable=true`。平衡档七类矿物均非零；逐矿物、多岩心段重复窄列、主导重复走廊和可剔除像元数均为 0。终检会把相距不超过 7 列的重复峰联合审查，避免同一条窄走廊内的相邻列互相提供伪地质支撑。

CoreSpec Mapper V5.3.0 是面向岩心高光谱影像的可配置、可复现、可审计矿物证据填图软件。V5.3 提供九步状态化桌面流程、应用内置智能岩心前景模型、标准谱独立确认、可编辑阈值联合试算、掩膜人工批准门、实时 ETA 和中文成果视图。

V5.3 的阈值优选不再只看组级 SAM：它在 Catalog 安全边界内联合评价分层场景证据、覆盖率、空间支持、固定列/边缘/孤立伪影风险，以及矿物特征门的可通过像元数。试算结果可在桌面端调整并重算，运行阶段会在全图上重新验证后才使用。无独立 XRD、拉曼、薄片或点光谱真值时，该流程不把无标签代理目标解释为矿物学准确率。

当项目提供同网格、人工复核的历史高可信分类时，V5.3 可启用项目级校准：按交错连续深度块划分训练与留出集，只学习组级检测阈值、矿物分数偏置、物理特征窗和紧凑证据门，不复制历史分类像元。伊利石、蒙脱石和高岭石在独立发现后还会进入共享跨组竞争，确保同一像元只保留一个粘土类胜者。

NC-1 项目提供 `validated_v3` 已验证配方路线：在 5,446 × 320 × 212 全景数据上，平衡/敏感两档共 14 个“档位×矿物”结果与冻结历史流程逐像元一致。该基准用于证明算法与参数回归等价，不能替代 XRD、拉曼、薄片或点光谱真值。

当前生产专家为 SWIR 反射率专家。软件能够对 24 个 SWIR 目标完成传感器条件检查、标准谱优选和后端配置，其中 7 个为桌面默认的已验证 SWIR 目标，17 个为实验级可选目标。赤铁矿/针铁矿仅保留 Fe 氧化物族资源，矿物级分相尚未实现；石英因无 TIR 数据和兼容发射率库而关闭。

## V5 核心能力

- 内置 `corespec_spectral_v5.sqlite3`：27 个源库、1,783 条测量、1,143 个样品、27 个 Catalog 目标、428 条纯相 eligible 候选。
- 24 个可运行 SWIR 目标：7 个已验证默认目标；原有 6 个及新增黄钾铁矾、绿脱石、滑石、透闪石、阳起石、黑云母、金云母、菱铁矿、海泡石、蛭石、水铵长石共 17 个实验级可选目标。
- `mineral_recognition_reserve_v5.json` 保留另外 241 个待专家复核矿相标签和 914 条纯相测量，全部禁用，不会仅凭名称自动开放识别。
- 一级矿物族、二级矿物的前端分类，以及面向相似谱形竞争的后端光谱族。
- 主分析立方体与可选 RGB/NIR/SWIR 伴随数据角色；未配准数据不会被强行融合。
- 自动材料掩膜只使用应用内置智能岩心前景模型；模型资产或 RGB 输入不完整时明确阻断，不再静默切换其他引擎。掩膜对过小、近全幅、碎片化和轮廓/裂缝型结果做质量门禁，并要求人工批准。
- 全深度分层抽样、Sensor Capability Card、SG 平滑和稳健列光谱偏差。
- 按样品去重的传感器适配标准谱；robust medoid 加谱形/来源/测量几何多样性优选。
- 组级 SAM、Catalog 有界自适应阈值、向量化分段线性连续统和相似矿物特征竞争。
- Conservative、Balanced、Sensitive 三档共享证据，并使用深度、特征、间隔、参考谱共识、稳定性和置信度门禁。
- 固定列、方向性细条带、狭长弱证据、小连通域、边缘和饱和风险控制。
- ENVI 栅格、标准谱曲线、阈值候选/最终值、拒绝瀑布、Al-OH 波长亚型、可缩放预览、CSV、JSON/Markdown/HTML 报告和完整 Manifest。
- 审计快照在同一进程内复用，避免同一次桌面流程重复审计和重复选谱。

## 能力边界

| 层级 | 矿物 | 状态 |
|---|---|---|
| 已验证 SWIR，默认勾选 | 方解石、白云石、硬石膏、石膏、伊利石、蒙脱石、高岭石 | 启用；仍需当前传感器能力门禁 |
| 实验级 SWIR | 白云母、地开石、叶蜡石、绿泥石、明矾石、绿帘石、黄钾铁矾、绿脱石、滑石、透闪石、阳起石、黑云母、金云母、菱铁矿、海泡石、蛭石、水铵长石 | 启用但不默认；需专项真值验收 |
| Fe 氧化物族 | 赤铁矿、针铁矿 | 当前 NIR 从 691 nm 开始且 RGB 非校准高光谱，矿物级关闭 |
| TIR 硅酸盐 | 石英 | 无 TIR 和兼容发射率参考谱，关闭 |

质量等级 A/B/C/D 衡量输入、输出完整性、三档嵌套、空间支持和伪影风险，不是矿物学准确率。现有工程运行和 NC-1 历史弱标签回归均没有像元级 XRD、拉曼、薄片或点光谱真值。

## 安装

### Windows 安装包（推荐）

1. 打开 [`v5.3.0` 发布页](https://github.com/TujinoO/CoreSpec_Mapper/releases/tag/v5.3.0)。
2. 下载 `CoreSpec_Mapper_V5.3.0_Stable_Setup.exe` 和同名 `.sha256` 文件。
3. 在下载目录打开 PowerShell，校验安装包：

```powershell
Get-FileHash .\CoreSpec_Mapper_V5.3.0_Stable_Setup.exe -Algorithm SHA256
```

输出必须为：

```text
DC4E190B6285A5A715D1FF4CE2641BF3365A587BAF0E9F9852115A3428EBB3A2
```

4. 双击安装包，保持“创建桌面快捷方式”选中，然后完成安装。
5. 从桌面或开始菜单启动 `CoreSpec Mapper V5.3`。

默认安装目录为 `%LOCALAPPDATA%\Programs\CoreSpec Mapper V5.3`。安装内容包括桌面/开始菜单快捷方式、卸载程序、CPU 运行时、光谱数据库、矿物 Catalog、智能掩膜模型、技术文档和用户指南。升级安装不会删除用户项目和识别成果；卸载可使用 Windows“已安装的应用”或开始菜单卸载项。

### 源码开发安装

仅开发源码时要求 Python 3.10 或更高版本：

```powershell
git clone https://github.com/TujinoO/CoreSpec_Mapper.git
cd CoreSpec_Mapper
git lfs pull
python -m pip install -e ".[desktop,test]"
```

只使用源码目录时：

```powershell
$env:PYTHONPATH = "src"
python -m corespec_mapper --help
```

## 启动桌面端

```powershell
corespec-desktop
```

等价命令：

```powershell
corespec desktop
```

V4 兼容桌面：

```powershell
corespec-desktop-v4
corespec desktop-v4
```

## V5 配置

仓库提供 `configs/v5_default.json`、`configs/v5_3dssz_validation.json` 和 `configs/v5_zkh3_validation.json`。最小结构如下，不含任何光谱库路径：

```json
{
  "schema_version": 2,
  "application_version": "5.3.0",
  "project": {
    "name": "CoreSpec_V5_Project",
    "non_destructive": true
  },
  "inputs": {
    "analysis_image": "D:/Project/SWIR.dat",
    "analysis_domain": "swir",
    "data_physics": "reflectance",
    "input_is_smoothed": false,
    "rgb": null,
    "nir": null,
    "swir": "D:/Project/SWIR.dat"
  },
  "mask": {
    "mode": "automatic",
    "engine": "auto",
    "path": null,
    "minimum_component_pixels": 64,
    "minimum_mask_fraction": 0.05,
    "minimum_interior_pixel_fraction": 0.65,
    "edge_guard_pixels": 2,
    "approval": {
      "required": true,
      "approved": false
    }
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
  }
}
```

若使用人工掩膜，把 `mask.mode` 改为 `external` 并填写与主立方体同网格的 `mask.path`。完成预览检查后再将 `mask.approval.approved` 设为 `true`；桌面端可直接勾选批准。

NC-1 全景验证配置为 `configs/nc1_v5_2_validation.json`。它保留通用自适应引擎，同时用 `mode.engine=validated_v3` 明确选择项目验证配方。

## CLI

先设置仓库外的统一结果根目录：

```powershell
$CoreSpecRunRoot = "D:\CoreSpec_Mapper_Data\runs"
New-Item -ItemType Directory -Path $CoreSpecRunRoot -Force | Out-Null
```

查看 V5 Catalog、能力和数据库统计：

```powershell
corespec v5-catalog
corespec v5-catalog --output (Join-Path $CoreSpecRunRoot "v5_catalog.json")
```

执行冻结基准比较：

```powershell
corespec v5-validate --config configs/nc1_v5_2_regression.example.json --output (Join-Path $CoreSpecRunRoot "regression.json")
```

审计输入、掩膜、传感器和内置标准谱：

```powershell
corespec v5-audit --config configs/v5_default.json --output (Join-Path $CoreSpecRunRoot "v5_audit.json")
```

导出本次传感器的标准谱选择：

```powershell
corespec v5-library --config configs/v5_default.json --output (Join-Path $CoreSpecRunRoot "v5_library.json")
```

完整运行：

```powershell
corespec v5-run `
  --config configs/v5_default.json `
  --output-root $CoreSpecRunRoot `
  --run-id core_v5_run
```

也可用 `--start-line` 和 `--stop-line` 裁剪写出行范围；当前组级 SAM 与阈值标定仍扫描全幅，因此这不是高效 ROI 加速模式。

V3/V4 命令继续保留，便于历史复算：

```powershell
corespec v4-audit --config configs/nc1_v4.json --output (Join-Path $CoreSpecRunRoot "v4_audit.json")
corespec v4-run --config configs/nc1_v4.json --output-root $CoreSpecRunRoot --run-id legacy_v4
corespec v3-pilot --config configs/nc1_v3.json --output (Join-Path $CoreSpecRunRoot "v3_pilot")
```

## 标准输出

每次运行创建：

`<output_root>/<project_name>/<run_id>/`

主要内容：

| 路径 | 内容 |
|---|---|
| `project.csmproj` | 项目、运行和主数据身份 |
| `config.resolved.json` | 用户配置与解析后的 Runtime 配置 |
| `audit.json` | 输入、掩膜、能力、标准谱和风险 |
| `capability/` | 能力卡、材料掩膜、样本计划和矿物可观测性 |
| `library_ensemble/` | 入选标准谱、候选/拒绝审计和 ENVI 光谱库 |
| `groups/<group>/` | 三档分类、分数、阈值、特征、置信度、稳定性、拒绝原因和伪影图 |
| `confidence/` | 跨组置信度、稳定性和拒绝原因 |
| `thresholds.json` | 全部候选目标与最终阈值 |
| `previews/` | 三档比较、各组叠加、1600 nm 背景、材料掩膜、条带和 Al-OH 亚型预览 |
| `tables/mineral_counts.csv` | 各档矿物像元计数 |
| `reports/` | 工程质量报告 |
| `run_manifest.json` | 输入指纹、资源哈希、标准谱、阈值、质量、耗时和输出清单 |

优先把 `final_balanced.dat` 作为解释主图，`final_conservative.dat` 检查高可信核心，`final_sensitive.dat` 检查潜在漏识别。零检出会作为警告保留，不会通过矿物配额制造结果。

## 四组实测运行

以下结果直接来自指定运行 Manifest。平衡档计数按“碳酸盐 / 钙硫酸盐 / 白云母—伊利石 / 蒙皂石 / 高岭石双峰族”列出，是各组计数而非跨组去重像元。

| 运行 | 主立方体 | 掩膜 | 平衡档组计数 | 总耗时 | 状态 |
|---|---|---|---|---:|---|
| `validation_runs/V5_Final_Raw_Validation/3dssz_raw_v5_final` | 5663×320×212 | 外部，12.2938% | 1495 / 303 / 587 / 0 / 0 | 293.375 s | B / Warning |
| `validation_runs/V5_Final_Raw_Validation/zkh3_raw_v5_final` | 4341×320×212 | 自动，37.5671% | 802 / 3464 / 213 / 0 / 0 | 243.000 s | B / Warning |
| `validation_runs/V5_Final_Validation/3dssz_lowres_v5_final2` | 293×160×367 | 自动，19.8976% | 0 / 220 / 3 / 0 / 0 | 12.484 s | B / Warning |
| `validation_runs/V5_Final_Validation/zkh3_lowres_v5_final2` | 297×158×367 | 自动，47.5387% | 223 / 528 / 221 / 0 / 0 | 14.000 s | B / Warning |

四组均完成 312 个 Manifest 输出项和 24 个预览资产（23 张 PNG + 1 个 `legend.json`），材料掩膜与 Al-OH 亚型 PNG 均存在；同时均报告 FWHM 缺失和无像元级真值。这些运行证明端到端工程链路可完成，不证明矿物准确率。

最终 3DSSZ 原始外部掩膜保留 159 个连通域，ZKH3 原始自动掩膜保留 263 个连通域，均触发碎片化警告；正式地质解释前仍需复核掩膜。

## 测试

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

桌面测试需要 PySide6；无显示环境可设置：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q
```

## 代码结构

```text
configs/                         V3/V4/V5 配置
docs/                            V5 设计、数据库和使用迁移文档
scripts/                         数据库构建与验证分析脚本
spec_lib/                        27 个原始 ENVI 光谱库资产
src/corespec_mapper/
  desktop_v5.py                  V5.3 九步状态化桌面
  foreground_model.py            应用内置智能岩心前景模型
  v5_threshold_trial.py          场景阈值联合试算与复核值生成
  v5_service.py                  V5 审计、运行与 Manifest
  masking.py                     材料掩膜
  spectral_v5.py                 来源、名称、纯度解析
  spectral_db.py                 私有 SQLite 数据库
  v5_library.py                  传感器适配标准谱优选
  v5_calibration.py              自适应阈值
  resources/                     Catalog、数据库、图标
tests/                           单元、CLI、桌面、数据库和服务测试
validation_runs/                 四组 V5 实测验收成果
```

## 文档

- [V5.3 GitHub 下载、安装与升级指南](docs/INSTALL_V5.3.md)
- [V5.3 稳定版技术方法与系统设计](docs/CoreSpec_Mapper_V5_3_稳定版技术方法与系统设计_2026-07-21.md)
- [V5.3 稳定版用户使用指南](docs/CoreSpec_Mapper_V5_3_稳定版用户使用指南_2026-07-21.md)
- [V5 软件总体设计方案](docs/CoreSpec_Mapper_V5_自适应矿物证据引擎_软件总体设计方案_2026-07-20.md)
- [V5 私有光谱数据库清单与识别能力](docs/CoreSpec_Mapper_V5_私有光谱数据库清单与识别能力_2026-07-20.md)
- [V5 桌面版使用与 V4 迁移指南](docs/CoreSpec_Mapper_V5_桌面版使用与V4迁移指南_2026-07-20.md)
- [V4 桌面版使用说明](docs/CoreSpec_Mapper_V4_桌面版使用说明指南_2026-07-19.md)

## 已知限制与后续接口

- RGB/NIR/SWIR 伴随数据当前只做来源审计；自动配准与像元融合未实现。
- Quick Calibration 控件因缺少像元真值而禁用并强制为 false；监督式 ROI/点位标定尚未接入。
- 三档输出固定启用；`mode.profile` 仅控制结果页默认查看偏好，不改变科学运行。
- `--start-line/--stop-line` 只裁剪输出，当前不会避免全幅组级 SAM 与阈值标定。
- 用户 overlay 数据库协议已定义，但 5.3.0 尚未合并外部 overlay。
- VNIR Fe 族专家和 TIR 发射率专家未实现。
- 27 个源库的许可状态当前均为 `unverified`，对外分发数据库前必须完成许可核查。
- 没有矿物学真值时，不应把像元数、质量等级或空间聚集解释为准确率。
