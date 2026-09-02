# CoreSpec Mapper V5.2 桌面版使用与 V4 迁移指南

- 适用版本：5.2.0
- 日期：2026-07-21
- 默认桌面快捷方式：`CoreSpec Mapper V5.2.lnk`
- 默认工作目录：`E:/Code/CoreSpec_Mapper`

## 1. 使用结论

V5.2 的正确使用顺序是：先确认主数据，再得到合格且人工批准的岩心掩膜，然后选择矿物和执行引擎，完成阈值试算，最后运行与复核结果。掩膜未批准、GeoCore 模型包不完整或项目冻结配方缺少基准时，软件会明确阻断或降级，不会静默继续。

对 NC-1 应选择 `validated_v3` 项目验证配方；对没有历史人工基准的新项目，使用 `adaptive_v5` 并把结果视为待真值验证的矿物证据。

## 2. 启动方式

### 2.1 桌面快捷方式

双击桌面的 `CoreSpec Mapper V5.2`。当前快捷方式目标为：

`D:/Users/anaconda/Scripts/corespec-desktop.exe`

工作目录为 `E:/Code/CoreSpec_Mapper`，图标来自软件资源目录。窗口标题应显示 `CoreSpec Mapper V5.2`；若仍显示旧版本，先退出所有旧窗口后重新启动快捷方式。

### 2.2 命令行启动

```powershell
cd E:\Code\CoreSpec_Mapper
corespec-desktop
```

等价命令：

```powershell
corespec desktop
```

V4 兼容桌面仍可通过以下命令启动，但新项目不建议继续使用：

```powershell
corespec-desktop-v4
corespec desktop-v4
```

## 3. 开始前的资料准备

| 资料 | 必需性 | 要求 |
|---|---:|---|
| 主分析 ENVI 立方体 | 必需 | `.dat`/无扩展名数据体与 `.hdr` 配套，波长和尺寸可读 |
| 同网格人工掩膜 | 推荐 | 与主分析影像行列完全一致；NC-1 生产路线优先使用 |
| RGB/NIR/SWIR 伴随数据 | 可选 | 用于来源记录；未经配准不会自动参与逐像元融合 |
| GeoCore M1-2 模型包 | 自动掩膜需要 | 注册表、Manifest、权重和配准参数完整 |
| 项目冻结配置与历史结果 | `validated_v3` 必需 | 参考谱/阈值配方、基准掩膜和基准分类路径有效 |
| 独立矿物真值 | 科学验收推荐 | XRD、拉曼、薄片或点光谱，且能与深度/像元对应 |

可选 NIR 若只有头文件而缺数据体，V5.2 会给出结构化警告；只要主分析 SWIR 有效，不会中断项目。

## 4. 八步桌面流程

### 4.1 项目

填写项目名称、说明和输出根目录。每次运行创建独立 `run_id`，不会覆盖历史成果。项目名应包含钻孔或样品批次，说明中记录数据版本、掩膜来源和真值状态。

### 4.2 数据

选择唯一的主分析立方体。主数据决定最终网格和矿物识别波段；RGB、NIR、SWIR 伴随路径仅作为可选来源。检查界面显示的行、列、波段、波长范围、数据物理和可读状态。

若输入已完成 SG 平滑，确认配置为 `input_is_smoothed=true`；否则程序会再次平滑并改变结果。

### 4.3 掩膜

可选三类路线：

1. GeoCore M1-2 自动掩膜：只有模型部署检查通过时使用。
2. 外部同网格掩膜：已有人工实验成果时优先，NC-1 使用此路线。
3. 光谱回退掩膜：仅用于没有模型与人工掩膜的探索性场景。

生成或加载后必须查看预览，并检查：

- 覆盖率是否合理；默认不得低于 5%；
- 岩心柱体内部是否被填满，内部填充率默认至少 65%；
- 是否只剩托盘外轮廓、裂缝、边缘或反光点；
- 是否漏掉暗色、破碎或湿润岩心；
- 组件数和碎片化是否符合实际岩心盒布局。

确认后填写/保留审核人、时间和备注，再勾选批准。未批准时运行按钮和服务端执行都会阻断。

当前 GeoCore M1-2 适配器已接入，但模型注册表、Manifest 和权重尚未齐备。界面应显示真实部署状态；不要把光谱回退结果误认为神经网络模型结果。

### 4.4 矿物

默认勾选 7 个已验证 SWIR 目标：方解石、白云石、硬石膏、石膏、伊利石、蒙脱石和高岭石。

17 个实验级目标可手动勾选，但必须专项验收。赤铁矿、针铁矿和石英因当前数据物理不满足保持关闭。开启“内部混淆矿物”有助于相似谱形竞争，不代表所有混淆矿物都成为最终输出类别。

### 4.5 模式

| 引擎 | 选择条件 | 结果解释 |
|---|---|---|
| `adaptive_v5` | 新项目，没有冻结人工基准 | Catalog 有界的无标签代理阈值；用于发现候选并等待真值 |
| `validated_v3` | 已有可靠历史人工实验和冻结配方 | 恢复项目基准并做逐像元回归；NC-1 使用 |

模式选择会改变参数来源。不要在 NC-1 回归时误选自适应引擎，也不要把 NC-1 冻结参数直接迁移到其他钻孔后声称可靠。

### 4.6 阈值试算

`adaptive_v5` 显示 Catalog 安全候选范围和场景代理目标；完整运行后保存最终候选、分量、拒绝原因和并列选择。`validated_v3` 显示项目冻结参数及其来源。

使用“导出 JSON”保存本次试算。检查阈值是否位于安全包络、是否出现过覆盖或零候选、宽松阈值是否主要增加固定列/边缘/孤立斑块。

### 4.7 运行

运行前系统再次检查：主数据可读、掩膜质量、人工批准、矿物能力、执行引擎和项目配方。运行页显示阶段、进度、已用时间、非递增 ETA 和结构化告警。

取消操作只停止当前运行，不删除已经产生的历史运行目录。遇到错误时保留目录与日志，便于复核。

### 4.8 结果

优先查看 Balanced，再用 Sensitive 做召回复核、Conservative 做高置信候选复核。使用适合窗口、50%、1:1、200% 缩放和滚动平移检查：

- 分类是否严格位于岩心掩膜内；
- 是否形成固定探测器列、平行细条或盒体边缘；
- 相似矿物边界是否突变；
- 三档是否满足严格包含关系；
- 质量等级、`publishable`、回归结果和警告是否一致。

运行完成不等于可发布。C/Warning 或 `publishable=false` 必须按风险处理，不能只看“完成”字样。

## 5. NC-1 推荐复跑

### 5.1 桌面端

1. 加载 `configs/nc1_v5_2_validation.json`。
2. 确认主分析数据为 `F:/NC-1-31_40/FILL/SWIR_SG`。
3. 确认外部掩膜为 `F:/NC-1-31_40/FILL/mask_SG`，并核对批准记录。
4. 确认 7 个默认目标全部选中。
5. 确认执行引擎为 `validated_v3`。
6. 复核历史结果根目录和两个档位配置均可读。
7. 运行后打开 `validation_report.json`，确认 14 项比较全部通过。

### 5.2 命令行

```powershell
corespec v5-run `
  --config configs/nc1_v5_2_validation.json `
  --output-root E:\Code\CoreSpec_Mapper\.codex_runs `
  --run-id nc1_v52_repeat
```

单独执行通用回归门前，把示例中的 `CANDIDATE_RUN` 替换为实际运行目录：

```powershell
corespec v5-validate `
  --config configs/nc1_v5_2_regression.example.json `
  --output output\nc1_v52_regression.json
```

## 6. 新项目最小配置

```json
{
  "schema_version": 2,
  "application_version": "5.2.0",
  "project": {
    "name": "New_Core_Project",
    "non_destructive": true
  },
  "inputs": {
    "analysis_image": "D:/Project/SWIR.dat",
    "analysis_domain": "swir",
    "data_physics": "reflectance",
    "input_is_smoothed": false
  },
  "mask": {
    "mode": "automatic",
    "engine": "auto",
    "minimum_mask_fraction": 0.05,
    "minimum_interior_pixel_fraction": 0.65,
    "approval": {
      "required": true,
      "approved": false
    }
  },
  "minerals": {
    "requested": [
      "calcite", "dolomite", "anhydrite", "gypsum",
      "illite", "montmorillonite", "kaolinite"
    ],
    "include_internal_confusers": true
  },
  "mode": {
    "engine": "adaptive_v5",
    "profile": "balanced",
    "write_all_profiles": true
  }
}
```

首次自动生成掩膜后，必须在桌面端查看预览并批准，再开始完整矿物运行。

## 7. 常用 CLI

```powershell
# 查看 Catalog 与数据库统计
corespec v5-catalog

# 只审计输入、掩膜、传感器和参考谱
corespec v5-audit --config configs/v5_default.json --output output\v5_audit.json

# 导出传感器适配参考谱
corespec v5-library --config configs/v5_default.json --output output\v5_library.json

# 完整自适应运行
corespec v5-run --config configs/v5_default.json --output-root output --run-id project_run
```

`--start-line/--stop-line` 只裁剪最终写出行范围；组级 SAM 与阈值标定当前仍扫描全幅，因此不能把它当作真正的 ROI 加速模式。

## 8. 输出目录检查

| 文件/目录 | 复核重点 |
|---|---|
| `config.resolved.json` | 实际输入、掩膜、矿物、引擎和参数 |
| `capability/` | 材料掩膜、能力卡、掩膜预览和质量指标 |
| `library_ensemble/` | 参考谱、来源、样品和选择原因 |
| `thresholds.json` | 自适应候选或冻结阈值及来源 |
| `validated_profiles/` / `groups/` | 各档分类、中间证据和预览 |
| `rejection_waterfall.json` | 各证据层拒绝数量与最终分类数量 |
| `validation_report.json` | 逐矿物 IoU、Precision、Recall、面积差与失败项 |
| `quality_report.json` | 工程质量等级与发布状态 |
| `run_manifest.json` | 输入、资源哈希、输出清单、耗时和完整审计链 |

## 9. V4 到 V5.2 字段迁移

| V4 概念/字段 | V5.2 对应 | 迁移注意 |
|---|---|---|
| `analysis_image` | `inputs.analysis_image` | 唯一像元级主网格 |
| `analysis_mask` | `mask.mode=external` + `mask.path` | 增加覆盖、填充和人工批准 |
| 外部光谱库根目录 | 删除 | V5.2 自动使用内置只读数据库 |
| 固定 Catalog 阈值 | `adaptive_v5` 安全包络或 `validated_v3` 冻结参数 | 必须明确参数来源 |
| 单一运行页面 | 八步状态流程 | 掩膜与阈值有独立复核步骤 |
| “运行结束” | `status` + `quality_grade` + `publishable` | 完成状态不再等同可发布 |
| 单一分类结果 | 三档共享证据 + QA/瀑布 | Balanced 默认，另两档用于复核 |
| 旧项目结果 | `validated_v3` + `v5-validate` | 建立逐矿物回归门，不用目测代替 |

V4 配置迁移后先运行 `v5-audit`，不要直接完整运行。现有可靠人工掩膜必须保留，不能因为 V5.2 支持自动掩膜就强制替换。

## 10. 常见问题

### 10.1 自动掩膜只剩轮廓或裂缝

查看覆盖率和内部填充率。低覆盖或低填充会阻断；优先检查 GeoCore 模型包状态。模型不可用时使用已批准同网格人工掩膜，不要盲目降低 5% 覆盖门。

### 10.2 GeoCore M1-2 显示不可用

当前适配器要求模型注册表、Manifest、权重及其依赖完整。缺任何一项都应明确显示部署不完整。补齐后还需验证 RGB→SWIR 配准，不能只验证模型能启动。

### 10.3 某矿物零检出

依次检查传感器窗口、参考谱数量、阈值候选、拒绝瀑布、吸收深度和空间清理。零检出本身不是软件错误；禁止设置最低配额制造阳性像元。

### 10.4 结果出现固定直列或细条

查看 `column_candidates`、`stripe_noise` 和拒绝瀑布。必要时回到数据与掩膜步骤检查坏波段、边缘和 SG 处理，不应只放宽分类阈值。

### 10.5 为什么 NC-1 要用冻结配方

因为已有人工实验结果可作为流程弱标签。先恢复基准能确认软件执行正确，再单独评价自适应技术路线，避免把掩膜错误、参考谱变化和阈值变化混在一起。

### 10.6 IoU = 1 是否代表矿物识别 100% 正确

不代表。它只表示 V5.2 与冻结历史分类逐像元一致。历史分类仍需要 XRD、拉曼、薄片或点光谱独立验证。

## 11. 发布前检查表

- 主立方体和数据物理正确；没有重复 SG 平滑。
- 掩膜覆盖、内部填充、组件和预览合理，且已人工批准。
- 矿物支持级别与传感器窗口匹配。
- 执行引擎和参数来源符合项目目的。
- 阈值候选或冻结参数有 JSON 记录。
- 结果位于掩膜内，三档关系正确，无明显固定列或边缘伪影。
- 质量等级、`publishable`、警告和回归门全部复核。
- 对外结论明确区分工程回归、弱标签与独立矿物真值。
