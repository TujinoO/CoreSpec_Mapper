# CoreSpec Mapper V4 桌面版使用说明指南

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 软件名称 | CoreSpec Mapper |
| 桌面版本 | V4.0.0 |
| 算法名称 | Adaptive Mineral Evidence Engine |
| 文档日期 | 2026-07-19 |
| 适用对象 | 岩心高光谱处理人员、地质解释人员、项目验收人员、算法复核人员 |
| 适用数据 | SWIR 反射率岩心高光谱 ENVI 影像及同网格掩膜 |
| 代码目录 | `E:\Code\CoreSpec_Mapper` |
| 推荐结果目录 | `E:\Code\CoreSpec_Mapper\output\<project_name>\<run_id>` |
| NC-1 最终验收运行 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final` |

CoreSpec Mapper V4 桌面版是一个离线 Windows 桌面程序，用于对岩心 SWIR 高光谱影像进行矿物证据提取、矿物组分类、相似矿物竞争、条带伪影控制、三档结果导出和质量验收。桌面端由 PySide6 构建，底层调用与命令行相同的 V4 服务层，因此桌面运行、CLI 运行和自动化测试使用的是同一套后端算法。

本指南面向实际操作人员，重点说明如何准备输入、启动软件、完成审计、运行 V4、查看和解释结果，以及遇到常见问题时如何处理。

## 2. 软件能做什么

CoreSpec Mapper V4 的核心能力包括：

1. 读取 ENVI 高光谱影像、ENVI 掩膜影像和 ENVI 光谱库。
2. 根据实际波长、波段数量、FWHM、数据物理和掩膜质量生成传感器能力卡。
3. 根据 Mineral Evidence Catalog 判断当前数据是否支持目标矿物识别。
4. 从 `spec_lib` 光谱库中自动筛选、重采样并集成代表性参考谱。
5. 在全深度上分层抽样，自动估计场景噪声、列偏差和逐列阈值。
6. 对碳酸盐、硫酸盐和黏土三组 SWIR 矿物计算公共光谱证据。
7. 对组内相似矿物进行重新竞争，不由第一步 SAM 直接锁死子类。
8. 输出 Conservative、Balanced、Sensitive 三档嵌套分类结果。
9. 生成置信度、稳定性、拒绝原因、条带掩膜、列风险、吸收深度和 SFF 质量等审计产品。
10. 自动生成 PNG 预览、CSV 统计表、JSON/Markdown/HTML 质量报告和运行清单。
11. 支持桌面端配置保存、配置重新打开、历史运行结果重开、进度显示和安全取消。

当前 V4.0.0 首版重点支持 SWIR 反射率岩心影像。VNIR 与 TIR 专家在 Catalog 中已有注册入口，但尚未作为生产分类能力开放。软件不会在缺少波段能力或专家实现的情况下强制输出赤铁矿、石英等当前不支持的矿物。

## 3. 当前支持的目标矿物

V4 当前通过完整 NC-1 影像验收的 SWIR 矿物共有 7 类：

| 矿物组 | 支持矿物 | 主要说明 |
|---|---|---|
| 碳酸盐 `carbonates` | Calcite 方解石、Dolomite 白云石 | 主要利用约 2.3 μm 碳酸根相关吸收特征 |
| 硫酸盐 `sulfates` | Anhydrite 硬石膏、Gypsum 石膏 | 主要利用水合、硫酸盐和相关组合吸收特征 |
| 黏土 `clays` | Illite 伊利石、Montmorillonite 蒙脱石、Kaolinite 高岭石 | 主要利用 Al-OH、约 2.16/2.20 μm 双吸收和形态特征 |

白云母、叶蜡石、绿泥石、明矾石等 SWIR 矿物具备后续扩展入口，但在没有完成项目验证前，应谨慎作为生产成果使用。赤铁矿、石英等矿物受 VNIR/TIR 专家和波段条件约束，不应在当前 SWIR 反射率流程中强制运行。

## 4. 使用前准备

### 4.1 硬件与系统建议

建议使用 Windows 工作站或笔记本，至少满足：

| 项目 | 建议 |
|---|---|
| 操作系统 | Windows 10/11 |
| 内存 | 16 GB 及以上，完整影像建议 32 GB |
| 磁盘 | 保留足够结果空间，完整运行会写出多组 ENVI、PNG、JSON 和报告文件 |
| Python | Python 3.10 及以上，项目当前以 Python 包方式运行 |
| 桌面依赖 | PySide6 |

软件本身不依赖 ENVI 运行环境。ENVI 主要用于后续人工查看 `.dat/.hdr` 分类结果。

### 4.2 输入文件清单

一次 V4 桌面运行至少需要 4 类输入：

| 输入 | 示例 | 要求 |
|---|---|---|
| 高光谱影像 | `F:\NC-1-31_40\FILL\SWIR_SG` | ENVI 数据文件与同名或匹配 `.hdr` 必须存在 |
| 岩心掩膜 | `F:\NC-1-31_40\FILL\mask_SG` | 与高光谱影像行列一致，可为掩膜立方体或二值栅格 |
| 光谱库目录 | `E:\Code\CoreSpec_Mapper\spec_lib` | 包含 USGS、JPL、JHU、IGCP 等 ENVI 光谱库 |
| 输出根目录 | `E:\Code\CoreSpec_Mapper\output` | 程序将在其中自动创建 `<project_name>\<run_id>` |

输入影像和掩膜的 ENVI 头文件应包含行数、列数、波段数、数据类型、交错方式、字节序等字段。高光谱影像必须包含波长向量。若 HDR 中缺少 FWHM，软件仍可运行，但质量报告会记录 `FWHM_UNKNOWN` 警告。

### 4.3 关于掩膜

掩膜用于限定岩心有效区域。掩膜外像元在最终分类中会写为 `Masked Pixels`，与 `Unclassified` 不同：

| 类别 | 含义 |
|---|---|
| `Masked Pixels` | 不属于分析区域，例如托盘、背景、无效区 |
| `Unclassified` | 属于岩心有效区域，但矿物证据不足或分类不稳定 |

掩膜必须与影像空间尺寸一致。若行列不一致，审计阶段会失败。若有效掩膜像元过少，软件会阻止生产运行。

### 4.4 已平滑影像与 SG 参数

桌面端有 `Input is already SG-smoothed` 选项：

| 情况 | 选择方式 |
|---|---|
| 输入已经是 SG 平滑后的影像，例如 NC-1 的 `SWIR_SG` | 勾选 |
| 输入是原始反射率影像，未做 SG 平滑 | 取消勾选，并设置 SG window / SG polynomial |

NC-1 验收配置使用 `SG window = 11`、`SG polynomial = 2`，且输入已平滑，因此默认勾选该选项。

## 5. 安装与启动

### 5.1 在开发目录中安装

在 PowerShell 中进入项目目录：

```powershell
cd E:\Code\CoreSpec_Mapper
python -m pip install -e ".[desktop,test]"
```

安装后会生成两个主要命令：

| 命令 | 用途 |
|---|---|
| `corespec` | 命令行入口 |
| `corespec-desktop` | 桌面程序入口 |

### 5.2 启动桌面端

推荐启动方式：

```powershell
corespec-desktop
```

也可以通过 CLI 子命令启动：

```powershell
corespec desktop
```

若尚未安装 editable package，也可以在项目根目录设置 `PYTHONPATH=src` 后运行：

```powershell
$env:PYTHONPATH = "src"
python -m corespec_mapper desktop
```

启动后窗口标题为 `CoreSpec Mapper V4`，默认大小约为 `1480 × 900`。

## 6. 桌面界面总览

桌面主窗口由 4 个主要区域组成：

| 区域 | 内容 |
|---|---|
| 顶部工具栏 | 打开配置、保存配置、打开历史运行、执行审计、执行运行、取消 |
| 左侧步骤导航 | Project、Data、Mask、Minerals、Mode、Review、Run & Results |
| 中央标签页 | Setup、Minerals、Results |
| 底部状态栏 | 当前阶段、总进度条、已用时间、预计剩余时间、取消按钮 |

左侧步骤导航用于快速切换工作阶段。当前桌面 MVP 中，Project/Data/Mask/Mode 主要对应 `Setup` 页，Minerals 对应 `Minerals` 页，Review/Run & Results 对应 `Results` 页。

### 6.1 顶部工具栏按钮

| 按钮 | 功能 |
|---|---|
| Open | 打开已有 JSON 配置 |
| Save | 将当前界面参数保存为 JSON 配置 |
| Open run | 打开已有 V4 运行目录并查看预览 |
| Audit | 仅执行项目审计，不写出完整分类 |
| Run | 执行完整 V4 运行 |
| Cancel | 请求取消当前后台任务 |

取消是安全取消。程序会在合适的块边界停止，关闭文件句柄并保留当前运行清单，不会把不完整成果标记为可发布结果。

### 6.2 Setup 页

`Setup` 页用于填写项目输入和主要参数。

#### Project inputs

| 控件 | 说明 | 推荐填写 |
|---|---|---|
| Project | 项目名称，用于输出目录 | `NC1_SWIR_v1` 或实际钻孔/样段名称 |
| Hyperspectral image | ENVI 高光谱数据或 HDR 路径 | `F:\NC-1-31_40\FILL\SWIR_SG` |
| Core mask | ENVI 掩膜数据或 HDR 路径 | `F:\NC-1-31_40\FILL\mask_SG` |
| Spectral libraries | 光谱库根目录 | `E:\Code\CoreSpec_Mapper\spec_lib` |
| Output root | 输出根目录 | `E:\Code\CoreSpec_Mapper\output` |
| Data physics | 数据物理类型 | SWIR 反射率选 `reflectance` |
| Preprocessing | 是否已 SG 平滑 | 已平滑影像勾选 |

路径输入框右侧的文件夹按钮可浏览选择文件或目录。

#### Advanced automatic settings

高级参数默认折叠，普通用户通常保持默认即可。

| 参数 | 默认值 | 含义 | 调整建议 |
|---|---:|---|---|
| SG window | 11 | SG 平滑窗口 | 必须为奇数。输入未平滑时才真正执行 |
| SG polynomial | 2 | SG 多项式阶数 | 通常 2 或 3 |
| Depth blocks | 12 | 全深度分层样本块数量 | 取值 8-16，影像很长时建议 12 |
| Rows per block | 64 | 每个样本块行数 | 默认 64 |
| Maximum references | 6 | 每个矿物最多代表谱数 | 默认 6，避免参考谱数量偏置 |

### 6.3 Minerals 页

`Minerals` 页显示当前 Catalog 中的矿物支持状态。

表格字段如下：

| 列名 | 含义 |
|---|---|
| Use | 是否参与本次运行 |
| Mineral | 英文名和中文名 |
| Group | 矿物组 |
| Support | 支持状态 |
| Reason | 不支持或有条件支持的原因 |

首次打开软件时，已验证 SWIR 矿物默认勾选。未实现专家或当前数据不支持的矿物会被取消勾选，且不能作为普通生产矿物运行。

建议工作方式：

1. 先在 Setup 页填写输入路径。
2. 点击 `Audit`。
3. 审计完成后进入 Minerals 页。
4. 查看每个矿物的 `Support` 和 `Reason`。
5. 保留需要识别且受支持的矿物。
6. 返回 Setup 页或直接点击 `Run V4`。

### 6.4 Results 页

`Results` 页用于查看完成后的预览、质量等级和运行日志。

| 控件 | 说明 |
|---|---|
| 预览下拉框 | 选择要查看的 PNG 预览 |
| Open run | 选择一个已有 V4 运行目录 |
| Open output | 在资源管理器中打开当前运行目录 |
| 中央预览区 | 显示分类总览、三档对比或条带诊断图 |
| Quality 标签 | 显示质量等级和运行状态 |
| Log | 显示审计/运行进度和关键消息 |

常见预览包括：

| 预览键 | 文件 | 说明 |
|---|---|---|
| `background` | `previews\swir_background_1600nm.png` | SWIR 背景图 |
| `comparison_balanced` | `previews\comparison_balanced.png` | 平衡版三组结果总览 |
| `comparison_conservative` | `previews\comparison_conservative.png` | 严格版三组结果总览 |
| `comparison_sensitive` | `previews\comparison_sensitive.png` | 宽松版三组结果总览 |
| `comparison_three_profiles` | `previews\comparison_three_profiles.png` | 三档综合对比 |
| `stripe_diagnosis` | `previews\stripe_diagnosis.png` | 条带风险与清理诊断 |
| `carbonates_balanced` | `previews\group_final_carbonates_balanced.png` | 碳酸盐平衡版 |
| `sulfates_balanced` | `previews\group_final_sulfates_balanced.png` | 硫酸盐平衡版 |
| `clays_balanced` | `previews\group_final_clays_balanced.png` | 黏土平衡版 |

## 7. 推荐完整操作流程

### 7.1 第一步：启动软件

```powershell
cd E:\Code\CoreSpec_Mapper
corespec-desktop
```

确认主窗口正常显示，底部状态为 `Ready`。

### 7.2 第二步：填写项目输入

在 `Setup` 页填写：

| 字段 | NC-1 示例 |
|---|---|
| Project | `NC1_SWIR_v1` |
| Hyperspectral image | `F:\NC-1-31_40\FILL\SWIR_SG` |
| Core mask | `F:\NC-1-31_40\FILL\mask_SG` |
| Spectral libraries | `E:\Code\CoreSpec_Mapper\spec_lib` |
| Output root | `E:\Code\CoreSpec_Mapper\output` |
| Data physics | `reflectance` |
| Input is already SG-smoothed | 勾选 |

若处理新项目，应将 Project 改为具体项目名，例如 `ZK001_SWIR_120_160m`。项目名会进入输出路径，应避免使用不适合文件路径的特殊字符。

### 7.3 第三步：保存配置

点击工具栏 `Save`，保存为 JSON，例如：

```text
E:\Code\CoreSpec_Mapper\configs\nc1_v4.json
```

配置保存会保留界面没有直接展示的科学参数，例如 `mask_bands`、`artifact_control`、`policy_overrides` 等，便于后续复现。

### 7.4 第四步：执行 Audit

点击 `Audit` 或 `Setup` 页底部的 `Audit` 按钮。

审计会做以下事情：

1. 打开高光谱影像、掩膜和 Catalog。
2. 检查影像和掩膜尺寸是否一致。
3. 读取波长向量、波段数量和数据物理。
4. 根据掩膜波段生成岩心有效掩膜。
5. 构建全深度分层样本块。
6. 估计输入质量、坏波段和传感器能力。
7. 判断每个矿物是否 Supported、Conditional 或 Unsupported。
8. 可选扫描光谱库，统计入选参考谱数量。

Audit 完成后，软件会自动切换到 Minerals 页。此时应重点查看：

| 检查项 | 正常情况 |
|---|---|
| 七种 SWIR 矿物 | `Supported` 或可运行状态 |
| VNIR/TIR 矿物 | 未实现或 Unsupported，不参与生产 |
| Reason | 没有关键错误 |
| Log | 显示已选择参考谱数量 |

若 Audit 报错，应先修正输入路径、掩膜或数据物理，不建议直接跳过审计。

### 7.5 第五步：选择目标矿物

在 Minerals 页勾选需要识别的矿物。生产运行建议从已验证的 7 类 SWIR 矿物开始：

```text
calcite
dolomite
anhydrite
gypsum
illite
montmorillonite
kaolinite
```

若某矿物被软件判定为 Unsupported，不应通过手工改 JSON 强行加入。Unsupported 通常表示缺少关键波段、数据物理不匹配、专家未实现或没有合格参考谱。

### 7.6 第六步：执行完整 Run

点击 `Run` 或 `Run V4`。

完整运行会经历以下阶段：

| 阶段 | 说明 |
|---|---|
| audit | 打开输入、确认能力和掩膜 |
| library | 自动扫描、重采样和筛选参考谱 |
| group_sam | 全深度计算矿物组级 SAM 证据 |
| calibration | 估计场景噪声、逐列阈值和三档策略 |
| classification | 计算连续统、SFF、吸收深度、相似矿物竞争 |
| outputs | 写出 ENVI、JSON、CSV 和中间证据图 |
| preview | 生成 PNG 预览 |
| QA | 重新打开最终 ENVI 文件并执行质量检查 |
| complete | 写入 run manifest，显示质量等级 |

运行过程中可以查看：

| 界面元素 | 用途 |
|---|---|
| 进度条 | 总体进度 |
| Stage 标签 | 当前阶段和当前消息 |
| Elapsed | 已用时间 |
| Remaining | 估计剩余时间 |
| Log | 阶段细节 |

### 7.7 第七步：查看结果

运行完成后，软件进入 Results 页。建议先看：

1. `Quality` 标签：确认质量等级是否为 A 或 B。
2. `comparison_balanced`：查看平衡版三组矿物空间分布。
3. `comparison_sensitive`：查看弱证据和潜在漏识别。
4. `comparison_three_profiles`：检查严格、平衡、宽松是否符合预期扩张。
5. `stripe_diagnosis`：检查固定列和方向性条带是否被控制。
6. `Open output`：打开运行目录，查看 ENVI 和报告文件。

NC-1 最终运行的质量等级为 `B`，状态为 `Warning`，但 `publishable` 为 `true`。警告主要来自缺少 FWHM 和无矿物学真值，不是固定列残余或输出损坏。

## 8. 输出目录详解

每次完整运行都会创建独立目录：

```text
<output_root>/<project_name>/<run_id>/
```

例如：

```text
E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final
```

目录结构如下：

```text
project.csmproj
config.resolved.json
audit.json
summary.json
run_manifest.json
capability/
library_ensemble/
groups/
confidence/
previews/
reports/
tables/
```

### 8.1 根目录文件

| 文件 | 说明 |
|---|---|
| `project.csmproj` | CoreSpec Mapper 项目记录，包含项目名、运行 ID、输入和输出根目录 |
| `config.resolved.json` | 本次运行实际使用的完整配置 |
| `audit.json` | 运行前输入审计和能力审计结果 |
| `summary.json` | 科学计算摘要、参数、各组统计、三档计数和清理记录 |
| `run_manifest.json` | 最重要的总清单，包含版本、输入指纹、预览、质量、输出文件清单和关键哈希 |

其中 `run_manifest.json` 是结果追溯和验收首选文件。它记录输入绝对路径、文件大小、修改时间、HDR 哈希、软件版本、算法名称、目标矿物、预览路径、质量等级和重要输出哈希。

### 8.2 capability 目录

| 文件 | 说明 |
|---|---|
| `sensor_capability_card.json` | 传感器能力卡，包括波长范围、波段域、FWHM、SNR、坏波段、数据物理 |
| `mineral_observability.json` | 每个矿物的可识别性记录 |
| `mineral_observability.csv` | 便于表格查看的矿物支持度 |
| `sample_plan.json` | 全深度分层样本块计划 |
| `catalog_summary.json` | 本次使用的 Catalog 摘要 |
| `column_spectral_bias.dat/.hdr` | 固定探测器列光谱偏差估计 |

若新数据运行异常，首先查看这里的能力卡和矿物支持度。

### 8.3 library_ensemble 目录

| 文件 | 说明 |
|---|---|
| `automatic_library_ensemble.sli/.hdr` | 本次自动集成后的参考光谱库 |
| `selection_manifest.json` | 每条候选谱的入选、拒绝、评分和来源记录 |
| `selection_manifest.csv` | 表格版选谱审计清单 |

该目录用于回答“本次分类到底用了哪些标准谱”。若结果与地质预期差异较大，应检查参考谱是否来自合适的矿物、是否存在混合名、是否有足够代表谱。

### 8.4 groups 目录

`groups` 下按矿物组分目录：

```text
groups\carbonates\
groups\sulfates\
groups\clays\
```

每个组都包含最终分类和中间证据产品。

#### 最终分类文件

| 文件 | 用途 |
|---|---|
| `final_conservative.dat/.hdr` | 严格版，高可信、低召回 |
| `final_balanced.dat/.hdr` | 平衡版，推荐作为主解释图 |
| `final_sensitive.dat/.hdr` | 宽松版，高召回，用于漏识别检查 |

默认在 ENVI 或其他 GIS/遥感软件中优先打开 `final_balanced.dat`。

#### 主要审计文件

| 文件 | 说明 |
|---|---|
| `group_sam_score.dat` | 组级最佳 SAM 角，单位 rad，越小越相似 |
| `raw_group_sam_score.dat` | 域校准前组级 SAM |
| `group_mineral_sam_scores.dat` | 各矿物层面的 SAM 证据 |
| `column_threshold.dat` | 三档逐列 SAM 阈值 |
| `column_candidates.dat` | 平衡版组级候选 |
| `raw_mineral_scores.dat` | 原始共享矿物分数，越低越好 |
| `calibrated_mineral_scores.dat` | 域校准后的矿物分数，越低越好 |
| `classes_before_artifact_filter.dat` | 伪影清理前平衡版分类 |
| `stripe_noise_mask.dat` | 被判为方向性条带或弱证据伪影的像元 |
| `column_risk_score.dat` | 固定探测器列风险 |
| `edge_risk_score.dat` | 岩心边界风险 |
| `saturation_mask.dat` | 高反射或饱和风险 |
| `absorption_depth.dat` | 连续统吸收深度 |
| `classification_margin.dat` | 第一名与第二名矿物分数间隔 |
| `sff_quality.dat` | SFF 拟合质量 |
| `confidence.dat` | 组内置信度，范围 0-1 |
| `stability.dat` | 标签、边际和参考谱共识稳定性 |
| `rejection_reason.dat` | 平衡版拒绝原因编码 |
| `summary.json` | 该矿物组的参数、统计、计数和清理记录 |

### 8.5 confidence 目录

| 文件 | 说明 |
|---|---|
| `confidence.dat/.hdr` | 跨矿物组最大置信度 |
| `stability.dat/.hdr` | 最大置信度对应的稳定性 |
| `rejection_reason.dat/.hdr` | 最大置信度对应的平衡版拒绝原因 |

这些文件用于整体质量评估。若某深度段分类很多但置信度低，应降低解释权重，并结合人工复核。

### 8.6 previews 目录

| 文件 | 说明 |
|---|---|
| `swir_background_1600nm.png` | SWIR 背景图 |
| `comparison_balanced.png` | 平衡版总览 |
| `comparison_conservative.png` | 严格版总览 |
| `comparison_sensitive.png` | 宽松版总览 |
| `comparison_three_profiles.png` | 三档对比总览 |
| `stripe_diagnosis.png` | 条带和伪影诊断 |
| `group_final_<group>_<policy>.png` | 单组单档预览 |
| `legend.json` | 预览颜色图例 |

桌面 Results 页读取的就是这些预览文件。

### 8.7 reports 目录

| 文件 | 说明 |
|---|---|
| `quality_report.json` | 结构化质量报告 |
| `quality_report.md` | Markdown 质量报告 |
| `quality_report.html` | 浏览器可读质量报告 |

推荐验收和汇报时打开 `quality_report.html`。若需要自动化解析，则读取 `quality_report.json`。

### 8.8 tables 目录

| 文件 | 说明 |
|---|---|
| `mineral_counts.csv` | 每个策略、矿物组和矿物的像元计数 |

注意：像元数不是矿物含量，不能直接解释为真实丰度。不同矿物组可以在同一像元形成独立证据，因此三组计数合计不等于空间唯一矿物像元数。

## 9. 三档结果如何使用

V4 一次运行会输出三档结果。三档结果来自同一套公共证据，不是三次互相独立的算法。

| 档位 | 英文 | 特点 | 推荐用途 |
|---|---|---|---|
| 严格版 | `conservative` | 阈值更严、结果更少、置信要求高 | 高可信矿物脉体或重点区域提取 |
| 平衡版 | `balanced` | 兼顾可靠性和召回率 | 默认主解释图、生产交付主图 |
| 宽松版 | `sensitive` | 阈值更宽、结果更多、弱证据更多 | 检查潜在漏识别、弱蚀变、脉体外围 |

推荐解释顺序：

1. 先看 `final_balanced.dat`，作为主图。
2. 再看 `final_sensitive.dat`，识别可能漏掉的弱信号。
3. 用 `final_conservative.dat` 检查最高可信核心位置。
4. 对三档差异很大的位置，查看 `confidence.dat`、`stability.dat` 和 `rejection_reason.dat`。
5. 对重要深度段进行 XRD、拉曼、薄片或点位光谱复核。

三档结果应满足嵌套关系：严格版是平衡版子集，平衡版是宽松版子集。QA 会自动检查该关系。

## 10. 如何在 ENVI 中查看结果

### 10.1 推荐打开文件

在 ENVI 中优先打开每组的平衡版：

```text
groups\carbonates\final_balanced.dat
groups\sulfates\final_balanced.dat
groups\clays\final_balanced.dat
```

需要检查弱蚀变时打开：

```text
groups\carbonates\final_sensitive.dat
groups\sulfates\final_sensitive.dat
groups\clays\final_sensitive.dat
```

需要高可信核心区域时打开：

```text
groups\carbonates\final_conservative.dat
groups\sulfates\final_conservative.dat
groups\clays\final_conservative.dat
```

### 10.2 ENVI 分类头

最终分类文件是 ENVI Classification。HDR 中包含：

| 类别 | 含义 |
|---|---|
| `Unclassified` | 有效岩心像元，但未通过该组矿物分类 |
| 组内矿物类别 | 例如 Calcite、Dolomite、Gypsum 等 |
| `Masked Pixels` | 掩膜外区域 |

颜色表已写入 HDR。若 ENVI 没有自动显示颜色，可检查 `class names`、`classes` 和 `class lookup` 字段是否被正确读取。

### 10.3 与原始影像叠加

建议将最终分类结果叠加在 SWIR 背景或原始岩心影像上：

1. 打开原始 SWIR 影像或背景波段。
2. 打开 `final_balanced.dat`。
3. 使用透明度叠加查看矿物空间位置。
4. 对可疑竖线或边缘响应，打开 `stripe_noise_mask.dat`、`column_risk_score.dat` 和 `edge_risk_score.dat` 复核。

## 11. 质量报告解读

V4 质量等级分为 A、B、C、D：

| 等级 | 含义 | 建议 |
|---|---|---|
| A | 输入完整、能力充分、稳定性高、伪影风险低 | 可作为高可信软件结果，仍需区分矿物学真值 |
| B | 主要检查通过，存在轻度警告 | 可用于生产解释，建议抽查 |
| C | 证据有限、参数敏感或伪影残余明显 | 仅作候选和复核，不建议直接交付 |
| D | 缺少关键能力或输出不完整 | 阻止发布 |

NC-1 最终验收结果为：

| 项目 | 结果 |
|---|---|
| 质量等级 | `B` |
| 状态 | `Warning` |
| 自动发布门禁 | Passed |
| 发布建议 | 可发布，但需注明 FWHM 与无真值限制 |
| 平衡版七类矿物 | 全部非零 |
| 固定列残余警告 | 无 |
| 可复现性 | 两次完整 5,446 行复算，9 个最终分类文件哈希完全一致 |
| 最终自动化测试 | 34 passed |

### 11.1 常见质量问题

| 问题代码 | 含义 | 处理建议 |
|---|---|---|
| `FWHM_UNKNOWN` | 输入 HDR 没有 FWHM，参考谱用受约束插值 | 可继续使用，但报告中注明该限制；有条件时补充传感器 FWHM |
| `NO_GROUND_TRUTH` | 无 XRD/拉曼/薄片/点位光谱真值 | 不能把质量等级解释为矿物学准确率 |
| `ZERO_DETECTION` | 某矿物平衡版无检出 | 检查该矿物是否真的存在、参考谱是否适配、宽松版是否有弱证据 |
| `FIXED_COLUMN_RESIDUAL` | 某组仍有高响应固定列 | 查看条带诊断图和列风险图，必要时重新标定或谨慎使用 |
| `POLICY_NOT_NESTED` | 三档嵌套关系异常 | 不建议发布，需排查运行或算法问题 |
| `OUTPUT_REOPEN_FAILED` | 最终 ENVI 文件无法重开 | 不可发布，检查磁盘、权限或中断 |
| `CLASS_HEADER_INVALID` | 分类头不完整 | 不可发布，需重新运行或修复写出逻辑 |

### 11.2 QA 检查内容

运行结束后，QA 会自动：

1. 重新打开所有最终 ENVI 分类文件。
2. 检查尺寸是否与处理行数一致。
3. 检查是否为单波段分类文件。
4. 检查 `Unclassified` 和 `Masked Pixels` 类别头。
5. 检查三档嵌套关系。
6. 检查掩膜外像元是否一致。
7. 检查置信度范围是否在 0-1。
8. 统计平衡版各组分类像元。
9. 检查是否存在固定列残余风险。
10. 记录零检出矿物和输入能力警告。

## 12. NC-1 最终验收结果参考

NC-1 完整影像信息：

| 项目 | 结果 |
|---|---:|
| 行数 | 5,446 |
| 列数 | 320 |
| 波段数 | 212 |
| 波长范围 | 978.571-2518.460 nm |
| 有效掩膜像元 | 332,815 |
| 掩膜比例 | 19.0974% |
| 分层样本块 | 12 |
| 分层样本有效像元 | 61,649 |
| 估计 SNR | 720.83 |
| Sensor Signature | `56e0629b9d30bc7c4122a04b` |
| FWHM | 未提供，受约束插值 |

七类矿物最终像元数：

| 矿物 | 严格版 | 平衡版 | 宽松版 |
|---|---:|---:|---:|
| 方解石 Calcite | 601 | 2,407 | 8,862 |
| 白云石 Dolomite | 43 | 462 | 11,266 |
| 硬石膏 Anhydrite | 0 | 425 | 1,672 |
| 石膏 Gypsum | 1,036 | 15,067 | 34,195 |
| 伊利石 Illite | 193 | 2,831 | 5,742 |
| 蒙脱石 Montmorillonite | 3 | 554 | 2,913 |
| 高岭石 Kaolinite | 203 | 1,324 | 2,721 |
| 三组分类像元合计 | 2,079 | 23,070 | 67,371 |

平衡版建议作为地质解释主图，宽松版用于检查潜在漏识别，严格版用于提取高可信核心。

## 13. 配置文件说明

桌面端保存的 JSON 配置类似：

```json
{
  "analysis_image": "F:\\NC-1-31_40\\FILL\\SWIR_SG",
  "analysis_mask": "F:\\NC-1-31_40\\FILL\\mask_SG",
  "analysis_input_is_smoothed": true,
  "chunk_rows": 64,
  "sg_window": 11,
  "sg_polyorder": 2,
  "v4": {
    "project_name": "NC1_SWIR_v1",
    "data_physics": "reflectance",
    "spectral_library_root": "E:\\Code\\CoreSpec_Mapper\\spec_lib",
    "requested_minerals": [
      "calcite",
      "dolomite",
      "anhydrite",
      "gypsum",
      "illite",
      "montmorillonite",
      "kaolinite"
    ],
    "mask_bands": [20, 106, 190],
    "minimum_mask_pixels": 1000,
    "preprocessing": {
      "sg_window": 11,
      "sg_polyorder": 2
    },
    "sampling": {
      "blocks": 12,
      "block_rows": 64,
      "minimum_valid_pixels_per_block": 128,
      "minimum_column_samples": 20
    },
    "library_ensemble": {
      "maximum_representatives_per_mineral": 6,
      "dedup_angle_rad": 0.02
    },
    "artifact_control": {
      "edge_width": 2,
      "column_spectral_correction": {
        "enabled": true,
        "strength": 0.70,
        "radius": 2
      }
    },
    "policy_overrides": {}
  },
  "desktop": {
    "output_root": "E:\\Code\\CoreSpec_Mapper\\output"
  }
}
```

普通用户主要通过桌面界面修改路径、项目名、数据物理、目标矿物和少量自动参数。算法专家可在 JSON 中配置更细的策略，但应保留审计记录，不建议直接修改物理硬门槛。

## 14. 命令行辅助用法

虽然本指南面向桌面端，但 CLI 可用于批处理、复现和诊断。

### 14.1 查看帮助

```powershell
corespec --help
```

### 14.2 打印 V4 Catalog

```powershell
corespec v4-catalog
```

输出到文件：

```powershell
corespec v4-catalog --output output\v4_catalog.json
```

### 14.3 执行 V4 审计

完整审计并扫描光谱库：

```powershell
corespec v4-audit --config configs\nc1_v4.json --output output\nc1_v4_audit.json
```

快速审计，跳过较慢的光谱库扫描：

```powershell
corespec v4-audit --config configs\nc1_v4.json --output output\nc1_v4_audit_fast.json --skip-library
```

### 14.4 执行 V4 完整运行

```powershell
corespec v4-run --config configs\nc1_v4.json --output-root output --run-id nc1_v4_full_5446_final
```

若不指定 `--run-id`，程序会按时间自动生成新的运行目录。

### 14.5 处理部分行

用于试运行或调试：

```powershell
corespec v4-run --config configs\nc1_v4.json --output-root output --run-id test_0000_0800 --start-line 0 --stop-line 800
```

注意：部分行结果只适合调试和检查，不应作为完整生产成果发布。

## 15. 常见问题与处理

### 15.1 软件启动失败，提示缺少 PySide6

原因：未安装桌面依赖。

处理：

```powershell
cd E:\Code\CoreSpec_Mapper
python -m pip install -e ".[desktop]"
```

### 15.2 点击 Audit 后提示影像和掩膜尺寸不一致

原因：高光谱影像和掩膜不是同一网格。

处理：

1. 确认选择的是同一批数据的影像和掩膜。
2. 检查 HDR 中 `samples`、`lines` 是否一致。
3. 若掩膜来自其他软件，应重新栅格化或导出为同尺寸 ENVI 文件。

### 15.3 提示 analysis image 没有 wavelength vector

原因：影像 HDR 缺少 `wavelength` 字段。

处理：

1. 检查 HDR 是否完整。
2. 从原始传感器产品或 ENVI 导出文件中恢复波长。
3. 没有波长向量时不能进行矿物光谱识别。

### 15.4 Audit 后矿物显示 Unsupported

常见原因：

| 原因 | 说明 |
|---|---|
| 缺少诊断波段 | 当前影像波长范围不覆盖该矿物必需窗口 |
| 数据物理不匹配 | 例如用亮温或辐亮度直接跑反射率专家 |
| 专家未实现 | VNIR/TIR 专家当前不开放生产 |
| 参考谱不足 | 光谱库中没有合格代表谱 |

处理建议：

1. 确认数据物理选择是否正确。
2. 确认光谱库目录是否完整。
3. 对新矿物先补充 Catalog 和参考谱，再进行验证。
4. 不要通过强行勾选或改 JSON 绕过 Unsupported 门禁。

### 15.5 运行完成但质量等级是 B

B 并不代表失败。B 表示主要检查通过，但有轻度警告。NC-1 最终验收就是 B，原因是缺少 FWHM 和无矿物学真值。

处理：

1. 打开 `reports\quality_report.html` 查看具体警告。
2. 若只有 `FWHM_UNKNOWN` 和 `NO_GROUND_TRUTH`，一般可作为工程结果发布。
3. 发布或汇报时注明限制。
4. 有条件时补充 FWHM 或 XRD/拉曼/薄片真值。

### 15.6 平衡版某矿物像元很少或为零

这不一定是错误。可能表示矿物确实少、信号弱、参考谱不适配或数据能力不足。

处理：

1. 查看该矿物在 `final_sensitive.dat` 中是否出现。
2. 查看 `rejection_reason.dat`，判断主要被什么条件拒绝。
3. 查看 `absorption_depth.dat` 和 `classification_margin.dat`。
4. 检查 `selection_manifest.csv` 中该矿物参考谱数量和来源。
5. 不建议通过全局放宽阈值来制造非零结果。

### 15.7 结果出现竖向条带

处理：

1. 打开 `previews\stripe_diagnosis.png`。
2. 查看对应组的 `stripe_noise_mask.dat`。
3. 查看 `column_risk_score.dat`。
4. 对比 `classes_before_artifact_filter.dat` 和 `final_balanced.dat`。
5. 若质量报告出现 `FIXED_COLUMN_RESIDUAL`，该结果应降级为候选或重新标定。

### 15.8 点击 Cancel 后程序没有立刻停止

这是正常行为。取消会在安全检查点生效，程序需要完成当前分块、关闭数据集并写入取消状态。等待底部状态变为 `Cancelled`。

### 15.9 打开历史运行时提示找不到 run manifest

原因：选择的不是 V4 运行根目录。

应选择包含 `run_manifest.json` 的目录，例如：

```text
E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final
```

不要选择 `previews`、`groups` 或更上一级 `output`。

### 15.10 预览无法显示

可能原因：

1. 运行未完成，预览文件不存在。
2. 运行目录被移动后，manifest 中记录的绝对路径失效。
3. PNG 文件损坏。

桌面端会尝试在当前运行目录的 `previews` 子目录中寻找同名文件。若仍无法显示，可直接打开 `previews` 目录检查 PNG 是否存在。

## 16. 结果解释注意事项

### 16.1 不要把像元数当作矿物含量

V4 输出的是矿物光谱证据分类，不是丰度解混。像元数受掩膜、阈值、空间分辨率、光谱质量、参考谱和矿物组合影响，不能直接换算为矿物含量。

### 16.2 不要把质量等级当作准确率

质量等级 A/B/C/D 是软件工程和证据一致性评价，检查输入完整性、输出完整性、三档嵌套、置信度、条带残余和发布门禁。没有 XRD、拉曼、薄片或点位光谱真值时，不能声明矿物学准确率。

### 16.3 平衡版是主图，宽松版是复核图

推荐：

1. `final_balanced.dat` 用于主解释。
2. `final_sensitive.dat` 用于查找弱蚀变和潜在漏识别。
3. `final_conservative.dat` 用于高可信提取。
4. 三者差异大的位置优先安排地质复核。

### 16.4 注意相似矿物竞争

方解石/白云石、硬石膏/石膏、伊利石/蒙脱石/高岭石之间存在光谱相似性。V4 已进行组内竞争和诊断特征门禁，但对弱吸收、混合像元和噪声区域仍应结合置信度、稳定性和地质背景解释。

### 16.5 关注边缘、饱和和固定列

边缘高反射、托盘残留、裂隙、饱和和固定探测器列可能产生伪影。V4 会输出风险图和清理记录，但强证据细脉会被保护，因此最终解释仍应结合预览和风险图。

## 17. 推荐生产规范

### 17.1 新数据处理规范

1. 为每个新项目建立独立项目名。
2. 保留原始影像、掩膜和光谱库路径，不覆盖输入。
3. 先运行 Audit，再运行完整 Run。
4. 只选择软件判定受支持的矿物。
5. 默认使用平衡版作为主图。
6. 保存每次运行的完整 `run_id` 目录。
7. 发布时同时提供 `run_manifest.json` 和 `quality_report.html`。
8. 重要解释区应结合真值或人工复核。

### 17.2 目录命名建议

项目名建议包含钻孔、传感器和深度段：

```text
NC1_SWIR_v1
ZK001_SWIR_120_160m
ZK008_SWIR_box03
```

运行 ID 建议包含目的和范围：

```text
full_5446_final
pilot_0000_0800
rerun_after_mask_fix
calibration_check_20260719
```

### 17.3 交付成果建议

一次正式交付至少包含：

| 成果 | 文件 |
|---|---|
| 平衡版 ENVI 主图 | `groups\<group>\final_balanced.dat/.hdr` |
| 宽松版复核图 | `groups\<group>\final_sensitive.dat/.hdr` |
| 总览预览 | `previews\comparison_balanced.png`、`comparison_three_profiles.png` |
| 条带诊断 | `previews\stripe_diagnosis.png` |
| 质量报告 | `reports\quality_report.html` |
| 运行清单 | `run_manifest.json` |
| 像元统计 | `tables\mineral_counts.csv` |
| 审计配置 | `config.resolved.json`、`audit.json` |

### 17.4 不建议的操作

1. 不要覆盖或删除历史运行目录。
2. 不要把宽松版直接作为唯一主图。
3. 不要因某矿物少而全局放宽阈值。
4. 不要绕过 Unsupported 门禁。
5. 不要把像元数解释为矿物含量。
6. 不要把无真值质量等级解释为准确率。
7. 不要只看 PNG 预览而不检查质量报告。
8. 不要在输入影像已经 SG 后重复执行 SG，除非明确需要重新处理。

## 18. 安全取消与历史结果重开

### 18.1 安全取消

运行中点击 `Cancel` 后：

1. 后端取消令牌被标记。
2. 当前处理块结束后停止后续计算。
3. 程序关闭数据文件句柄。
4. 运行目录保留 `run_manifest.json`，状态为 `Cancelled` 或失败记录。
5. 不完整输出不会被标记为可发布成果。

如果关闭窗口时任务仍在运行，软件会询问是否取消当前任务并在安全检查点后关闭。

### 18.2 打开历史运行

点击工具栏 `Open run`，选择包含 `run_manifest.json` 的 V4 运行目录。

软件会读取：

1. `run_manifest.json` 中的质量信息；
2. `previews` 中的 PNG 预览；
3. 必要时读取 `reports\quality_report.json`；
4. 在 Results 页显示质量等级和预览。

这适合复核历史成果、比较不同运行或向他人演示结果。

## 19. 版本与复现

V4.0.0 最终验收信息：

| 项目 | 结果 |
|---|---|
| 软件版本 | 4.0.0 |
| 算法 | Adaptive Mineral Evidence Engine |
| CLI 入口 | `corespec` |
| 桌面入口 | `corespec-desktop` |
| V3 兼容 | 保留 |
| 最终测试 | 34 passed |
| NC-1 完整运行 | 5,446 行 |
| 可复现性 | 两次完整复算，9 个最终分类 `.dat` 哈希完全一致 |
| 质量等级 | B |
| 发布门禁 | Passed |
| 固定列残余警告 | 无 |

若需要复现某次结果，应保存：

1. 软件版本和 Git 版本；
2. `config.resolved.json`；
3. `run_manifest.json`；
4. 输入影像和掩膜的 HDR 哈希；
5. 光谱库目录版本；
6. 最终输出文件哈希。

## 20. 快速检查清单

### 20.1 运行前

- [ ] 高光谱影像和 `.hdr` 存在。
- [ ] 掩膜和 `.hdr` 存在。
- [ ] 影像与掩膜行列一致。
- [ ] 高光谱 HDR 包含波长向量。
- [ ] 数据物理选择为 `reflectance`。
- [ ] 已平滑影像勾选 `Input is already SG-smoothed`。
- [ ] 光谱库目录选择 `spec_lib` 或项目认可的库目录。
- [ ] 输出根目录有足够磁盘空间。
- [ ] 先执行 Audit。

### 20.2 运行后

- [ ] Results 页显示质量等级 A 或 B。
- [ ] `reports\quality_report.html` 中无 error。
- [ ] `run_manifest.json` 存在。
- [ ] `previews\comparison_balanced.png` 可打开。
- [ ] 三个矿物组的 `final_balanced.dat/.hdr` 存在。
- [ ] 平衡版目标矿物统计符合预期。
- [ ] 没有 `FIXED_COLUMN_RESIDUAL` 警告。
- [ ] 对重点深度段结合置信度和地质复核。

### 20.3 发布前

- [ ] 明确说明结果是光谱证据分类，不是矿物含量。
- [ ] 明确说明是否缺少 FWHM。
- [ ] 明确说明是否缺少矿物学真值。
- [ ] 随成果附带质量报告和运行清单。
- [ ] 保留完整运行目录，便于追溯。

## 21. 附录：NC-1 推荐路径

| 用途 | 路径 |
|---|---|
| 代码目录 | `E:\Code\CoreSpec_Mapper` |
| V4 配置 | `E:\Code\CoreSpec_Mapper\configs\nc1_v4.json` |
| 光谱库目录 | `E:\Code\CoreSpec_Mapper\spec_lib` |
| 最终运行目录 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final` |
| 平衡版总览 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final\previews\comparison_balanced.png` |
| 三档总览 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final\previews\comparison_three_profiles.png` |
| 条带诊断 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final\previews\stripe_diagnosis.png` |
| 质量报告 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final\reports\quality_report.html` |
| 运行清单 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final\run_manifest.json` |
| 碳酸盐平衡版 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final\groups\carbonates\final_balanced.dat` |
| 硫酸盐平衡版 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final\groups\sulfates\final_balanced.dat` |
| 黏土平衡版 | `E:\Code\CoreSpec_Mapper\output\NC1_SWIR_v1\nc1_v4_full_5446_final\groups\clays\final_balanced.dat` |

---

本指南用于 CoreSpec Mapper V4.0.0 SWIR 桌面 MVP 的操作、复核和交付说明。正式对外发布或用于矿物学结论时，应结合项目真值数据、地质背景和质量报告共同解释。
