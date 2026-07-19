# CoreSpec Mapper

- **当前版本：** V4.0.0 Adaptive Mineral Evidence Engine
- **版本日期：** 2026-07-19
- **定位：** 可复用、可审计的 SWIR 岩心高光谱矿物填图桌面工具与后端

CoreSpec Mapper 是面向岩心 SWIR 高光谱影像的可配置、可复现、可审计矿物填图软件。它不依赖 ENVI 运行环境，可完成 ENVI 影像读取、岩心掩膜恢复、传感器能力审计、自动标准谱筛选与重采样、SG 预处理、组级 SAM、连续统形态与 SFF、相似矿物竞争、条带伪影控制、三档 ENVI 分类导出和质量验收。

V4 在 V3 的 NC-1 工程验证基础上增加自适应能力卡、Catalog 驱动的矿物扩展、全深度分层标定、严格/平衡/宽松三档共享证据、非覆盖式运行目录、质量发布门禁和 PySide6 桌面 MVP。VNIR 与 TIR 专家已注册但尚未实现，软件不会把不具备波段与数据物理条件的矿物强行分类。V3 命令与历史配置继续保留。

完整桌面操作、输出目录、质量报告和 ENVI 查看说明见：

```text
docs/CoreSpec_Mapper_V4_桌面版使用说明指南_2026-07-19.md
```

## V4 核心能力

- 根据真实波长、FWHM、坏波段、数据物理和参考谱数量生成 Sensor Capability Card。
- 从完整光谱库自动筛选典型纯矿物谱，保留来源可追溯的锚点与代表谱集成清单。
- 使用 8-16 个非相邻全深度块自动估计场景噪声、列偏差和组级 SAM 阈值。
- 公共光谱证据只计算一次，并派生 `conservative`、`balanced`、`sensitive` 三档嵌套结果。
- 子类竞争使用未做列形态修正的 SG 光谱，避免去条带过程扭曲吸收中心和连续统形态。
- 对固定列、方向性细条带、狭长连通域、边缘和饱和风险分别审计，并保护强证据细脉。
- 每次运行创建独立 `run_id` 目录，输出 ENVI 栅格、JSON/CSV、预览图、HTML/Markdown 质量报告和输入指纹。
- CLI 与桌面端调用同一 V4 服务层；任务支持结构化进度和安全取消。

## V3 兼容基线

V3 不再依赖人工挑选并预先重采样的小型 7 类标准谱库，而是从 `spec_lib` 下的 ENVI 光谱库自动构建目标矿物参考集：

- 自动扫描 `usgs_min`、`jpl_lib`、`jhu_lib`、`igcp264` 等库目录。
- 按纯矿物名称、波长覆盖、重采样质量、诊断吸收特征、近重复程度和来源多样性筛选标准谱。
- 每种目标矿物保留 4 条代表谱，避免不同矿物因参考谱数量不同产生先验偏置。
- 修正 JHU 光谱库微米单位读取问题，按 ENVI 头文件 `wavelength units = Micrometers` 转为纳米。
- 先按碳酸盐、硫酸盐和黏土三个大组进行列分布校准后的 SAM 候选检测，再在组内进行解锁的矿物子类判别。
- 用完整 5,446 行数据估计每个探测器列的 SAM 分布，降低固定列竖向条带进入候选池的概率。
- 同时输出校准前后矿物分数、列阈值、条带噪声掩膜、空间清理前结果和最终 ENVI Classification，便于审计。

V3 解决了 V2 中“先由 SAM 锁死子类，后续 SFF 只能验收或拒绝”的结构性问题。通过大组候选和基本吸收深度检查的像元，会在目标矿物内根据综合分数选择类别，不再把相似矿物压成单一输出。

## 目标矿物

| 大组 | 矿物 |
|---|---|
| 碳酸盐 | Calcite 方解石、Dolomite 白云石 |
| 硫酸盐 | Anhydrite 硬石膏、Gypsum 石膏 |
| 黏土 | Illite 伊利石、Montmorillonite 蒙脱石、Kaolinite 高岭石 |

## 代码结构

```text
configs/                 NC-1 配置，包括 V3 balanced / relaxed 参数
docs/                    V4 桌面版使用、复核和交付说明
scripts/                 审计、阈值扫描和诊断脚本
spec_lib/                ENVI 光谱库输入
src/corespec_mapper/     后端包源码
tests/                   单元、ENVI I/O、真实数据、V3 回归、V4 与桌面冒烟测试
output/                  本地运行输出，默认不纳入 Git 版本库
```

## 环境与安装

项目当前后端依赖保持轻量，运行期核心依赖为 NumPy。

```powershell
cd E:\Code\CoreSpec_Mapper
python -m pip install -e ".[desktop,test]"
$env:PYTHONPATH = "src"
```

如果不安装为 editable package，也可以仅设置 `PYTHONPATH=src` 后运行 `python -m corespec_mapper ...`。

## 常用命令

V4 项目能力与光谱库审计：

```powershell
corespec v4-audit --config configs/nc1_v4.json --output output/nc1_v4_audit.json
```

V4 完整运行（三档结果一次生成）：

```powershell
corespec v4-run --config configs/nc1_v4.json --output-root output --run-id nc1_v4_full_5446_final
```

启动桌面端：

```powershell
corespec-desktop
```

也可通过 CLI 子命令启动桌面端：

```powershell
corespec desktop
```

未安装 editable package 时，可在项目根目录设置 `PYTHONPATH=src` 后运行：

```powershell
$env:PYTHONPATH = "src"
python -m corespec_mapper desktop
```

基础审计：

```powershell
$env:PYTHONPATH = "src"
python -m corespec_mapper audit --config configs/nc1.json --output output/audit.json
```

历史 ENVI SAM 回归：

```powershell
python -m corespec_mapper sam-baseline --config configs/nc1.json --group carbonates --output output/sam_baseline_carbonates
```

V2 风格前 800 行试运行：

```powershell
python -m corespec_mapper pilot --config configs/nc1.json --start-line 0 --stop-line 800 --output output/pilot_0000_0800
```

V3 平衡版前 800 行验证：

```powershell
python -m corespec_mapper v3-pilot --config configs/nc1_v3.json --start-line 0 --stop-line 800 --output output/v3_balanced_optimized_0000_0800
```

V3 宽松版前 800 行验证：

```powershell
python -m corespec_mapper v3-pilot --config configs/nc1_v3_relaxed.json --start-line 0 --stop-line 800 --output output/v3_relaxed_optimized_0000_0800
```

V3 完整 5,446 行生产运行：

```powershell
python -m corespec_mapper v3-pilot --config configs/nc1_v3.json --start-line 0 --output F:\NC-1-31_40\Result\CoreSpec_Mapper_V3_Balanced
python -m corespec_mapper v3-pilot --config configs/nc1_v3_relaxed.json --start-line 0 --output F:\NC-1-31_40\Result\CoreSpec_Mapper_V3_Relaxed
```

运行自动化测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

也可运行：

```powershell
python -m pytest -q
```

## V4 标准输出

每个运行目录包含 `project.csmproj`、`config.resolved.json`、`audit.json`、`summary.json`、`run_manifest.json`，以及以下子目录：

| 目录 | 主要内容 |
|---|---|
| `capability/` | 传感器能力卡、矿物可识别性、抽样计划、列光谱偏差 |
| `library_ensemble/` | 入选/拒绝参考谱、重采样光谱库与选谱审计 |
| `groups/<group>/` | 三档分类、置信度、SFF、吸收深度、拒绝原因和伪影风险图 |
| `confidence/` | 跨矿物组最大置信度、稳定性和拒绝原因 |
| `previews/` | 三档叠加对比、条带清理和各组预览 |
| `reports/` | JSON、Markdown 与 HTML 质量报告 |

ENVI 中优先打开各组的 `final_balanced.dat` 作为主解释图，使用 `final_sensitive.dat` 检查潜在漏识别，并结合 `confidence.dat`、`stripe_noise_mask.dat` 和质量报告复核。

## V4 验收结果

NC-1 V4 最终验收运行：

```text
output/NC1_SWIR_v1/nc1_v4_full_5446_final
```

完整影像信息：

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

最终验收状态：

- 质量等级：`B`
- 状态：`Warning`
- 自动发布门禁：Passed
- 发布建议：可发布，但需注明 FWHM 与无真值限制
- 平衡版七类矿物：全部非零
- 固定列残余警告：无
- 可复现性：两次完整 5,446 行复算，9 个最终分类文件哈希完全一致
- 最终自动化测试：`34 passed`

平衡版建议作为地质解释主图，宽松版用于检查潜在漏识别，严格版用于提取高可信核心。

## V3 输出文件

每个大组目录 `carbonates`、`sulfates`、`clays` 均包含以下核心成果：

| 文件 | 含义 |
|---|---|
| `column_calibrated_candidates.dat` | 完整深度列分位校准后的大组候选 |
| `group_sam_score.dat` | 大组最佳 SAM 角，单位 rad |
| `column_sam_threshold.dat` | 每列使用的 SAM 分位阈值 |
| `v3_raw_mineral_scores.dat` | 域校准前矿物分数，越低越好 |
| `v3_mineral_scores.dat` | 域校准后矿物分数，越低越好 |
| `v3_classes_before_spatial.dat` | 空间清理前子类结果 |
| `stripe_noise_mask.dat` | 被判为方向性条带的像元 |
| `v3_final_classes.dat` | 最终 ENVI Classification 结果 |
| `summary.json` | 参数、计数、阈值、参考谱和诊断统计 |

ENVI 中建议直接打开三个大组目录下的 `v3_final_classes.dat`。输出类别包含 `Unclassified`、目标矿物类别和 `Masked Pixels`，并写入颜色表。

## V3 验证结果

### 前 800 行 V3 平衡版与宽松版

| 矿物 | 平衡版最终像元 | 宽松版最终像元 |
|---|---:|---:|
| 方解石 Calcite | 978 | 3,717 |
| 白云石 Dolomite | 83 | 200 |
| 硬石膏 Anhydrite | 111 | 250 |
| 石膏 Gypsum | 1,549 | 2,595 |
| 伊利石 Illite | 267 | 313 |
| 蒙脱石 Montmorillonite | 134 | 183 |
| 高岭石 Kaolinite | 373 | 515 |

前 800 行两版均输出 `800 x 320` ENVI Classification，七种目标矿物均有非零像元。平衡版更适合作为下一轮地质审查默认基线；宽松版用于检查弱蚀变和脉体外围漏检。

### 完整 5,446 行生产成果

| 矿物 | 平衡版最终像元 | 宽松版最终像元 |
|---|---:|---:|
| 方解石 Calcite | 3,199 | 10,427 |
| 白云石 Dolomite | 1,556 | 14,844 |
| 硬石膏 Anhydrite | 423 | 886 |
| 石膏 Gypsum | 22,757 | 34,496 |
| 伊利石 Illite | 5,759 | 6,869 |
| 蒙脱石 Montmorillonite | 3,692 | 5,260 |
| 高岭石 Kaolinite | 4,025 | 4,275 |

完整影像两版共 6 幅最终分类栅格均通过项目 ENVI 读取器二次打开校验：

- 尺寸：`5446 lines x 320 samples x 1 band`
- 处理行范围：`0-5446`
- 两版 `previews/comparison_final.png` 均覆盖完整深度，像素尺寸为 `1280 x 5446`
- 项目自动化测试：`20 passed`

## 结果边界

V4 当前是 SWIR 光谱证据分类软件，不是丰度解混或实验室真值精度报告。在没有 XRD、薄片、拉曼或点位光谱真值之前，不应把像元数直接解释为真实矿物含量，也不应把质量等级 A/B/C/D 理解为矿物学准确率。

正式解释建议：

- 以 V4 平衡版作为主图。
- 以 V4 宽松版作为漏识别检查图。
- 以 V4 严格版提取高可信核心。
- 针对具体深度段和矿物类别做地质复核。
- 后续参数调整应基于空间位置反馈，不建议继续无方向地全局放宽阈值。

## 版本管理

本仓库使用 Git 标签标注可回退版本。V4 发布标签建议使用：

```powershell
git tag -a v4.0.0 -m "CoreSpec Mapper V4.0.0 Adaptive Mineral Evidence Engine"
git push origin main
git push origin v4.0.0
```

回退或新建回退分支：

```powershell
git checkout v4.0.0
git switch -c restore-v4 v4.0.0
```

查看版本历史：

```powershell
git log --oneline --decorate --graph --all
git tag --list
```
