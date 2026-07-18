# CoreSpec Mapper

**当前版本：V3.0.0 Automatic Library Ensemble**  
**版本日期：2026-07-19**  
**定位：SWIR 岩心高光谱蚀变矿物填图后端**

CoreSpec Mapper 是面向岩心 SWIR 高光谱影像的可配置、可复现、可审计矿物填图后端。项目目标是脱离 ENVI 交互式操作，在代码流程中完成 ENVI 影像读取、岩心掩膜恢复、光谱预处理、连续统去除、SAM/SFF、诊断吸收约束、空间去噪、ENVI 分类成果导出和质量统计。

V3 版本已经完成 NC-1 数据的后端工程验证。桌面 GUI 和 Windows 安装包仍暂缓，当前仓库优先冻结后端接口、配置、测试和 V3 版本记录。

## V3 核心变化

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
scripts/                 审计、阈值扫描和诊断脚本
spec_lib/                ENVI 光谱库输入
src/corespec_mapper/     后端包源码
tests/                   单元测试、ENVI I/O 测试、真实数据冒烟测试和 V3 测试
output/                  本地运行输出，默认不纳入 Git 版本库
```

## 环境与安装

项目当前后端依赖保持轻量，运行期核心依赖为 NumPy。

```powershell
cd E:\Code\CoreSpec_Mapper
python -m pip install -e .
$env:PYTHONPATH = "src"
```

如果不安装为 editable package，也可以仅设置 `PYTHONPATH=src` 后运行 `python -m corespec_mapper ...`。

## 常用命令

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

## 验证结果

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

V3 当前是工程可行性和流程验收版本，不是实验室真值精度报告。在没有 XRD、薄片、拉曼或点位光谱真值之前，不应把像元数直接解释为真实矿物含量，也不应仅凭与历史 PPT 相似程度判断算法精度。

正式解释建议：

- 以平衡版作为主图。
- 以宽松版作为漏识别检查图。
- 针对具体深度段和矿物类别做地质复核。
- 后续参数调整应基于空间位置反馈，不建议继续无方向地全局放宽阈值。

## 版本管理

本仓库使用 Git 标签标注可回退版本。当前 V3 冻结点建议使用：

```powershell
git tag -a v3.0.0 -m "CoreSpec Mapper V3.0.0 Automatic Library Ensemble"
git push origin main
git push origin v3.0.0
```

回退或新建回退分支：

```powershell
git checkout v3.0.0
git switch -c restore-v3 v3.0.0
```

查看版本历史：

```powershell
git log --oneline --decorate --graph --all
git tag --list
```
