# CoreSpec Mapper 数据与成果外置存储规范

生效日期：2026-09-02

## 1. 目的

`E:\Code\CoreSpec_Mapper` 只保存可版本控制的源码、配置模板、测试、轻量参考资源、正式文档和发布校验信息；原始高光谱数据、中间栅格、实验迭代、运行成果、渲染 QA、构建树、本地打包环境和安装包副本不得长期存放在仓库路径中。

## 2. 统一外置根目录

| 类别 | 路径 | 用途 |
|---|---|---|
| 清理审计 | `E:\Grp_data\CoreSpec_Mapper_Data\00_清理审计_20260902` | 移动计划、哈希清单、恢复脚本 |
| 正式成果与验证证据 | `E:\Grp_data\CoreSpec_Mapper_Data\01_保留科研成果与验证证据` | 已明确保留的正式工程输出、掩膜和报告 |
| 图件原型与 QA | `E:\Grp_data\CoreSpec_Mapper_Data\02_保留图件原型与QA` | 可编辑原型、渲染与替换测试证据 |
| 历史任务成果 | `E:\Grp_data\CoreSpec_Mapper_Data\03_保留NC1历史任务成果` | NC1/EQ2 历史任务的 manifest 和成果暂存 |
| 历史工程审计 | `E:\Grp_data\CoreSpec_Mapper_Data\04_保留历史工程审计` | 轻量审计与实验边界记录 |
| 新运行数据 | `E:\Grp_data\CoreSpec_Mapper_Data\05_运行数据` | 后续所有 CLI/桌面运行的输出根目录 |

原始 RGB、VNIR、SWIR 数据应继续放在专用数据盘或项目数据区，通过配置引用，不复制进源码仓库。

## 3. 运行规则

PowerShell 中先定义统一运行根目录：

```powershell
$CoreSpecRunRoot = "E:\Grp_data\CoreSpec_Mapper_Data\05_运行数据"
New-Item -ItemType Directory -Path $CoreSpecRunRoot -Force | Out-Null
```

所有 `--output`、`--output-root`、桌面端结果目录和脚本 `--output-dir` 都必须指向该根目录或专用外置数据盘，不再使用仓库内的 `output/`、`.codex_runs/` 或 `validation_runs/`。

## 4. 保留与清理边界

- 保留：正式配置、manifest、参数快照、质量报告、人工批准记录、校验哈希、最终预览、独立真值或取样关联信息。
- 条件保留：正式 ENVI 栅格、模型权重、注册代理和概率图；只有确认存在可复算输入、固定代码/参数和完整审计后才能列为删除候选。
- 可进入待删除包：构建树、虚拟环境、解释器缓存、重复安装包、被正式版本明确替代的多轮中间栅格、临时渲染和失败试跑。
- 禁止直接批量删除；先移动到单一日期化待删除包，生成 SHA-256 清单和恢复脚本，再由用户人工确认删除。

工程 QA、包结构检查、哈希一致或自动发布门通过，只说明工程流程和文件完整性，不证明矿物学准确率，也不能替代 XRD、拉曼、薄片、点光谱或化学分析真值。

## 5. 历史文档

旧版文档中的仓库内 `output` 示例属于历史快照，不作为新任务存储规范；当前 README 和本规范优先。冻结报告不因路径迁移而重写，审计清单保留原路径到新路径的一对一映射。
