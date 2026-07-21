# CoreSpec Mapper V5.3 下载、安装与升级指南

## 1. 选择正确的下载文件

正式发布页：<https://github.com/TujinoO/CoreSpec_Mapper/releases/tag/v5.3.0>

在页面底部的 **Assets** 区域下载：

- `CoreSpec_Mapper_V5.3.0_Stable_Setup.exe`：Windows x64 安装包；
- `CoreSpec_Mapper_V5.3.0_Stable_Setup.sha256`：完整性校验文件。

不要把 GitHub 自动生成的 `Source code (zip)` 当作可直接运行的安装包。源码压缩包面向开发者，不包含已经安装好的桌面程序。

## 2. 系统要求

- Windows 10 或 Windows 11，64 位；
- 建议 16 GB 内存，长岩心或多矿物项目建议 32 GB；
- 建议至少保留输入数据体积 3 至 5 倍的可用磁盘空间；
- 推荐 1920 × 1080 显示分辨率；
- 自动掩膜需要与主分析立方体对应的 RGB 影像。

安装包已经包含 CPU 运行时、Qt、光谱数据库、矿物 Catalog 和智能掩膜模型。普通用户不需要安装 Python、Conda、PySide6、Torch 或 Inno Setup。

## 3. 校验安装包

在下载目录打开 PowerShell，运行：

```powershell
Get-FileHash .\CoreSpec_Mapper_V5.3.0_Stable_Setup.exe -Algorithm SHA256
```

V5.3.0 Stable 的正确 SHA256 为：

```text
DC4E190B6285A5A715D1FF4CE2641BF3365A587BAF0E9F9852115A3428EBB3A2
```

如果结果不同，不要运行该文件，应重新从正式 Release 页面下载。

## 4. 标准安装

1. 双击 `CoreSpec_Mapper_V5.3.0_Stable_Setup.exe`。
2. 当前安装包尚未进行商业代码签名；若 SmartScreen 显示“未知发布者”，先核对 SHA256，再选择“更多信息 → 仍要运行”。
3. 保持“创建桌面快捷方式”选中。
4. 完成安装后，从桌面或开始菜单启动 `CoreSpec Mapper V5.3`。
5. 首次启动确认标题栏版本为 V5.3。

默认安装目录：

```text
%LOCALAPPDATA%\Programs\CoreSpec Mapper V5.3
```

开始菜单同时提供用户指南和卸载入口。安装目录中的 `_internal`、数据库、模型和 DLL 必须保持原有相对结构，不要单独移动。

## 5. 升级与卸载

- 同一 V5.3 系列可直接运行新版安装包覆盖升级；
- 升级只替换应用文件，不删除用户选择的项目目录、输入数据或识别成果；
- 卸载可通过 Windows“设置 → 应用 → 已安装的应用”，或开始菜单的“卸载 CoreSpec Mapper V5.3”；
- 卸载应用不会自动删除用户输出目录。

## 6. 首次运行建议

1. 新建项目并选择独立输出目录；
2. 选择主 ENVI 高光谱立方体及必要的伴随数据；
3. 自动生成或加载外部掩膜，查看预览并完成人工批准；
4. 选择矿物并确认实际入选标准谱；
5. 审查 Conservative、Balanced、Sensitive 三档阈值试算；
6. 运行后优先查看 Balanced 结果，并结合 Conservative、Sensitive、置信度、拒绝原因和条带诊断共同解释。

## 7. 源码开发安装

开发者需要 Git LFS 和 Python 3.10 或更高版本：

```powershell
git clone https://github.com/TujinoO/CoreSpec_Mapper.git
cd CoreSpec_Mapper
git lfs pull
python -m pip install -e ".[desktop,test]"
corespec-desktop
```

运行测试：

```powershell
python -m pytest -q
```

V5.3.0 Stable 发布基线为 `144 passed`。

## 8. 常见问题

### 安装包为什么不在源码目录中？

安装包大于 GitHub 普通 Git 文件的 100 MB 限制，因此作为 GitHub Release 资产发布。仓库中的 `.sha256` 仅用于核对 Release 资产。

### 双击后提示未知发布者怎么办？

这是因为当前安装包尚未使用商业代码签名证书，不代表 SHA256 校验失败。只应运行从正式 Release 下载且 SHA256 完全一致的文件。

### 自动掩膜不可用怎么办？

确认已经选择 RGB 影像，并保持安装目录中的模型资产完整。若输入或模型缺失，V5.3 会明确阻断，不会静默切换到其他掩膜引擎。

### 识别结果出现 Warning 是否等于失败？

不等于。质量等级和 `publishable` 描述工程链路、输出完整性与伪影控制，不是独立矿物学准确率。缺少 XRD、拉曼、薄片或点光谱真值时仍需地质人员复核。
