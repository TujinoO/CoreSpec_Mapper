# CoreSpec Mapper V5 私有光谱数据库清单与识别能力

- 数据库 release：5.1.0
- schema：1
- Catalog：5.1.0
- 核对日期：2026-07-20
- 数据文件：`src/corespec_mapper/resources/corespec_spectral_v5.sqlite3`
- 预留名单：`src/corespec_mapper/resources/mineral_recognition_reserve_v5.json`

## 1. 结论先行

V5.1 内置数据库仍完整保存 27 个源光谱库、1,783 条测量和 1,143 个归并样品。知识 Catalog 已由 16 项扩展到 27 项，其中 24 种矿物具备可运行的 SWIR 专家配置：7 种为已验证默认目标，17 种为实验级可选目标；赤铁矿、针铁矿和石英共 3 项因当前传感器/专家物理条件不足而保持关闭。

本次加入黄钾铁矾、绿脱石、滑石、透闪石、阳起石、黑云母、金云母、菱铁矿、海泡石、蛭石和水铵长石后，eligible 纯相测量由 288 条增加到 428 条；其中当前 24 个可运行 SWIR 目标合计 365 条。新增目标均标记为 `experimental_swir`，代表技术链路已经可运行，不代表矿物学精度已经通过真值验证。

其余 241 个自动解析矿相标签、914 条纯相测量保存在禁用预留注册表中。它们不会出现在当前识别树或运行时输出中，必须补齐专家知识和验收记录后才能逐项启用。

## 2. 数据库核心统计与完整性

| 项目 | 当前值 |
|---|---:|
| SQLite 文件大小 | 10,145,792 bytes（约 9.68 MiB） |
| SHA-256 | `eb21a8fe7d59364694ee7011392a21e5bb3ffe76eddb876070205cb2c299aaf1` |
| 源 ENVI 光谱库 | 27 |
| 完整测量曲线 | 1,783 |
| 归并样品 | 1,143 |
| Catalog 目标矿物 | 27 |
| 可运行 SWIR 目标 | 24 |
| 已验证 / 实验级 SWIR | 7 / 17 |
| Catalog 纯相且 eligible | 428 |
| 标签冲突测量 | 2（均排除） |
| `PRAGMA integrity_check` | `ok` |

eligible 只表示：来源角色为矿物标准、名称判为单一矿相、已进入 Catalog、数值 QC 合格且无标签冲突。每次运行仍会按传感器波长/FWHM/坏波段做兼容性过滤，按物理样品去重，再以 robust medoid 和谱形/来源/测量几何多样性选择少量代表谱。

## 3. 数据来源与质量分布

### 3.1 来源角色

| 来源角色 | 光谱库数 | 测量数 |
|---|---:|---:|
| `lunar` | 1 | 17 |
| `manmade` | 2 | 33 |
| `meteorite` | 1 | 59 |
| `mineral_reference` | 10 | 1,374 |
| `rock` | 6 | 149 |
| `snow_ice` | 1 | 4 |
| `soil` | 1 | 25 |
| `vegetation` | 4 | 119 |
| `water` | 1 | 3 |

### 3.2 纯度/材料判定

| 状态 | 测量数 |
|---|---:|
| `ambiguous` | 1 |
| `declared_pure` | 1,342 |
| `mixture` | 25 |
| `not_mineral_reference` | 415 |

### 3.3 QC 状态

| 状态 | 测量数 |
|---|---:|
| `eligible` | 428 |
| `excluded` | 440 |
| `review` | 915 |

## 4. 27 个 Catalog 目标与实际候选数量

| 一级分类 | 矿物 | 竞争族 | eligible 测量 / 独立样品 | 必需窗口（nm） | 关键特征（nm） | 状态 |
|---|---|---|---:|---|---|---|
| 碳酸盐矿物 | 方解石 `calcite` | `carbonate_2300` | 21 / 9 | 2248–2402 | 2335 CO3 | 已验证 SWIR；默认勾选 |
| 碳酸盐矿物 | 白云石 `dolomite` | `carbonate_2300` | 22 / 8 | 2248–2402 | 2318 CO3 | 已验证 SWIR；默认勾选 |
| 硫酸盐矿物 | 硬石膏 `anhydrite` | `calcium_sulfates` | 5 / 3 | 1350–1800；1850–2300 | 1444、1940、2212 | 已验证 SWIR；默认勾选 |
| 硫酸盐矿物 | 石膏 `gypsum` | `calcium_sulfates` | 16 / 6 | 1350–1800；1850–2300 | 1448、1493、1748、1945、2215 | 已验证 SWIR；默认勾选 |
| 含水层状硅酸盐 | 伊利石 `illite` | `white_mica_illite` | 17 / 9 | 2100–2380 | 2205 Al-OH、2350 | 已验证 SWIR；默认勾选 |
| 含水层状硅酸盐 | 白云母 `muscovite` | `white_mica_illite` | 27 / 17 | 2100–2380 | 2205 Al-OH、2345 | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 蒙脱石 `montmorillonite` | `smectites` | 16 / 13 | 1350–1450；1850–1980；2100–2256 | 1413、1905、2210 Al-OH | 已验证 SWIR；默认勾选 |
| 含水层状硅酸盐 | 高岭石 `kaolinite` | `kaolin_2170_2205` | 24 / 14 | 2140–2225 | 2170 Al-OH_doublet、2205 Al-OH_doublet | 已验证 SWIR；默认勾选 |
| 含水层状硅酸盐 | 地开石 `dickite` | `kaolin_2170_2205` | 3 / 3 | 2120–2240 | 2175 Al-OH_doublet、2205 Al-OH_doublet | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 叶蜡石 `pyrophyllite` | `al_oh_2165` | 14 / 6 | 2100–2380 | 2165 Al-OH、2315 | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 绿泥石 `chlorite` | `mg_fe_oh` | 23 / 11 | 2200–2400 | 2245、2325 Mg-Fe-OH、2370 | 实验级 SWIR；可选、默认不勾选 |
| 硫酸盐矿物 | 明矾石 `alunite` | `acid_sulfates` | 25 / 12 | 1400–1800；2050–2250 | 1481、1764、2160、2209 | 实验级 SWIR；可选、默认不勾选 |
| 其他热液硅酸盐 | 绿帘石 `epidote` | `epidote_family` | 12 / 6 | 1500–1580；2200–2350 | 1546、2252、2332 | 实验级 SWIR；可选、默认不勾选 |
| 铁氧化物与氢氧化物 | 赤铁矿 `hematite` | `fe_oxide_family` | 23 / 15 | 450–1000 | 511、549、899 | 关闭；仅保留资源/定义 |
| 铁氧化物与氢氧化物 | 针铁矿 `goethite` | `fe_oxide_family` | 16 / 8 | 450–1000 | 499、673、894 | 关闭；仅保留资源/定义 |
| 无水硅酸盐 | 石英 `quartz` | `silicate_tir` | 24 / 11 | 7500–14000 | 8000–9500 | 关闭；仅保留资源/定义 |
| 硫酸盐矿物 | 黄钾铁矾 `jarosite` | `acid_sulfates` | 17 / 11 | 1400–1530；1810–1900；2180–2300 | 1470、1855、2265 Fe-OH/SO4 | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 绿脱石 `nontronite` | `smectites` | 17 / 10 | 1350–1455；1850–1985；2240–2420 | 1415、1905、2290 Fe-OH、2400 | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 滑石 `talc` | `mg_fe_oh` | 15 / 7 | 2260–2490 | 2290、2312 Mg-OH、2388、2465 | 实验级 SWIR；可选、默认不勾选 |
| 角闪石族 | 透闪石 `tremolite` | `amphibole_mg_fe_oh` | 15 / 6 | 2050–2150；2260–2430 | 2110、2315 Mg-OH、2385 | 实验级 SWIR；可选、默认不勾选 |
| 角闪石族 | 阳起石 `actinolite` | `amphibole_mg_fe_oh` | 13 / 7 | 2050–2150；2260–2430 | 2115、2318 Mg-Fe-OH、2385 | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 黑云母 `biotite` | `mg_fe_oh` | 9 / 4 | 2240–2440 | 2300、2350 Mg-Fe-OH、2410 | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 金云母 `phlogopite` | `mg_fe_oh` | 10 / 6 | 2210–2420 | 2245、2325 Mg-OH、2378 | 实验级 SWIR；可选、默认不勾选 |
| 碳酸盐矿物 | 菱铁矿 `siderite` | `carbonate_2300` | 9 / 3 | 2248–2402 | 1945、2335 CO3 | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 海泡石 `sepiolite` | `smectites` | 13 / 7 | 1350–1455；1850–1985；2260–2420 | 1415、1915、2315 Mg-OH、2388 | 实验级 SWIR；可选、默认不勾选 |
| 含水层状硅酸盐 | 蛭石 `vermiculite` | `smectites` | 12 / 7 | 1350–1455；1850–1985；2260–2420 | 1390、1925、2315 Mg-Fe-OH、2395 | 实验级 SWIR；可选、默认不勾选 |
| 铵质矿物 | 水铵长石 `buddingtonite` | `ammonium_feldspar` | 10 / 4 | 1500–1600；1980–2160 | 1560、2045 NH4、2122 NH4 | 实验级 SWIR；可选、默认不勾选 |

透闪石/阳起石和黑云母/金云母在前端分别作为独立矿物选择，在后端分别进入同一竞争族。菱铁矿与方解石在约 2.33 µm 高度混淆，透闪石与阳起石也存在连续固溶体特征；这些输出必须结合完整谱形、参考谱共识、竞争间隔和地质背景解释，不能只凭单个吸收中心硬判。

## 5. 两级分类与后端竞争族

### 5.1 前端一级分类

| 一级分类 | 二级矿物成员 |
|---|---|
| 碳酸盐矿物 / `carbonates` | 方解石、白云石、菱铁矿 |
| 硫酸盐矿物 / `sulfates` | 硬石膏、石膏、明矾石、黄钾铁矾 |
| 含水层状硅酸盐 / `phyllosilicates` | 伊利石、白云母、蒙脱石、高岭石、地开石、叶蜡石、绿泥石、绿脱石、滑石、黑云母、金云母、海泡石、蛭石 |
| 其他热液硅酸盐 / `hydrothermal_silicates` | 绿帘石 |
| 铁氧化物与氢氧化物 / `iron_oxides` | 赤铁矿、针铁矿 |
| 无水硅酸盐 / `anhydrous_silicates` | 石英 |
| 角闪石族 / `amphiboles` | 透闪石、阳起石 |
| 铵质矿物 / `ammonium_minerals` | 水铵长石 |

### 5.2 后端光谱竞争族

| 竞争族 | 成员 | 检测窗口（nm） | 关键运行时特征 |
|---|---|---|---|
| `carbonate_2300` | 方解石、白云石、菱铁矿 | 2248–2402 | carbonate_center_nm、carbonate_primary_depth、carbonate_short_ratio、carbonate_long_ratio、carbonate_1945_ratio |
| `calcium_sulfates` | 硬石膏、石膏 | 1350–1800；1850–2300 | sulfate_1450_ratio、sulfate_1750_ratio、sulfate_1940_ratio、sulfate_2200_ratio、sulfate_1940_center_nm |
| `white_mica_illite` | 伊利石、白云母 | 2100–2250；2300–2380 | aloh_2200_center_nm、aloh_2200_depth、white_mica_2345_ratio |
| `smectites` | 蒙脱石、绿脱石、海泡石、蛭石 | 1350–1455；1850–1985；2100–2420 | smectite_1400_ratio、smectite_1900_ratio、smectite_aloh_ratio、smectite_aloh_center_nm、smectite_2290_center_nm、smectite_2290_ratio、smectite_2390_ratio |
| `kaolin_2170_2205` | 高岭石、地开石 | 2120–2240 | kaolin_short_center_nm、kaolin_long_center_nm、kaolin_short_ratio、kaolin_long_ratio、kaolin_2380_ratio |
| `al_oh_2165` | 叶蜡石 | 2100–2225；2280–2350 | pyrophyllite_2165_center_nm、pyrophyllite_2165_ratio、pyrophyllite_2315_ratio |
| `mg_fe_oh` | 绿泥石、滑石、黑云母、金云母 | 2200–2490 | chlorite_2250_ratio、chlorite_2325_center_nm、chlorite_2325_ratio、chlorite_2370_ratio、mgfe_2325_center_nm、mgfe_2325_ratio、mgfe_2380_ratio、mgfe_2465_ratio |
| `acid_sulfates` | 明矾石、黄钾铁矾 | 1400–1530；1715–1900；2050–2300 | alunite_1480_ratio、alunite_1760_ratio、alunite_2170_center_nm、alunite_2170_ratio、jarosite_1470_ratio、jarosite_1850_ratio、jarosite_2265_center_nm、jarosite_2265_ratio |
| `epidote_family` | 绿帘石 | 1500–1585；2200–2360 | epidote_1545_ratio、epidote_2250_ratio、epidote_2330_center_nm、epidote_2330_ratio |
| `fe_oxide_family` | 赤铁矿、针铁矿 | 450–1000 | fe_oxide_long_center_nm |
| `silicate_tir` | 石英 | 7500–14000 | reststrahlen_center_nm |
| `amphibole_mg_fe_oh` | 透闪石、阳起石 | 2050–2150；2260–2430 | amphibole_2110_ratio、amphibole_2315_center_nm、amphibole_2315_ratio、amphibole_2385_ratio |
| `ammonium_feldspar` | 水铵长石 | 1500–1600；1980–2160 | ammonium_1560_ratio、ammonium_2045_center_nm、ammonium_2045_ratio、ammonium_2120_center_nm、ammonium_2120_ratio |

## 6. 本次新增 11 种矿物的识别约束

- 黄钾铁矾：使用约 1.47、1.855 和 2.265 µm 多窗口证据，并与明矾石同族竞争。
- 绿脱石：使用 1.4/1.9 µm 水—OH 和约 2.29/2.40 µm Fe-OH 组合，与蒙脱石、海泡石、蛭石竞争。
- 滑石：使用约 2.29、2.31、2.39、2.465 µm Mg-OH 谱形，并与绿泥石、云母类共同竞争。
- 透闪石/阳起石：使用约 2.11、2.315、2.385 µm 组合；两者分相为实验级条件性输出，要求较高竞争间隔。
- 黑云母/金云母：黑云母 SWIR 特征偏弱，必须依赖参考谱共识和竞争间隔；两者分别输出、同族竞争。
- 菱铁矿：以 2.335 µm 碳酸根为主，并辅助使用约 1.945 µm；与方解石高度混淆，必须结合完整谱形。
- 海泡石/蛭石：联合使用 1.4、1.9、2.31–2.32 和约 2.39 µm 水—OH/Mg-Fe-OH 证据。
- 水铵长石：使用约 1.56、2.035–2.055 和 2.12 µm NH4 特征，独立进入铵长石专家族。

### 6.1 3DSSZ 实测数据链路审计

使用 `configs/v5_3dssz_validation.json` 对新增 11 种矿物执行 V5.1 完整审计，结果为 `Ready`：11/11 目标的必需窗口覆盖率均为 1.0，每种均选出 3–4 条当前传感器可用代表谱；连同内部混淆矿物共选出 51 条标准谱。该审计确认数据读取、掩膜、传感器门禁、参考谱优选和竞争族配置能够跑通；输入仍缺 FWHM，且没有像元级矿物真值，因此不构成准确率结论。

## 7. 标准谱自动优选链路

1. 只查询同一矿相、`declared_pure`、`eligible`、矿物标准来源的测量；
2. 检查数据物理属性和全部必需窗口至少 97% 覆盖；
3. 已知 FWHM 时做高斯卷积，未知时做跨间隙受限插值；
4. 计算波长覆盖、吸收深度、粗糙度和谱形质量；
5. 先按 `sample_id` 去重，避免重复测量放大先验权重；
6. 以 robust medoid 起步，再按谱形差异、来源族和测量几何多样性补充；
7. 运行时输出每条标准谱的选择/拒绝原因、曲线预览和来源审计。

## 8. 其他矿物的禁用预留名单

预留注册表含 241 个名称级矿相、914 条纯相测量。以下名称来自源库主标签自动解析，只是待专家复核的入口，不代表已经确认矿相、建立了可识别专家或允许程序输出。

| 预留 phase_id | 测量数 | 独立样品数 | 来源数 | 状态 |
|---|---:|---:|---:|---|
| `olivine` | 46 | 30 | 2 | `reserved_expert_required` / 禁用 |
| `topaz` | 31 | 23 | 5 | `reserved_expert_required` / 禁用 |
| `hypersthene` | 13 | 11 | 3 | `reserved_expert_required` / 禁用 |
| `almandine` | 12 | 8 | 5 | `reserved_expert_required` / 禁用 |
| `microcline` | 12 | 8 | 5 | `reserved_expert_required` / 禁用 |
| `smectite` | 12 | 3 | 6 | `reserved_expert_required` / 禁用 |
| `beryl` | 11 | 5 | 5 | `reserved_expert_required` / 禁用 |
| `grossular` | 11 | 7 | 5 | `reserved_expert_required` / 禁用 |
| `labradorite` | 11 | 5 | 5 | `reserved_expert_required` / 禁用 |
| `lepidolite` | 11 | 7 | 4 | `reserved_expert_required` / 禁用 |
| `orthoclase` | 11 | 5 | 5 | `reserved_expert_required` / 禁用 |
| `pyrite` | 11 | 7 | 5 | `reserved_expert_required` / 禁用 |
| `antigorite` | 10 | 8 | 2 | `reserved_expert_required` / 禁用 |
| `augite` | 10 | 6 | 3 | `reserved_expert_required` / 禁用 |
| `diopside` | 10 | 6 | 5 | `reserved_expert_required` / 禁用 |
| `tourmaline` | 10 | 4 | 5 | `reserved_expert_required` / 禁用 |
| `fluorite` | 9 | 3 | 4 | `reserved_expert_required` / 禁用 |
| `galena` | 9 | 7 | 4 | `reserved_expert_required` / 禁用 |
| `halloysite` | 9 | 5 | 6 | `reserved_expert_required` / 禁用 |
| `sanidine` | 9 | 5 | 3 | `reserved_expert_required` / 禁用 |
| `andradite` | 8 | 6 | 2 | `reserved_expert_required` / 禁用 |
| `anorthite` | 8 | 5 | 5 | `reserved_expert_required` / 禁用 |
| `magnetite` | 8 | 4 | 5 | `reserved_expert_required` / 禁用 |
| `natrolite` | 8 | 4 | 5 | `reserved_expert_required` / 禁用 |
| `rhodonite` | 8 | 4 | 5 | `reserved_expert_required` / 禁用 |
| `riebeckite` | 8 | 4 | 5 | `reserved_expert_required` / 禁用 |
| `rutile` | 8 | 4 | 5 | `reserved_expert_required` / 禁用 |
| `sphalerite` | 8 | 6 | 4 | `reserved_expert_required` / 禁用 |
| `albite` | 7 | 5 | 5 | `reserved_expert_required` / 禁用 |
| `barite` | 7 | 3 | 5 | `reserved_expert_required` / 禁用 |
| `cordierite` | 7 | 3 | 5 | `reserved_expert_required` / 禁用 |
| `enstatite` | 7 | 3 | 5 | `reserved_expert_required` / 禁用 |
| `nepheline` | 7 | 3 | 5 | `reserved_expert_required` / 禁用 |
| `spodumene` | 7 | 3 | 5 | `reserved_expert_required` / 禁用 |
| `vesuvianite` | 7 | 3 | 5 | `reserved_expert_required` / 禁用 |
| `wollastonite` | 7 | 3 | 5 | `reserved_expert_required` / 禁用 |
| `cerussite` | 6 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `cookeite` | 6 | 4 | 4 | `reserved_expert_required` / 禁用 |
| `hemimorphite` | 6 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `heulandite` | 6 | 4 | 2 | `reserved_expert_required` / 禁用 |
| `hornblende` | 6 | 4 | 2 | `reserved_expert_required` / 禁用 |
| `lizardite` | 6 | 5 | 2 | `reserved_expert_required` / 禁用 |
| `prehnite` | 6 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `rectorite` | 6 | 3 | 5 | `reserved_expert_required` / 禁用 |
| `sodalite` | 6 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `spessartine` | 6 | 5 | 2 | `reserved_expert_required` / 禁用 |
| `acmite` | 5 | 4 | 2 | `reserved_expert_required` / 禁用 |
| `chalcopyrite` | 5 | 3 | 4 | `reserved_expert_required` / 禁用 |
| `datolite` | 5 | 3 | 2 | `reserved_expert_required` / 禁用 |
| `hedenbergite` | 5 | 3 | 2 | `reserved_expert_required` / 禁用 |
| `oligoclase` | 5 | 3 | 4 | `reserved_expert_required` / 禁用 |
| `pectolite` | 5 | 3 | 2 | `reserved_expert_required` / 禁用 |
| `pyrrhotite` | 5 | 3 | 3 | `reserved_expert_required` / 禁用 |
| `rhodochrosite` | 5 | 3 | 4 | `reserved_expert_required` / 禁用 |
| `richterite` | 5 | 3 | 2 | `reserved_expert_required` / 禁用 |
| `saponite` | 5 | 4 | 3 | `reserved_expert_required` / 禁用 |
| `serpentine` | 5 | 3 | 4 | `reserved_expert_required` / 禁用 |
| `stilbite` | 5 | 3 | 4 | `reserved_expert_required` / 禁用 |
| `ulexite` | 5 | 3 | 4 | `reserved_expert_required` / 禁用 |
| `analcime` | 4 | 3 | 3 | `reserved_expert_required` / 禁用 |
| `andalusite` | 4 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `andesine` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `anthophyllite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `arsenopyrite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `azurite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `brucite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `cassiterite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `celestite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `chabazite` | 4 | 3 | 3 | `reserved_expert_required` / 禁用 |
| `chromite` | 4 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `clinochlore` | 4 | 4 | 2 | `reserved_expert_required` / 禁用 |
| `clinochlore_fe` | 4 | 4 | 1 | `reserved_expert_required` / 禁用 |
| `clinozoisite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `colemanite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `corrensite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `corundum` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `cummingtonite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `glauconite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `glaucophane` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `halite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `howlite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `ilmenite` | 4 | 3 | 2 | `reserved_expert_required` / 禁用 |
| `jadeite` | 4 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `malachite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `mizzonite` | 4 | 4 | 1 | `reserved_expert_required` / 禁用 |
| `monticellite` | 4 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `mordenite` | 4 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `pyrope` | 4 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `sillimanite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `staurolite` | 4 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `strontianite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `sulfur` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `thuringite` | 4 | 4 | 1 | `reserved_expert_required` / 禁用 |
| `tincalconite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `trona` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `witherite` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `zircon` | 4 | 2 | 4 | `reserved_expert_required` / 禁用 |
| `zoisite` | 4 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `amblygonite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `anglesite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `antlerite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `apatite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `aphthitalite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `aragonite` | 3 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `atacamite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `borax` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `bornite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `bustamite` | 3 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `chalcocite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `columbite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `cryolite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `desert_varnish` | 3 | 3 | 1 | `reserved_expert_required` / 禁用 |
| `elbaite` | 3 | 3 | 1 | `reserved_expert_required` / 禁用 |
| `erionite` | 3 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `fayalite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `ferroaxinite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `gahnite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `gibbsite` | 3 | 3 | 2 | `reserved_expert_required` / 禁用 |
| `glauberite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `graphite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `hydroxyapophyllite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `johannsenite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `kernite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `kyanite` | 3 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `leucite` | 3 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `magnesiochromite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `magnesite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `marcasite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `mimetite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `molybdenite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `montebrasite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `natrojarosite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `palygorskite` | 3 | 3 | 2 | `reserved_expert_required` / 禁用 |
| `plumbojarosite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `prochlorite` | 3 | 3 | 1 | `reserved_expert_required` / 禁用 |
| `pyrolusite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `realgar` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `scheelite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `scorodite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `smithsonite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `stibnite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `titanite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `triphylite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `tschermigite` | 3 | 1 | 3 | `reserved_expert_required` / 禁用 |
| `annite` | 2 | 2 | 1 | `reserved_expert_required` / 禁用 |
| `bytownite` | 2 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `carnallite` | 2 | 2 | 1 | `reserved_expert_required` / 禁用 |
| `chrysotile` | 2 | 2 | 2 | `reserved_expert_required` / 禁用 |
| `clinoptilolite` | 2 | 2 | 1 | `reserved_expert_required` / 禁用 |
| `hectorite` | 2 | 2 | 1 | `reserved_expert_required` / 禁用 |
| `mascagnite` | 2 | 2 | 1 | `reserved_expert_required` / 禁用 |
| `meionite` | 2 | 2 | 1 | `reserved_expert_required` / 禁用 |
| `opal` | 2 | 2 | 1 | `reserved_expert_required` / 禁用 |
| `thenardite` | 2 | 2 | 1 | `reserved_expert_required` / 禁用 |
| `adularia` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `allanite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `ammonio_jarosite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `ammonio_smectite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `ammonioalunite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `ammonium_chloride` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `amphibole` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `anatase` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `axinite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `bassanite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `bloedite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `bronzite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `brookite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `butlerite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `carbon_black` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `celsian` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `chalcedony` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `chert` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `chlorapatite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `chrysocolla` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `cinnabar` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `clintonite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `cobaltite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `copiapite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `coquimbite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `covellite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `cristobalite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `cronstedtite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `cuprite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `diaspore` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `dipyre` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `dumortierite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `endellite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `epsomite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `eugsterite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `europium_oxide` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `fassaite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `ferrihydrite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `fluorapatite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `forsterite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `gaylussite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `h2o_ice` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `holmquistite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `hornblende_fe` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `hornblende_mg` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `hydrogrossular` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `hydroxyl_apatite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `kainite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `kerogen` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `laumontite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `lazurite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `lepidocrosite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `limonite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `maghemite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `manganite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `margarite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `marialite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `mirabilite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `monazite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `nacrite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `neodymium_oxide` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `nephrite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `niter` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `paragonite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `periclase` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `perthite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `pigeonite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `pinnoite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `pitch_limonite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `polyhalite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `praseodymium_oxide` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `psilomelane` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `pyroxene` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `rivadavite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `roscoelite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `samarium_oxide` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `sauconite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `scolecite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `siderophyllite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `smaragdite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `sodium_bicarbonate` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `sphene` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `syngenite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `tephroite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `uralite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `uvarovite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |
| `zincite` | 1 | 1 | 1 | `reserved_expert_required` / 禁用 |

逐项启用至少需要：确认矿物身份和别名、确定一级分类与竞争族、定义必需波段和诊断特征、列出主要混淆矿物、设置三档特征门禁、完成参考谱 QC、传感器覆盖测试、混淆竞争测试和真值/地质验收。仅增加名称或库中存在曲线不能自动获得识别权限。

## 9. 27 个源光谱库清单

| # | 源库 ID | 来源族 | 角色 | 测量数 | 波长范围（nm） | 测量几何 |
|---:|---|---|---|---:|---|---|
| 1 | `igcp264/igcp_1.sli` | `igcp264` | `mineral_reference` | 27 | 717.0–2506.0 | `mixed_instrument_reflectance` |
| 2 | `igcp264/igcp_2.sli` | `igcp264` | `mineral_reference` | 27 | 300.0–2600.0 | `mixed_instrument_reflectance` |
| 3 | `igcp264/igcp_3.sli` | `igcp264` | `mineral_reference` | 29 | 400.6–2496.8 | `mixed_instrument_reflectance` |
| 4 | `igcp264/igcp_4.sli` | `igcp264` | `mineral_reference` | 29 | 401.1–2496.0 | `mixed_instrument_reflectance` |
| 5 | `igcp264/igcp_5.sli` | `igcp264` | `mineral_reference` | 27 | 1300.0–2500.0 | `mixed_instrument_reflectance` |
| 6 | `jhu_lib/ign_crs.sli` | `jhu_igneous` | `rock` | 34 | 400.0–14011.2 | `directional_hemispherical_reflectance` |
| 7 | `jhu_lib/ign_fn.sli` | `jhu_igneous` | `rock` | 33 | 400.0–14011.0 | `directional_hemispherical_reflectance` |
| 8 | `jhu_lib/lunar.sli` | `jhu_lunar` | `lunar` | 17 | 2079.5–14011.0 | `biconical_reflectance` |
| 9 | `jhu_lib/manmade1.sli` | `jhu_manmade` | `manmade` | 14 | 420.0–14000.0 | `mixed` |
| 10 | `jhu_lib/manmade2.sli` | `jhu_manmade` | `manmade` | 19 | 300.0–12500.0 | `mixed` |
| 11 | `jhu_lib/meta_crs.sli` | `jhu_metamorphic` | `rock` | 25 | 400.0–14983.1 | `directional_hemispherical_reflectance` |
| 12 | `jhu_lib/meta_fn.sli` | `jhu_metamorphic` | `rock` | 29 | 400.0–14983.0 | `directional_hemispherical_reflectance` |
| 13 | `jhu_lib/meteor.sli` | `jhu_meteorite` | `meteorite` | 59 | 2080.3–25044.0 | `biconical_reflectance` |
| 14 | `jhu_lib/minerals.sli` | `jhu_minerals` | `mineral_reference` | 324 | 2079.5–25044.2 | `biconical_reflectance` |
| 15 | `jhu_lib/sed_crs.sli` | `jhu_sedimentary` | `rock` | 15 | 400.0–14011.2 | `directional_hemispherical_reflectance` |
| 16 | `jhu_lib/sed_fn.sli` | `jhu_sedimentary` | `rock` | 13 | 424.0–14983.0 | `directional_hemispherical_reflectance` |
| 17 | `jhu_lib/snow.sli` | `jhu_snow` | `snow_ice` | 4 | 300.0–14010.0 | `mixed` |
| 18 | `jhu_lib/soils.sli` | `jhu_soils` | `soil` | 25 | 400.0–14011.2 | `directional_hemispherical_reflectance` |
| 19 | `jhu_lib/veg.sli` | `vegetation` | `vegetation` | 3 | 302.0–14000.0 | `mixed` |
| 20 | `jhu_lib/water.sli` | `jhu_water` | `water` | 3 | 2079.5–14011.2 | `directional_hemispherical_reflectance` |
| 21 | `jpl_lib/jpl1.sli` | `jpl` | `mineral_reference` | 160 | 400.0–2500.0 | `hemispherical_reflectance` |
| 22 | `jpl_lib/jpl2.sli` | `jpl` | `mineral_reference` | 135 | 400.0–2500.0 | `hemispherical_reflectance` |
| 23 | `jpl_lib/jpl3.sli` | `jpl` | `mineral_reference` | 135 | 400.0–2500.0 | `hemispherical_reflectance` |
| 24 | `usgs_min/usgs_min.sli` | `usgs_v1` | `mineral_reference` | 481 | 395.1–2560.0 | `laboratory_reflectance` |
| 25 | `veg_lib/usgs_veg.sli` | `vegetation` | `vegetation` | 17 | 395.1–2560.0 | `mixed` |
| 26 | `veg_lib/veg_1dry.sli` | `vegetation` | `vegetation` | 74 | 400.0–2500.0 | `mixed` |
| 27 | `veg_lib/veg_2grn.sli` | `vegetation` | `vegetation` | 25 | 400.0–2500.0 | `mixed` |

全部 27 个来源的许可状态仍为 `unverified`；本地技术使用不改变这一事实，对外再分发数据库或制作商业安装包前必须逐源完成许可核查。

## 10. 能力边界与发布判据

1. `experimental_swir` 表示曲线、窗口、特征、门禁、竞争族和运行链路已经接通，不等于准确率已验证。
2. 当前四组历史 V5.0 实测运行没有像元级 XRD、拉曼、薄片或点光谱真值，不能据此给出矿物识别准确率。
3. 赤铁矿/针铁矿仍缺完整校准 VNIR 专家条件；石英仍缺 TIR 传感器和兼容发射率参考谱，因此保持关闭。
4. 影像坏条带、列噪声、边缘、孤立小斑块和材料掩膜质量仍会直接控制结果可靠性；空间聚集性只能作为地质合理性复核，不能替代光谱证据。
5. 数据库曲线数量多不等于识别更准确；独立样品数、谱形代表性、混淆矿物覆盖、传感器适配和真值验收同等重要。
6. 新增光谱资源应先进入隔离 staging，完成来源/许可/物理属性登记、名称解析、纯度与数值 QC、样品去重和标签冲突检查，再由专家审核进入 Catalog。

## 11. 数据库表和可追溯字段

| 表 | 主要内容 |
|---|---|
| `metadata` | release、schema、Catalog 版本和汇总 |
| `taxonomy_group` | 前端一级矿物分类 |
| `spectral_family` | 后端光谱竞争族和专家 |
| `mineral` | 27 个目标、成熟度、窗口和启用状态 |
| `source` | 27 个源库、角色、物理属性、几何、波长轴和文件哈希 |
| `sample` | 归并样品、主相、组成相、纯度、样品码和解析原因 |
| `measurement` | 原始名称、源索引、数值统计、压缩曲线和哈希 |
| `qc_result` | eligible/review/excluded、数值状态、冲突和原因 |

数据库以只读 URI 打开；运行 Manifest 同时记录数据库 release、Catalog 版本和 SHA-256，保证同名资源内容发生变化时仍可审计。
