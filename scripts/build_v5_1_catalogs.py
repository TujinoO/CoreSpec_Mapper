from __future__ import annotations

"""Deterministically extend the V5 catalogs with the V5.1 SWIR targets."""

from copy import deepcopy
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = ROOT / "src" / "corespec_mapper" / "resources"
KNOWLEDGE_PATH = RESOURCE_DIR / "mineral_catalog_v5.json"
RUNTIME_PATH = RESOURCE_DIR / "mineral_catalog_v5_runtime.json"
VERSION = "5.1.0"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    return list(dict.fromkeys([*values, *additions]))


def _upsert_features(group: dict, additions: list[dict]) -> None:
    identifiers = {item["id"] for item in additions}
    group["features"] = [item for item in group["features"] if item.get("id") not in identifiers]
    group["features"].extend(additions)


def _knowledge_minerals() -> dict[str, dict]:
    return {
        "jarosite": {
            "display_name_en": "Jarosite", "display_name_zh": "黄钾铁矾", "formula": "KFe3(SO4)2(OH)6",
            "primary_aliases": ["jarosite"], "taxonomy_group": "sulfates", "spectral_family": "acid_sulfates",
            "alteration_roles": ["acid_sulfate", "oxidation", "gossan"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[1400, 1530], [1810, 1900], [2180, 2300]],
            "key_features_nm": [{"center": 1470}, {"center": 1855}, {"center": 2265, "range": [2215, 2300], "kind": "Fe-OH/SO4"}],
            "confusers": ["alunite", "goethite"],
        },
        "nontronite": {
            "display_name_en": "Nontronite", "display_name_zh": "绿脱石", "formula": "Na0.3Fe2(Si,Al)4O10(OH)2·nH2O",
            "primary_aliases": ["nontronite", "notronite"], "taxonomy_group": "phyllosilicates", "spectral_family": "smectites",
            "alteration_roles": ["argillic", "weathering", "ferric_smectite"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[1350, 1455], [1850, 1985], [2240, 2420]],
            "key_features_nm": [{"center": 1415}, {"center": 1905}, {"center": 2290, "kind": "Fe-OH"}, {"center": 2400}],
            "confusers": ["montmorillonite", "sepiolite", "vermiculite"],
        },
        "talc": {
            "display_name_en": "Talc", "display_name_zh": "滑石", "formula": "Mg3Si4O10(OH)2",
            "primary_aliases": ["talc"], "taxonomy_group": "phyllosilicates", "spectral_family": "mg_fe_oh",
            "alteration_roles": ["talc_carbonate", "magnesian", "hydrothermal"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[2260, 2490]],
            "key_features_nm": [{"center": 2290}, {"center": 2312, "kind": "Mg-OH"}, {"center": 2388}, {"center": 2465}],
            "confusers": ["chlorite", "phlogopite", "sepiolite"],
        },
        "tremolite": {
            "display_name_en": "Tremolite", "display_name_zh": "透闪石", "formula": "Ca2Mg5Si8O22(OH)2",
            "primary_aliases": ["tremolite"], "taxonomy_group": "amphiboles", "spectral_family": "amphibole_mg_fe_oh",
            "alteration_roles": ["calc_silicate", "skarn", "metamorphic"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[2050, 2150], [2260, 2430]],
            "key_features_nm": [{"center": 2110}, {"center": 2315, "kind": "Mg-OH"}, {"center": 2385}],
            "confusers": ["actinolite", "talc", "chlorite"],
        },
        "actinolite": {
            "display_name_en": "Actinolite", "display_name_zh": "阳起石", "formula": "Ca2(Mg,Fe)5Si8O22(OH)2",
            "primary_aliases": ["actinolite"], "taxonomy_group": "amphiboles", "spectral_family": "amphibole_mg_fe_oh",
            "alteration_roles": ["propylitic", "greenschist", "metamorphic"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[2050, 2150], [2260, 2430]],
            "key_features_nm": [{"center": 2115}, {"center": 2318, "kind": "Mg-Fe-OH"}, {"center": 2385}],
            "confusers": ["tremolite", "chlorite", "epidote"],
        },
        "biotite": {
            "display_name_en": "Biotite", "display_name_zh": "黑云母", "formula": "K(Mg,Fe)3AlSi3O10(F,OH)2",
            "primary_aliases": ["biotite", "black mica"], "taxonomy_group": "phyllosilicates", "spectral_family": "mg_fe_oh",
            "alteration_roles": ["potassic", "metamorphic", "magmatic"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[2240, 2440]],
            "key_features_nm": [{"center": 2300}, {"center": 2350, "kind": "Mg-Fe-OH"}, {"center": 2410}],
            "confusers": ["phlogopite", "chlorite"],
        },
        "phlogopite": {
            "display_name_en": "Phlogopite", "display_name_zh": "金云母", "formula": "KMg3AlSi3O10(F,OH)2",
            "primary_aliases": ["phlogopite"], "taxonomy_group": "phyllosilicates", "spectral_family": "mg_fe_oh",
            "alteration_roles": ["potassic", "skarn", "metamorphic"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[2210, 2420]],
            "key_features_nm": [{"center": 2245}, {"center": 2325, "kind": "Mg-OH"}, {"center": 2378}],
            "confusers": ["biotite", "talc", "chlorite"],
        },
        "siderite": {
            "display_name_en": "Siderite", "display_name_zh": "菱铁矿", "formula": "FeCO3",
            "primary_aliases": ["siderite"], "taxonomy_group": "carbonates", "spectral_family": "carbonate_2300",
            "alteration_roles": ["carbonate", "replacement", "iron_carbonate"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[2248, 2402]],
            "key_features_nm": [{"center": 1945}, {"center": 2335, "range": [2320, 2360], "kind": "CO3"}],
            "confusers": ["calcite", "dolomite"],
        },
        "sepiolite": {
            "display_name_en": "Sepiolite", "display_name_zh": "海泡石", "formula": "Mg4Si6O15(OH)2·6H2O",
            "primary_aliases": ["sepiolite"], "taxonomy_group": "phyllosilicates", "spectral_family": "smectites",
            "alteration_roles": ["argillic", "weathering", "magnesian"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[1350, 1455], [1850, 1985], [2260, 2420]],
            "key_features_nm": [{"center": 1415}, {"center": 1915}, {"center": 2315, "kind": "Mg-OH"}, {"center": 2388}],
            "confusers": ["talc", "nontronite", "vermiculite"],
        },
        "vermiculite": {
            "display_name_en": "Vermiculite", "display_name_zh": "蛭石", "formula": "(Mg,Fe,Al)3(Al,Si)4O10(OH)2·4H2O",
            "primary_aliases": ["vermiculite"], "taxonomy_group": "phyllosilicates", "spectral_family": "smectites",
            "alteration_roles": ["weathering", "hydrothermal", "hydrated_clay"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[1350, 1455], [1850, 1985], [2260, 2420]],
            "key_features_nm": [{"center": 1390}, {"center": 1925}, {"center": 2315, "kind": "Mg-Fe-OH"}, {"center": 2395}],
            "confusers": ["montmorillonite", "nontronite", "sepiolite", "chlorite"],
        },
        "buddingtonite": {
            "display_name_en": "Buddingtonite", "display_name_zh": "水铵长石", "formula": "NH4AlSi3O8",
            "primary_aliases": ["buddingtonite", "ammonium feldspar"], "taxonomy_group": "ammonium_minerals", "spectral_family": "ammonium_feldspar",
            "alteration_roles": ["ammonium_alteration", "hydrothermal", "feldspar_replacement"], "support_level": "experimental_swir", "recognition_enabled": True,
            "required_windows_nm": [[1500, 1600], [1980, 2160]],
            "key_features_nm": [{"center": 1560}, {"center": 2045, "range": [2025, 2060], "kind": "NH4"}, {"center": 2122, "kind": "NH4"}],
            "confusers": ["muscovite", "alunite"],
        },
    }


def _policy_template(absolute: tuple[float, float, float], margins: tuple[float, float, float]) -> dict:
    names = ("conservative", "balanced", "sensitive")
    percentiles = (0.015, 0.05, 0.10)
    depths = (0.022, 0.013, 0.007)
    confidences = (0.66, 0.56, 0.45)
    consensuses = (0.70, 0.54, 0.38)
    stabilities = (0.74, 0.60, 0.44)
    return {
        name: {
            "column_percentile": percentiles[index],
            "absolute_sam_threshold_rad": absolute[index],
            "minimum_absorption_depth": depths[index],
            "minimum_margin": margins[index],
            "confidence_threshold": confidences[index],
            "minimum_reference_consensus": consensuses[index],
            "minimum_evidence_stability": stabilities[index],
        }
        for index, name in enumerate(names)
    }


def _runtime_groups(runtime: dict) -> None:
    spatial = deepcopy(runtime["groups"]["mg_fe_oh"]["spatial_cleanup"])

    carbonate = runtime["groups"]["carbonate_2300"]
    _upsert_features(carbonate, [{"id": "carbonate_1945_ratio", "type": "depth_ratio", "window_nm": [1880, 2010]}])
    carbonate["classification_windows_nm"] = [[1880, 2010], [2200, 2450]]

    smectites = runtime["groups"]["smectites"]
    smectites["detection_windows_nm"] = [[1350, 1455], [1850, 1985], [2100, 2420]]
    smectites["classification_windows_nm"] = [[1350, 2420]]
    _upsert_features(smectites, [
        {"id": "smectite_2290_center_nm", "type": "center_nm", "window_nm": [2250, 2335]},
        {"id": "smectite_2290_ratio", "type": "depth_ratio", "window_nm": [2250, 2335]},
        {"id": "smectite_2390_ratio", "type": "depth_ratio", "window_nm": [2350, 2420]},
    ])
    smectites["policies"] = _policy_template((0.080, 0.115, 0.155), (0.020, 0.012, 0.006))

    mgfe = runtime["groups"]["mg_fe_oh"]
    mgfe["detection_windows_nm"] = [[2200, 2490]]
    mgfe["classification_windows_nm"] = [[2100, 2490]]
    _upsert_features(mgfe, [
        {"id": "mgfe_2325_center_nm", "type": "center_nm", "window_nm": [2280, 2360]},
        {"id": "mgfe_2325_ratio", "type": "depth_ratio", "window_nm": [2280, 2360]},
        {"id": "mgfe_2380_ratio", "type": "depth_ratio", "window_nm": [2350, 2420]},
        {"id": "mgfe_2465_ratio", "type": "depth_ratio", "window_nm": [2420, 2490]},
    ])
    mgfe["policies"] = _policy_template((0.080, 0.115, 0.155), (0.022, 0.013, 0.006))

    acid = runtime["groups"]["acid_sulfates"]
    acid["detection_windows_nm"] = [[1400, 1530], [1715, 1900], [2050, 2300]]
    acid["classification_windows_nm"] = [[1350, 2320]]
    _upsert_features(acid, [
        {"id": "jarosite_1470_ratio", "type": "depth_ratio", "window_nm": [1430, 1515]},
        {"id": "jarosite_1850_ratio", "type": "depth_ratio", "window_nm": [1810, 1900]},
        {"id": "jarosite_2265_center_nm", "type": "center_nm", "window_nm": [2215, 2300]},
        {"id": "jarosite_2265_ratio", "type": "depth_ratio", "window_nm": [2215, 2300]},
    ])
    acid["policies"] = _policy_template((0.085, 0.120, 0.165), (0.022, 0.013, 0.006))

    runtime["groups"]["amphibole_mg_fe_oh"] = {
        "display_name": "Tremolite–Actinolite Amphibole Competition",
        "expert": "swir_reflectance",
        "detection_windows_nm": [[2050, 2150], [2260, 2430]],
        "classification_windows_nm": [[2020, 2450]],
        "features": [
            {"id": "amphibole_2110_ratio", "type": "depth_ratio", "window_nm": [2070, 2150]},
            {"id": "amphibole_2315_center_nm", "type": "center_nm", "window_nm": [2285, 2345]},
            {"id": "amphibole_2315_ratio", "type": "depth_ratio", "window_nm": [2285, 2345]},
            {"id": "amphibole_2385_ratio", "type": "depth_ratio", "window_nm": [2350, 2420]},
        ],
        "score_weights": {"shape": 0.55, "derivative": 0.20, "features": 0.25, "fit": 0.0},
        "domain_calibration_strength": 0.0,
        "detection_best_k": 1,
        "classification_best_k": 2,
        "policies": _policy_template((0.075, 0.105, 0.145), (0.028, 0.018, 0.010)),
        "spatial_cleanup": deepcopy(spatial),
    }
    runtime["groups"]["ammonium_feldspar"] = {
        "display_name": "Ammonium Feldspar",
        "expert": "swir_reflectance",
        "detection_windows_nm": [[1500, 1600], [1980, 2160]],
        "classification_windows_nm": [[1450, 2180]],
        "features": [
            {"id": "ammonium_1560_ratio", "type": "depth_ratio", "window_nm": [1515, 1600]},
            {"id": "ammonium_2045_center_nm", "type": "center_nm", "window_nm": [2000, 2070]},
            {"id": "ammonium_2045_ratio", "type": "depth_ratio", "window_nm": [2000, 2070]},
            {"id": "ammonium_2120_center_nm", "type": "center_nm", "window_nm": [2080, 2160]},
            {"id": "ammonium_2120_ratio", "type": "depth_ratio", "window_nm": [2080, 2160]},
        ],
        "score_weights": {"shape": 0.48, "derivative": 0.20, "features": 0.32, "fit": 0.0},
        "domain_calibration_strength": 0.0,
        "detection_best_k": 1,
        "classification_best_k": 2,
        "policies": _policy_template((0.085, 0.120, 0.165), (0.018, 0.010, 0.005)),
        "spatial_cleanup": deepcopy(spatial),
    }


def _runtime_minerals() -> dict[str, dict]:
    base = {
        mineral_id: {
            "display_name_en": value[0], "display_name_zh": value[1], "formula": value[2], "group": value[3],
            "taxonomy_group": value[4], "spectral_family": value[3], "recognition_enabled": True,
            "name_patterns": value[5], "reject_patterns": [], "confusers": value[6], "canonical_anchors": [],
            "support_level": "experimental_swir",
            "experts": {"swir_reflectance": value[7]},
        }
        for mineral_id, value in {
            "jarosite": ("Jarosite", "黄钾铁矾", "KFe3(SO4)2(OH)6", "acid_sulfates", "sulfates", [r"^\s*jarosite(?=$|[^A-Za-z0-9])"], ["alunite", "goethite"], {"required_windows_nm": [[1400, 1530], [1810, 1900], [2180, 2300]], "minimum_valid_bands": 30, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "jarosite_2265_center_nm", "center_window_nm_by_policy": {"conservative": [2248, 2282], "balanced": [2240, 2290], "sensitive": [2230, 2300]}, "minimum_absorption_depth_by_policy": {"conservative": 0.026, "balanced": 0.016, "sensitive": 0.009}, "minimum_feature_values": {"jarosite_1470_ratio": 0.10, "jarosite_1850_ratio": 0.06, "jarosite_2265_ratio": 0.12}}}),
            "nontronite": ("Nontronite", "绿脱石", "Na0.3Fe2(Si,Al)4O10(OH)2·nH2O", "smectites", "phyllosilicates", [r"^\s*nontronite(?=$|[^A-Za-z0-9])", r"^\s*notronite(?=$|[^A-Za-z0-9])"], ["montmorillonite", "sepiolite", "vermiculite"], {"required_windows_nm": [[1350, 1455], [1850, 1985], [2240, 2420]], "minimum_valid_bands": 35, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "smectite_2290_center_nm", "center_window_nm_by_policy": {"conservative": [2278, 2302], "balanced": [2272, 2308], "sensitive": [2265, 2315]}, "minimum_absorption_depth_by_policy": {"conservative": 0.024, "balanced": 0.015, "sensitive": 0.009}, "minimum_feature_values": {"smectite_1400_ratio": 0.10, "smectite_1900_ratio": 0.12, "smectite_2290_ratio": 0.14}}}),
            "talc": ("Talc", "滑石", "Mg3Si4O10(OH)2", "mg_fe_oh", "phyllosilicates", [r"^\s*talc(?=$|[^A-Za-z0-9])"], ["chlorite", "phlogopite", "sepiolite"], {"required_windows_nm": [[2260, 2490]], "minimum_valid_bands": 20, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "mgfe_2325_center_nm", "center_window_nm_by_policy": {"conservative": [2295, 2322], "balanced": [2288, 2328], "sensitive": [2280, 2335]}, "minimum_absorption_depth_by_policy": {"conservative": 0.025, "balanced": 0.015, "sensitive": 0.008}, "minimum_feature_values": {"mgfe_2380_ratio": 0.08, "mgfe_2465_ratio": 0.04}}}),
            "tremolite": ("Tremolite", "透闪石", "Ca2Mg5Si8O22(OH)2", "amphibole_mg_fe_oh", "amphiboles", [r"^\s*tremolite(?=$|[^A-Za-z0-9])"], ["actinolite", "talc", "chlorite"], {"required_windows_nm": [[2050, 2150], [2260, 2430]], "minimum_valid_bands": 25, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "amphibole_2315_center_nm", "center_window_nm_by_policy": {"conservative": [2302, 2324], "balanced": [2298, 2328], "sensitive": [2292, 2334]}, "minimum_absorption_depth_by_policy": {"conservative": 0.024, "balanced": 0.014, "sensitive": 0.008}, "minimum_feature_values": {"amphibole_2110_ratio": 0.05, "amphibole_2385_ratio": 0.10}}}),
            "actinolite": ("Actinolite", "阳起石", "Ca2(Mg,Fe)5Si8O22(OH)2", "amphibole_mg_fe_oh", "amphiboles", [r"^\s*actinolite(?=$|[^A-Za-z0-9])"], ["tremolite", "chlorite", "epidote"], {"required_windows_nm": [[2050, 2150], [2260, 2430]], "minimum_valid_bands": 25, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "amphibole_2315_center_nm", "center_window_nm_by_policy": {"conservative": [2308, 2332], "balanced": [2302, 2337], "sensitive": [2296, 2342]}, "minimum_absorption_depth_by_policy": {"conservative": 0.024, "balanced": 0.014, "sensitive": 0.008}, "minimum_feature_values": {"amphibole_2110_ratio": 0.05, "amphibole_2385_ratio": 0.08}}}),
            "biotite": ("Biotite", "黑云母", "K(Mg,Fe)3AlSi3O10(F,OH)2", "mg_fe_oh", "phyllosilicates", [r"^\s*biotite(?=$|[^A-Za-z0-9])", r"^\s*black[ _-]+mica(?=$|[^A-Za-z0-9])"], ["phlogopite", "chlorite"], {"required_windows_nm": [[2240, 2440]], "minimum_valid_bands": 18, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "mgfe_2325_center_nm", "center_window_nm_by_policy": {"conservative": [2290, 2350], "balanced": [2285, 2356], "sensitive": [2280, 2360]}, "minimum_absorption_depth_by_policy": {"conservative": 0.016, "balanced": 0.010, "sensitive": 0.006}, "minimum_feature_values": {"mgfe_2380_ratio": 0.035}}}),
            "phlogopite": ("Phlogopite", "金云母", "KMg3AlSi3O10(F,OH)2", "mg_fe_oh", "phyllosilicates", [r"^\s*phlogopite(?=$|[^A-Za-z0-9])"], ["biotite", "talc", "chlorite"], {"required_windows_nm": [[2210, 2420]], "minimum_valid_bands": 18, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "mgfe_2325_center_nm", "center_window_nm_by_policy": {"conservative": [2312, 2340], "balanced": [2305, 2348], "sensitive": [2298, 2355]}, "minimum_absorption_depth_by_policy": {"conservative": 0.022, "balanced": 0.013, "sensitive": 0.007}, "minimum_feature_values": {"chlorite_2250_ratio": 0.05, "mgfe_2380_ratio": 0.07}}}),
            "siderite": ("Siderite", "菱铁矿", "FeCO3", "carbonate_2300", "carbonates", [r"^\s*siderite(?=$|[^A-Za-z0-9])"], ["calcite", "dolomite"], {"required_windows_nm": [[2248, 2402]], "minimum_valid_bands": 12, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "carbonate_center_nm", "center_window_nm_by_policy": {"conservative": [2328, 2350], "balanced": [2324, 2355], "sensitive": [2320, 2360]}, "minimum_absorption_depth_by_policy": {"conservative": 0.045, "balanced": 0.030, "sensitive": 0.018}, "minimum_feature_values": {"carbonate_1945_ratio": 0.04}}}),
            "sepiolite": ("Sepiolite", "海泡石", "Mg4Si6O15(OH)2·6H2O", "smectites", "phyllosilicates", [r"^\s*sepiolite(?=$|[^A-Za-z0-9])"], ["talc", "nontronite", "vermiculite"], {"required_windows_nm": [[1350, 1455], [1850, 1985], [2260, 2420]], "minimum_valid_bands": 35, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "smectite_2290_center_nm", "center_window_nm_by_policy": {"conservative": [2302, 2328], "balanced": [2296, 2334], "sensitive": [2290, 2340]}, "minimum_absorption_depth_by_policy": {"conservative": 0.022, "balanced": 0.013, "sensitive": 0.007}, "minimum_feature_values": {"smectite_1900_ratio": 0.12, "smectite_2390_ratio": 0.10}}}),
            "vermiculite": ("Vermiculite", "蛭石", "(Mg,Fe,Al)3(Al,Si)4O10(OH)2·4H2O", "smectites", "phyllosilicates", [r"^\s*vermiculite(?=$|[^A-Za-z0-9])"], ["montmorillonite", "nontronite", "sepiolite", "chlorite"], {"required_windows_nm": [[1350, 1455], [1850, 1985], [2260, 2420]], "minimum_valid_bands": 35, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "smectite_2290_center_nm", "center_window_nm_by_policy": {"conservative": [2298, 2335], "balanced": [2292, 2340], "sensitive": [2285, 2345]}, "minimum_absorption_depth_by_policy": {"conservative": 0.022, "balanced": 0.013, "sensitive": 0.007}, "minimum_feature_values": {"smectite_1400_ratio": 0.10, "smectite_1900_ratio": 0.12, "smectite_2390_ratio": 0.08}}}),
            "buddingtonite": ("Buddingtonite", "水铵长石", "NH4AlSi3O8", "ammonium_feldspar", "ammonium_minerals", [r"^\s*buddingtonite(?=$|[^A-Za-z0-9])", r"^\s*ammonium[ _-]+feldspar(?=$|[^A-Za-z0-9])"], ["muscovite", "alunite"], {"required_windows_nm": [[1500, 1600], [1980, 2160]], "minimum_valid_bands": 25, "maximum_effective_fwhm_nm": 25, "feature_gate": {"center_feature": "ammonium_2045_center_nm", "center_window_nm_by_policy": {"conservative": [2028, 2060], "balanced": [2022, 2065], "sensitive": [2015, 2070]}, "minimum_absorption_depth_by_policy": {"conservative": 0.022, "balanced": 0.013, "sensitive": 0.007}, "minimum_feature_values": {"ammonium_1560_ratio": 0.06, "ammonium_2045_ratio": 0.12, "ammonium_2120_ratio": 0.08}}}),
        }.items()
    }
    base["tremolite"]["phase_discrimination_note"] = "Experimental conditional split from actinolite; require a strong competition margin and geological review."
    base["actinolite"]["phase_discrimination_note"] = "Experimental conditional split from tremolite; require a strong competition margin and geological review."
    base["biotite"]["phase_discrimination_note"] = "Weak SWIR features require reference consensus and geological review."
    base["siderite"]["phase_discrimination_note"] = "The 2.33 µm carbonate feature overlaps calcite; phase output is conditional on full-shape competition."
    return base


def main() -> int:
    knowledge = _read(KNOWLEDGE_PATH)
    runtime = _read(RUNTIME_PATH)
    if knowledge.get("version") not in {"5.0.0", VERSION} or runtime.get("version") not in {"5.0.0", VERSION}:
        raise RuntimeError("Catalog input is not the expected V5.0/V5.1 baseline")

    additions = list(_knowledge_minerals())
    knowledge["version"] = VERSION
    knowledge["description"] = "CoreSpec Mapper V5.1 curated mineral taxonomy and sensor-aware spectral capability catalog"
    knowledge["recognition_reserve_registry"] = "mineral_recognition_reserve_v5.json"
    knowledge["taxonomy_groups"]["carbonates"]["members"] = _append_unique(knowledge["taxonomy_groups"]["carbonates"]["members"], ["siderite"])
    knowledge["taxonomy_groups"]["sulfates"]["members"] = _append_unique(knowledge["taxonomy_groups"]["sulfates"]["members"], ["jarosite"])
    knowledge["taxonomy_groups"]["phyllosilicates"]["members"] = _append_unique(
        knowledge["taxonomy_groups"]["phyllosilicates"]["members"],
        ["nontronite", "talc", "biotite", "phlogopite", "sepiolite", "vermiculite"],
    )
    knowledge["taxonomy_groups"]["amphiboles"] = {"display_name_en": "Amphiboles", "display_name_zh": "角闪石族", "members": ["tremolite", "actinolite"]}
    knowledge["taxonomy_groups"]["ammonium_minerals"] = {"display_name_en": "Ammonium Minerals", "display_name_zh": "铵质矿物", "members": ["buddingtonite"]}
    knowledge["spectral_families"]["carbonate_2300"]["competition_members"] = _append_unique(knowledge["spectral_families"]["carbonate_2300"]["competition_members"], ["siderite"])
    knowledge["spectral_families"]["acid_sulfates"]["competition_members"] = _append_unique(knowledge["spectral_families"]["acid_sulfates"]["competition_members"], ["jarosite"])
    knowledge["spectral_families"]["smectites"]["competition_members"] = _append_unique(knowledge["spectral_families"]["smectites"]["competition_members"], ["nontronite", "sepiolite", "vermiculite"])
    knowledge["spectral_families"]["mg_fe_oh"]["competition_members"] = _append_unique(knowledge["spectral_families"]["mg_fe_oh"]["competition_members"], ["talc", "biotite", "phlogopite"])
    knowledge["spectral_families"]["amphibole_mg_fe_oh"] = {"display_name_en": "Mg-Fe-OH Amphiboles", "display_name_zh": "镁铁羟基角闪石族", "expert": "swir_reflectance", "competition_members": ["tremolite", "actinolite"]}
    knowledge["spectral_families"]["ammonium_feldspar"] = {"display_name_en": "Ammonium Feldspar", "display_name_zh": "铵长石族", "expert": "swir_reflectance", "competition_members": ["buddingtonite"]}
    knowledge["minerals"].update(_knowledge_minerals())

    runtime["version"] = VERSION
    runtime["description"] = "CoreSpec Mapper V5.1 runtime expert configuration with 24 runnable SWIR targets and explicit experimental phase gates."
    runtime["recognition_reserve_registry"] = "mineral_recognition_reserve_v5.json"
    runtime["taxonomy_groups"]["carbonates"]["members"] = _append_unique(runtime["taxonomy_groups"]["carbonates"]["members"], ["siderite"])
    runtime["taxonomy_groups"]["sulfates"]["members"] = _append_unique(runtime["taxonomy_groups"]["sulfates"]["members"], ["jarosite"])
    runtime["taxonomy_groups"]["phyllosilicates"]["members"] = _append_unique(runtime["taxonomy_groups"]["phyllosilicates"]["members"], ["nontronite", "talc", "biotite", "phlogopite", "sepiolite", "vermiculite"])
    runtime["taxonomy_groups"]["amphiboles"] = {"members": ["tremolite", "actinolite"]}
    runtime["taxonomy_groups"]["ammonium_minerals"] = {"members": ["buddingtonite"]}
    runtime["spectral_families"]["carbonate_2300"]["competition_members"] = _append_unique(runtime["spectral_families"]["carbonate_2300"]["competition_members"], ["siderite"])
    runtime["spectral_families"]["acid_sulfates"]["competition_members"] = _append_unique(runtime["spectral_families"]["acid_sulfates"]["competition_members"], ["jarosite"])
    runtime["spectral_families"]["smectites"]["competition_members"] = _append_unique(runtime["spectral_families"]["smectites"]["competition_members"], ["nontronite", "sepiolite", "vermiculite"])
    runtime["spectral_families"]["mg_fe_oh"]["competition_members"] = _append_unique(runtime["spectral_families"]["mg_fe_oh"]["competition_members"], ["talc", "biotite", "phlogopite"])
    runtime["spectral_families"]["amphibole_mg_fe_oh"] = {"competition_members": ["tremolite", "actinolite"]}
    runtime["spectral_families"]["ammonium_feldspar"] = {"competition_members": ["buddingtonite"]}
    _runtime_groups(runtime)
    runtime["minerals"].update(_runtime_minerals())

    if len(knowledge["minerals"]) != 27 or len(runtime["minerals"]) != 27:
        raise RuntimeError("V5.1 catalogs must contain exactly 27 targets")
    if any(mineral_id not in knowledge["minerals"] for mineral_id in additions):
        raise RuntimeError("Missing V5.1 knowledge target")
    _write(KNOWLEDGE_PATH, knowledge)
    _write(RUNTIME_PATH, runtime)
    print(json.dumps({"version": VERSION, "targets": len(knowledge["minerals"]), "added": additions}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
