import json
import unittest

import numpy as np

from corespec_mapper.catalog import (
    MineralCatalog,
    default_catalog_path,
    default_v5_runtime_catalog_path,
)
from corespec_mapper.sensor import resolve_mineral_support
from corespec_mapper.spectral_v5 import load_v5_catalog
from corespec_mapper.v4_models import SensorCapabilityCard, SupportLevel


RUNNABLE_SWIR_TARGETS = {
    "calcite",
    "dolomite",
    "anhydrite",
    "gypsum",
    "illite",
    "muscovite",
    "montmorillonite",
    "kaolinite",
    "dickite",
    "pyrophyllite",
    "chlorite",
    "alunite",
    "epidote",
    "jarosite",
    "nontronite",
    "talc",
    "tremolite",
    "actinolite",
    "biotite",
    "phlogopite",
    "siderite",
    "sepiolite",
    "vermiculite",
    "buddingtonite",
}


class V5RuntimeCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = load_v5_catalog()
        cls.runtime_path = default_v5_runtime_catalog_path()
        cls.runtime_raw = json.loads(cls.runtime_path.read_text(encoding="utf-8"))
        cls.runtime = MineralCatalog.load(cls.runtime_path)

    def test_v4_remains_the_implicit_default(self):
        self.assertEqual(default_catalog_path().name, "mineral_catalog_v4.json")
        self.assertEqual(MineralCatalog.load().version, "4.0.0")
        self.assertEqual(self.runtime_path.name, "mineral_catalog_v5_runtime.json")

    def test_runtime_catalog_validates_and_matches_the_v5_knowledge_catalog(self):
        self.assertTrue(self.runtime_path.is_file())
        self.assertEqual(self.runtime.version, "5.1.0")
        self.assertEqual(set(self.runtime.minerals), set(self.knowledge["minerals"]))
        for mineral_id, knowledge in self.knowledge["minerals"].items():
            with self.subTest(mineral=mineral_id):
                runtime = self.runtime_raw["minerals"][mineral_id]
                self.assertEqual(runtime["taxonomy_group"], knowledge["taxonomy_group"])
                self.assertEqual(runtime["spectral_family"], knowledge["spectral_family"])
                self.assertEqual(runtime["group"], knowledge["spectral_family"])
                self.assertEqual(runtime["support_level"], knowledge["support_level"])
                self.assertEqual(runtime["recognition_enabled"], knowledge["recognition_enabled"])

    def test_two_level_memberships_and_runtime_competition_groups_are_consistent(self):
        for taxonomy_id, knowledge in self.knowledge["taxonomy_groups"].items():
            with self.subTest(taxonomy=taxonomy_id):
                self.assertEqual(
                    set(self.runtime_raw["taxonomy_groups"][taxonomy_id]["members"]),
                    set(knowledge["members"]),
                )
        for family_id, knowledge in self.knowledge["spectral_families"].items():
            with self.subTest(family=family_id):
                expected = set(knowledge["competition_members"])
                self.assertEqual(
                    set(self.runtime_raw["spectral_families"][family_id]["competition_members"]),
                    expected,
                )
                self.assertEqual(set(self.runtime.group_minerals(family_id)), expected)

    def test_runnable_and_unavailable_experts_are_explicit(self):
        enabled = {
            mineral_id
            for mineral_id, mineral in self.runtime_raw["minerals"].items()
            if mineral["recognition_enabled"]
        }
        self.assertEqual(enabled, RUNNABLE_SWIR_TARGETS)
        self.assertTrue(self.runtime.experts["swir_reflectance"].implemented)
        for mineral_id in RUNNABLE_SWIR_TARGETS:
            self.assertIn("swir_reflectance", self.runtime.mineral(mineral_id).experts)

        self.assertFalse(self.runtime.experts["rgb_nir_fe_oxide"].implemented)
        self.assertFalse(self.runtime.experts["tir_emissivity"].implemented)
        for mineral_id in ("hematite", "goethite", "quartz"):
            self.assertFalse(self.runtime_raw["minerals"][mineral_id]["recognition_enabled"])
        self.assertEqual(self.runtime.mineral("hematite").support_level, "family_only_rgb_nir")
        self.assertEqual(self.runtime.mineral("goethite").support_level, "family_only_rgb_nir")
        self.assertEqual(self.runtime.mineral("quartz").support_level, "unsupported_no_tir")

    def test_current_harmonized_wavelength_range_cannot_unlock_fe_oxide_or_quartz(self):
        wavelengths = np.linspace(691.0, 2518.0, 367)
        card = SensorCapabilityCard(
            "v5-test",
            "vnir+nir+swir",
            "reflectance",
            float(wavelengths[0]),
            float(wavelengths[-1]),
            wavelengths.size,
            wavelengths.size,
            float(np.median(np.diff(wavelengths))),
            "measured",
            10.0,
            120.0,
            (),
            "samples",
            "harmonized",
            ("swir_reflectance",),
            (),
        )
        support = resolve_mineral_support(
            card,
            self.runtime,
            wavelengths_nm=wavelengths,
            reference_counts={mineral_id: 3 for mineral_id in self.runtime.minerals},
        )
        self.assertEqual(support["calcite"].level, SupportLevel.SUPPORTED)
        self.assertEqual(support["hematite"].level, SupportLevel.UNSUPPORTED)
        self.assertEqual(support["goethite"].level, SupportLevel.UNSUPPORTED)
        self.assertEqual(support["quartz"].level, SupportLevel.UNSUPPORTED)

    def test_swir_policy_profiles_form_a_catalog_safe_nested_envelope(self):
        order = ("conservative", "balanced", "sensitive")
        for group in self.runtime.groups.values():
            if group.expert_id != "swir_reflectance":
                continue
            with self.subTest(group=group.group_id):
                policies = [group.policies[name] for name in order]
                self.assertLess(policies[0]["column_percentile"], policies[1]["column_percentile"])
                self.assertLess(policies[1]["column_percentile"], policies[2]["column_percentile"])
                self.assertLess(policies[0]["absolute_sam_threshold_rad"], policies[1]["absolute_sam_threshold_rad"])
                self.assertLess(policies[1]["absolute_sam_threshold_rad"], policies[2]["absolute_sam_threshold_rad"])
                self.assertGreater(policies[0]["minimum_absorption_depth"], policies[1]["minimum_absorption_depth"])
                self.assertGreater(policies[1]["minimum_absorption_depth"], policies[2]["minimum_absorption_depth"])
                self.assertGreater(policies[0]["confidence_threshold"], policies[1]["confidence_threshold"])
                self.assertGreater(policies[1]["confidence_threshold"], policies[2]["confidence_threshold"])
                self.assertGreater(policies[0]["minimum_reference_consensus"], policies[1]["minimum_reference_consensus"])
                self.assertGreater(policies[1]["minimum_reference_consensus"], policies[2]["minimum_reference_consensus"])
                self.assertGreater(policies[0]["minimum_evidence_stability"], policies[1]["minimum_evidence_stability"])
                self.assertGreater(policies[1]["minimum_evidence_stability"], policies[2]["minimum_evidence_stability"])
                self.assertLessEqual(policies[2]["column_percentile"], 0.12)
                self.assertLessEqual(policies[2]["absolute_sam_threshold_rad"], 0.18)
                self.assertEqual(group.domain_calibration_strength, 0.0)
                self.assertAlmostEqual(sum(group.score_weights.values()), 1.0)
                self.assertGreaterEqual(group.spatial_cleanup["min_component"], 2)

    def test_every_runtime_feature_is_inside_its_classification_windows(self):
        for group in self.runtime.groups.values():
            with self.subTest(group=group.group_id):
                for feature in group.features:
                    self.assertTrue(
                        any(
                            lower <= feature.window_nm[0] and feature.window_nm[1] <= upper
                            for lower, upper in group.classification_windows_nm
                        ),
                        f"{group.group_id}/{feature.feature_id} lies outside classification windows",
                    )

    def test_white_mica_competition_preserves_the_aloh_subtype_axis(self):
        group = self.runtime.group("white_mica_illite")
        self.assertEqual(set(self.runtime.group_minerals(group.group_id)), {"illite", "muscovite"})
        self.assertGreaterEqual(group.policies["balanced"]["minimum_margin"], 0.015)
        subtype = self.runtime_raw["groups"][group.group_id]["subtype_axis"]
        self.assertEqual(subtype["feature_id"], "aloh_2200_center_nm")
        classes = subtype["classes"]
        self.assertEqual([item["id"] for item in classes], [
            "short_wave_aloh",
            "medium_wave_aloh",
            "long_wave_aloh",
        ])
        self.assertEqual(classes[0]["upper_exclusive_nm"], 2201)
        self.assertEqual(classes[1]["lower_inclusive_nm"], 2201)
        self.assertEqual(classes[1]["upper_exclusive_nm"], 2209)
        self.assertEqual(classes[2]["lower_inclusive_nm"], 2209)
        for mineral_id in ("illite", "muscovite"):
            gate = self.runtime.mineral(mineral_id).experts["swir_reflectance"]["feature_gate"]
            self.assertEqual(gate["center_feature"], "aloh_2200_center_nm")

    def test_kaolinite_and_dickite_compete_with_doublet_gates(self):
        group = self.runtime.group("kaolin_2170_2205")
        self.assertEqual(set(self.runtime.group_minerals(group.group_id)), {"kaolinite", "dickite"})
        self.assertGreaterEqual(group.policies["balanced"]["minimum_margin"], 0.01)
        feature_ids = {feature.feature_id for feature in group.features}
        self.assertTrue({
            "kaolin_short_center_nm",
            "kaolin_long_center_nm",
            "kaolin_short_ratio",
            "kaolin_long_ratio",
        }.issubset(feature_ids))
        for mineral_id in ("kaolinite", "dickite"):
            gate = self.runtime.mineral(mineral_id).experts["swir_reflectance"]["feature_gate"]
            self.assertEqual(gate["center_feature"], "kaolin_short_center_nm")
            self.assertGreaterEqual(gate["minimum_feature_values"]["kaolin_short_ratio"], 0.30)
            self.assertGreaterEqual(gate["minimum_feature_values"]["kaolin_long_ratio"], 0.45)

    def test_name_matching_is_primary_label_anchored_and_rejects_mixtures(self):
        self.assertEqual(self.runtime.match_spectrum_name("CALCITE_CO2004"), ("calcite", None))
        self.assertEqual(self.runtime.match_spectrum_name("White_mica_SAMPLE_01"), ("muscovite", None))
        self.assertEqual(
            self.runtime.match_spectrum_name("ILLITE-SMECTITE_IS200"),
            ("illite", "mineral_specific_rejected_name"),
        )
        contaminated_parent_labels = (
            "Prehnite Ca2Al2Si3O10(OH)2 (Phyllosilicates (Chlorite Group) prehnite.1)",
            "Ilmenite FeTiO3 (Hematite Group ilmenite.1)",
        )
        for name in contaminated_parent_labels:
            with self.subTest(name=name):
                self.assertEqual(self.runtime.match_spectrum_name(name), (None, None))
        self.assertEqual(
            self.runtime.match_spectrum_name("Talc Mg3Si4O10(OH)2 (Talc/Pyrophyllite Group talc.1)"),
            ("talc", None),
        )
        self.assertEqual(
            self.runtime.match_spectrum_name("Quartz Monzonite (Intermediate Qmonzonite.H1)"),
            ("quartz", "rock_or_mixture_name"),
        )


if __name__ == "__main__":
    unittest.main()
