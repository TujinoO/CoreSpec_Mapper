from pathlib import Path
import json
import unittest

from corespec_mapper.spectral_db import NoUserOverlayProvider, V5SpectralDatabase, default_v5_database_path
from corespec_mapper.spectral_v5 import (
    MineralAliasIndex,
    PurityStatus,
    SourceRole,
    describe_source,
    load_v5_catalog,
    parse_spectrum_name,
)


class V5PrimaryPhaseParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aliases = MineralAliasIndex.load()

    def parse(self, path: str, name: str):
        return parse_spectrum_name(path, name, self.aliases)

    def test_igcp_underscore_is_a_phase_boundary(self):
        value = self.parse("igcp264/igcp_1.sli", "CALCITE_CO2004")
        self.assertEqual(value.primary_phase_id, "calcite")
        self.assertEqual(value.sample_code, "CO2004")
        self.assertEqual(value.purity_status, PurityStatus.DECLARED_PURE)

    def test_mixed_layer_illite_is_not_a_pure_illite_reference(self):
        value = self.parse("igcp264/igcp_1.sli", "ILLITE-SMECTITE_IS200")
        self.assertEqual(value.primary_phase_id, "illite")
        self.assertEqual(value.purity_status, PurityStatus.MIXTURE)

    def test_percent_mixture_is_rejected(self):
        value = self.parse("usgs_min/usgs_min.sli", "hematit1.spc Hematite 2%+98%Qtz GDS76")
        self.assertEqual(value.primary_phase_id, "hematite")
        self.assertEqual(value.purity_status, PurityStatus.MIXTURE)

    def test_taxonomy_parentheses_do_not_override_primary_phase(self):
        cases = (
            ("Prehnite Ca2Al2Si3O10(OH)2 (Phyllosilicates (Chlorite Group) prehnite.1)", "prehnite", False),
            ("Talc Mg3Si4O10(OH)2 (Talc/Pyrophyllite Group talc.1)", "talc", True),
            ("Ilmenite FeTiO3 (Hematite Group ilmenite.1)", "ilmenite", False),
        )
        for raw_name, expected, catalogued in cases:
            with self.subTest(raw_name=raw_name):
                value = self.parse("jhu_lib/minerals.sli", raw_name)
                self.assertEqual(value.primary_phase_id, expected)
                self.assertEqual(value.catalogued, catalogued)

    def test_rock_library_cannot_supply_a_pure_quartz_reference(self):
        value = self.parse("jhu_lib/ign_fn.sli", "Quartz Monzonite (Intermediate Qmonzonite.H1)")
        self.assertEqual(value.primary_phase_id, "quartz")
        self.assertEqual(value.purity_status, PurityStatus.NOT_MINERAL_REFERENCE)
        self.assertEqual(value.material_kind, "rock")

    def test_rock_crystal_is_a_quartz_variety_not_a_rock_rejection(self):
        value = self.parse("jpl_lib/jpl1.sli", "QUARTZ ROCK CRYSTAL TS-1A")
        self.assertEqual(value.primary_phase_id, "quartz")
        self.assertEqual(value.purity_status, PurityStatus.DECLARED_PURE)

    def test_missing_jhu_primary_label_is_quarantined(self):
        value = self.parse("jhu_lib/minerals.sli", "( calcite.2)")
        self.assertIsNone(value.primary_phase_id)
        self.assertEqual(value.purity_status, PurityStatus.AMBIGUOUS)

    def test_source_roles_are_explicit(self):
        self.assertEqual(describe_source("jhu_lib/minerals.sli").role, SourceRole.MINERAL_REFERENCE)
        self.assertEqual(describe_source("jhu_lib/meta_fn.sli").role, SourceRole.ROCK)
        self.assertEqual(describe_source("veg_lib/usgs_veg.sli").role, SourceRole.VEGETATION)

    def test_vegetation_record_inside_igcp_library_is_not_a_mineral(self):
        value = self.parse("igcp264/igcp_2.sli", "DRYGRASS")
        self.assertEqual(value.material_kind, "vegetation")
        self.assertEqual(value.purity_status, PurityStatus.NOT_MINERAL_REFERENCE)


class V5CatalogTests(unittest.TestCase):
    def test_catalog_references_valid_two_level_groups(self):
        catalog = load_v5_catalog()
        self.assertEqual(catalog["version"], "5.1.0")
        self.assertEqual(len(catalog["minerals"]), 27)
        for mineral_id, mineral in catalog["minerals"].items():
            with self.subTest(mineral=mineral_id):
                self.assertIn(mineral["taxonomy_group"], catalog["taxonomy_groups"])
                self.assertIn(mineral["spectral_family"], catalog["spectral_families"])
                self.assertTrue(mineral["alteration_roles"])
                self.assertTrue(mineral["required_windows_nm"])

    def test_sensor_limitations_are_not_presented_as_supported(self):
        minerals = load_v5_catalog()["minerals"]
        self.assertEqual(minerals["quartz"]["support_level"], "unsupported_no_tir")
        self.assertFalse(minerals["quartz"]["recognition_enabled"])
        self.assertEqual(minerals["hematite"]["support_level"], "family_only_rgb_nir")
        self.assertEqual(minerals["goethite"]["support_level"], "family_only_rgb_nir")

    def test_white_mica_competes_in_one_spectral_family(self):
        minerals = load_v5_catalog()["minerals"]
        self.assertEqual(minerals["illite"]["spectral_family"], "white_mica_illite")
        self.assertEqual(minerals["muscovite"]["spectral_family"], "white_mica_illite")


class V5PrivateDatabaseTests(unittest.TestCase):
    EXPECTED_CANDIDATES = {
        "actinolite": 13,
        "alunite": 25,
        "anhydrite": 5,
        "biotite": 9,
        "buddingtonite": 10,
        "calcite": 21,
        "chlorite": 23,
        "dickite": 3,
        "dolomite": 22,
        "epidote": 12,
        "goethite": 16,
        "gypsum": 16,
        "hematite": 23,
        "illite": 17,
        "jarosite": 17,
        "kaolinite": 24,
        "montmorillonite": 16,
        "muscovite": 27,
        "nontronite": 17,
        "phlogopite": 10,
        "pyrophyllite": 14,
        "quartz": 24,
        "sepiolite": 13,
        "siderite": 9,
        "talc": 15,
        "tremolite": 15,
        "vermiculite": 12,
    }

    def test_shipped_database_contains_the_complete_source_collection(self):
        self.assertTrue(default_v5_database_path().is_file())
        with V5SpectralDatabase() as database:
            summary = database.summary()
            self.assertEqual(summary["database_release"], "5.1.0")
            self.assertEqual(summary["source_count"], 27)
            self.assertEqual(summary["measurement_count"], 1783)
            self.assertEqual(summary["mineral_count"], 27)
            self.assertEqual(summary["eligible_measurements"], 428)
            self.assertEqual(summary["label_conflict_measurements"], 2)
            self.assertEqual(database.candidate_counts(), self.EXPECTED_CANDIDATES)

    def test_read_only_query_restores_wavelengths_and_values(self):
        with V5SpectralDatabase() as database:
            spectra = list(database.iter_spectra("calcite"))
        self.assertEqual(len(spectra), 21)
        self.assertTrue(all(item.primary_phase_id == "calcite" for item in spectra))
        self.assertTrue(all(item.wavelengths_nm.size == item.values.size for item in spectra))
        self.assertTrue(all(item.wavelengths_nm[0] < item.wavelengths_nm[-1] for item in spectra))

    def test_sensor_physics_and_required_window_filtering(self):
        expected = {
            "calcite": 21,
            "anhydrite": 4,
            "gypsum": 10,
            "alunite": 19,
            "epidote": 9,
            "hematite": 18,
            "goethite": 8,
        }
        with V5SpectralDatabase() as database:
            for mineral_id, count in expected.items():
                with self.subTest(mineral=mineral_id):
                    self.assertEqual(len(list(database.iter_compatible_spectra(mineral_id))), count)
            self.assertEqual(
                len(list(database.iter_compatible_spectra("quartz", data_physics="emissivity"))),
                0,
            )

    def test_user_overlay_extension_point_defaults_to_empty(self):
        provider = NoUserOverlayProvider()
        self.assertEqual(tuple(provider.database_paths()), ())

    def test_disabled_recognition_reserve_is_packaged(self):
        path = default_v5_database_path().with_name("mineral_recognition_reserve_v5.json")
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["database_release"], "5.1.0")
        self.assertEqual(value["summary"]["reserved_phase_count"], 241)
        self.assertEqual(value["summary"]["reserved_measurement_count"], 914)
        self.assertTrue(all(not item["recognition_enabled"] for item in value["entries"]))


if __name__ == "__main__":
    unittest.main()
