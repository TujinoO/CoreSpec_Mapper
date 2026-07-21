from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import unittest

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from corespec_mapper import desktop_v5
from corespec_mapper.v4_models import RunCancelled


@unittest.skipIf(desktop_v5.QApplication is None, "PySide6 desktop dependency is not installed")
class DesktopV5SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = desktop_v5.CoreSpecV5Window(reference_counts={"calcite": 21, "dolomite": 22})

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_real_nine_step_stack_and_v5_taxonomy_tree(self):
        self.assertEqual(self.window.pages.count(), 9)
        self.assertEqual(self.window.steps.count(), 9)
        self.assertEqual(len({id(self.window.pages.widget(i)) for i in range(9)}), 9)
        self.assertGreaterEqual(self.window.mineral_tree.topLevelItemCount(), 6)
        self.assertIn("calcite", self.window.mineral_items)
        self.assertEqual(self.window.mineral_items["calcite"].text(3), "21")
        self.assertTrue(self.window.mineral_items["quartz"].isDisabled())
        self.window.steps.setCurrentRow(4)
        self.assertEqual(self.window.pages.currentIndex(), 4)
        self.assertIn("标准谱", self.window.page_title.text())

    def test_configuration_round_trip_preserves_unknown_science_overrides(self):
        config = {
            "schema_version": 1,
            "application_version": "5.1.0",
            "project": {"name": "V5_Test", "description": "cross sensor", "output_root": "C:/results"},
            "inputs": {
                "analysis_image": "C:/data/SWIR.dat",
                "analysis_domain": "swir",
                "data_physics": "reflectance",
                "input_is_smoothed": False,
                "rgb": "C:/data/RGB.dat",
                "nir": "C:/data/NIR.dat",
                "swir": "C:/data/SWIR.dat",
            },
            "mask": {"mode": "automatic", "minimum_component_pixels": 96, "edge_guard_pixels": 3},
            "minerals": {"requested": ["calcite", "dolomite"], "include_internal_confusers": True},
            "mode": {"profile": "balanced", "automatic": True, "write_all_profiles": True},
            "advanced": {
                "user_overrides_enabled": True,
                "preprocessing": {"sg_window": 13, "sg_polyorder": 3},
                "sampling": {"blocks": 10, "block_rows": 48},
                "execution": {"chunk_rows": 32},
                "library_ensemble": {"maximum_representatives_per_mineral": 5},
                "artifact_control": {"edge_width": 4},
            },
            "science_overrides": {"preserve_me": {"threshold_guard": 0.12}},
            "desktop": {"last_step": 4},
        }
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            target = Path(temporary) / "target.json"
            source.write_text(json.dumps(config), encoding="utf-8")
            self.window.load_config_file(source)
            self.window.save_config_file(target)
            restored = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(restored["science_overrides"], config["science_overrides"])
        self.assertEqual(restored["minerals"]["requested"], ["calcite", "dolomite"])
        self.assertEqual(restored["advanced"]["preprocessing"], {"sg_window": 13, "sg_polyorder": 3})
        self.assertEqual(restored["application_version"], "5.3.0")
        self.assertEqual(restored["mask"]["engine"], "integrated_core_foreground_v2")
        self.assertEqual(restored["mode"]["engine"], "adaptive_v5")
        self.assertTrue(restored["mask"]["approval"]["required"])
        self.assertNotIn("spectral_library_root", json.dumps(restored))

    def test_data_companion_auto_discovery(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                name: root / name
                for name in ("RGB-20260720.dat", "NIR-20260720.dat", "SWIR-20260720.dat")
            }
            for path in files.values():
                path.write_bytes(b"")
            self.window.analysis_path.setText(files["SWIR-20260720.dat"])
            self.window._discover_companions()

        self.assertTrue(self.window.rgb_path.text().endswith("RGB-20260720.dat"))
        self.assertTrue(self.window.nir_path.text().endswith("NIR-20260720.dat"))
        self.assertTrue(self.window.swir_path.text().endswith("SWIR-20260720.dat"))

    def test_audit_populates_spectra_thresholds_mask_and_risks(self):
        audit = {
            "ready": True,
            "summary": "可进入完整运行",
            "inputs": {
                "swir": {
                    "shape": [4341, 320, 212],
                    "wavelength_range_nm": [978.571, 2518.46],
                    "data_physics": "reflectance",
                    "status": "Supported",
                }
            },
            "mask": {"valid_pixels": 224000, "valid_fraction": 0.161, "component_count": 8},
            "selected_references": [
                {
                    "mineral_id": "calcite",
                    "name": "Calcite CO2004",
                    "source": "IGCP-264",
                    "score": 0.91,
                    "wavelengths_nm": [2200, 2300, 2400],
                    "values": [0.54, 0.31, 0.47],
                },
                {
                    "mineral_id": "dolomite",
                    "name": "Dolomite HS102",
                    "source": "USGS",
                    "score": 0.87,
                    "wavelengths_nm": [2200, 2300, 2400],
                    "values": [0.49, 0.34, 0.46],
                },
            ],
            "thresholds": [
                {
                    "group": "carbonates",
                    "parameter": "absolute_sam",
                    "candidate_range": [0.05, 0.12],
                    "final": 0.083,
                    "source": "scene calibration",
                }
            ],
            "risk_summary": {"fixed_column": "low", "missing_fwhm": "warning"},
        }
        self.window.load_audit_result(audit)
        self.assertEqual(self.window.input_table.rowCount(), 1)
        self.assertEqual(self.window.mask_metrics.rowCount(), 3)
        self.assertEqual(len(self.window.spectrum_plot.curves), 2)
        self.assertEqual(self.window.reference_list.count(), 2)
        self.assertEqual(self.window.threshold_table.rowCount(), 1)
        self.assertEqual(self.window.risk_table.rowCount(), 2)
        self.assertIn("审计通过", self.window.evidence_status.text())

    def test_threshold_trial_is_editable_and_exported_to_runtime_config(self):
        self.window.analysis_path.setText("C:/data/SWIR.dat")
        self.window.rgb_path.setText("C:/data/RGB.dat")
        audit = {
            "ready": True,
            "threshold_trial": {
                "status": "resolved",
                "method": "catalog_bounded_scene_proxy_plus_feature_support_v5_3",
                "warnings": [],
            },
            "thresholds": {
                "carbonate_2300": {
                    "conservative": {
                        "status": "resolved",
                        "candidate_range": {
                            "column_percentile": [0.02, 0.12],
                            "absolute_sam_rad": [0.085, 0.15],
                        },
                        "final": {"column_percentile": 0.04, "absolute_sam_rad": 0.09},
                        "selected_proxy_metrics": {"candidate_pixels": 120, "feature_ready_pixels": 24},
                    },
                    "balanced": {
                        "status": "resolved",
                        "candidate_range": {
                            "column_percentile": [0.02, 0.12],
                            "absolute_sam_rad": [0.085, 0.15],
                        },
                        "final": {"column_percentile": 0.06, "absolute_sam_rad": 0.11},
                        "selected_proxy_metrics": {"candidate_pixels": 180, "feature_ready_pixels": 42},
                    },
                    "sensitive": {
                        "status": "resolved",
                        "candidate_range": {
                            "column_percentile": [0.02, 0.12],
                            "absolute_sam_rad": [0.085, 0.15],
                        },
                        "final": {"column_percentile": 0.09, "absolute_sam_rad": 0.14},
                        "selected_proxy_metrics": {"candidate_pixels": 260, "feature_ready_pixels": 61},
                    },
                }
            },
        }
        self.window.load_audit_result(audit)
        self.assertEqual(self.window.threshold_table.rowCount(), 3)
        percentile, absolute = self.window._threshold_editors[("carbonate_2300", "balanced")]
        percentile.setValue(7.0)
        absolute.setValue(0.115)
        config = self.window.build_config()
        override = config["threshold_trial"]["overrides"]["carbonate_2300"]["balanced"]
        self.assertAlmostEqual(override["column_percentile"], 0.07)
        self.assertAlmostEqual(override["absolute_sam_threshold_rad"], 0.115)
        self.assertTrue(self.window._threshold_trial_dirty)

    def test_external_mask_previews_immediately_and_advanced_settings_expand(self):
        from corespec_mapper.envi import write_envi

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.dat"
            write_envi(np.pad(np.ones((6, 4), dtype=np.uint8), ((1, 1), (1, 1))), path)
            self.window.mask_external.setChecked(True)
            self.window.mask_path.setText(path)
            self.app.processEvents()
            self.assertIsNotNone(self.window.mask_preview.pixmap())
            self.assertIsNotNone(self.window.mask_preview_pixmap)
            self.assertIn("已预览", self.window.mask_status.text())
            self.window.mask_zoom.setCurrentIndex(self.window.mask_zoom.findData(2.0))
            self.window._scale_mask_preview()
            self.assertEqual(self.window.mask_preview.pixmap().width(), 2 * self.window.mask_preview_pixmap.width())

        self.assertFalse(self.window.advanced_body.isVisible())
        self.window.advanced_group.setChecked(True)
        self.app.processEvents()
        self.assertFalse(self.window.advanced_body.isHidden())
        self.assertIsInstance(self.window.advanced_group, desktop_v5._VisibleCheckableGroupBox)

    def test_visible_checkboxes_and_automatic_mask_refresh_action(self):
        for checkbox in (
            self.window.input_smoothed,
            self.window.auto_discover,
            self.window.mask_approved,
            self.window.references_confirmed,
        ):
            self.assertIsInstance(checkbox, desktop_v5._VisibleCheckBox)
            checkbox.setChecked(True)
            self.assertTrue(checkbox.isChecked())

        captured = []
        self.window.mask_auto.setChecked(True)
        self.window._start_audit = lambda _checked=False, *, target_step=None: captured.append(target_step)
        self.window.generate_mask_button.click()
        self.assertEqual(captured, [2])
        self.assertIn("生成/刷新", self.window.generate_mask_button.text())

    def test_multiple_reference_curves_remain_individual_and_click_highlights(self):
        audit = {
            "selected_references": [
                {
                    "mineral_id": "dolomite",
                    "name": f"Dolomite reference {index}",
                    "source": "built-in",
                    "wavelengths_nm": [2200, 2300, 2400],
                    "values": [0.50 + 0.01 * index, 0.32, 0.46 - 0.01 * index],
                }
                for index in range(1, 4)
            ]
        }
        self.window._load_references(audit)
        self.window.evidence_mineral.setCurrentIndex(1)
        self.app.processEvents()

        self.assertEqual(len(self.window.spectrum_plot.curves), 3)
        self.assertEqual(self.window.reference_list.count(), 3)
        self.assertEqual(len({curve["color"] for curve in self.window.spectrum_plot.curves}), 3)
        self.window.reference_list.setCurrentRow(2)
        self.assertEqual(self.window.spectrum_plot.highlighted_index, 2)
        self.assertIn("3 条实际入选参考谱", self.window.reference_curve_summary.text())
        self.assertIn("不是合并谱", self.window.reference_curve_summary.text())

    def test_eta_falls_back_to_observed_progress_and_preview_labels_are_chinese(self):
        self.window._task_started_at = desktop_v5.monotonic() - 10.0
        self.window._on_progress(
            {
                "overall_fraction": 0.25,
                "estimated_remaining_seconds": 0,
                "message": "处理中",
            }
        )
        self.assertNotIn("--:--", self.window.timing_label.text())
        self.assertNotIn("剩余 00:00", self.window.timing_label.text())
        self.assertEqual(self.window._preview_label("comparison_balanced"), "平衡档结果对比（推荐）")

    def test_worker_uses_injected_v5_services_and_safe_cancel(self):
        from PySide6.QtTest import QSignalSpy

        def fake_audit(config, **kwargs):
            kwargs["progress"]({"stage": "audit", "overall_fraction": 0.5, "message": "checked"})
            return {"ready": True}

        worker = desktop_v5._V5Worker("audit", {}, audit_function=fake_audit)
        progress = QSignalSpy(worker.progress_event)
        completed = QSignalSpy(worker.completed)
        worker.start()
        if completed.count() == 0:
            completed.wait(3000)
        worker.wait()
        self.assertEqual(completed.count(), 1)
        self.assertGreaterEqual(progress.count(), 1)

        def fake_run(config, output_root, **kwargs):
            kwargs["progress"]({"stage": "run", "overall_percent": 75, "message": "mapped"})
            return {"run_directory": output_root}

        worker = desktop_v5._V5Worker("run", {}, "C:/results", run_function=fake_run)
        completed = QSignalSpy(worker.completed)
        worker.start()
        if completed.count() == 0:
            completed.wait(3000)
        worker.wait()
        self.assertEqual(completed.count(), 1)

        def cancelled_audit(config, **kwargs):
            kwargs["cancel_token"].raise_if_cancelled()
            raise RunCancelled("cancelled")

        worker = desktop_v5._V5Worker("audit", {}, audit_function=cancelled_audit)
        cancelled = QSignalSpy(worker.cancelled)
        worker.cancel()
        worker.start()
        if cancelled.count() == 0:
            cancelled.wait(3000)
        worker.wait()
        self.assertEqual(cancelled.count(), 1)

    def test_existing_run_reopens_and_switches_preview(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap

        with TemporaryDirectory() as temporary:
            run = Path(temporary) / "v5_run"
            previews = run / "previews"
            previews.mkdir(parents=True)
            pixmap = QPixmap(8, 8)
            pixmap.fill(Qt.GlobalColor.white)
            preview = previews / "comparison_balanced.png"
            self.assertTrue(pixmap.save(str(preview)))
            manifest = {
                "quality": {"quality_grade": "B", "status": "Warning", "warnings": ["missing_fwhm"]},
                "previews": {"balanced": str(preview)},
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            opened = self.window.load_run_directory(run)

            self.assertEqual(opened, run)
            self.assertEqual(self.window.pages.currentIndex(), 8)
            self.assertEqual(self.window.preview_combo.count(), 1)
            self.assertIsNotNone(self.window.preview_pixmap)
            self.assertTrue(self.window.open_output_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
