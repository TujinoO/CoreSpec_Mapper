from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from corespec_mapper import desktop
from corespec_mapper.v4_models import ProgressEvent, RunCancelled


@unittest.skipIf(desktop.QApplication is None, "PySide6 desktop dependency is not installed")
class DesktopSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = desktop.CoreSpecMainWindow()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_configuration_round_trip_preserves_scientific_overrides(self):
        config = {
            "analysis_image": "C:/data/image.dat",
            "analysis_mask": "C:/data/mask.dat",
            "analysis_input_is_smoothed": True,
            "chunk_rows": 32,
            "v4": {
                "project_name": "Desktop_Test",
                "data_physics": "reflectance",
                "spectral_library_root": "C:/data/libraries",
                "requested_minerals": ["calcite", "dolomite"],
                "mask_bands": [1, 2, 3],
                "preprocessing": {"sg_window": 13, "sg_polyorder": 3},
                "sampling": {"blocks": 10, "block_rows": 48},
                "library_ensemble": {"maximum_representatives_per_mineral": 5, "dedup_angle_rad": 0.015},
                "artifact_control": {"edge_width": 3},
            },
            "desktop": {"output_root": "C:/data/results"},
        }
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            target = Path(temporary) / "target.json"
            source.write_text(json.dumps(config), encoding="utf-8")
            self.window.load_config_file(source)
            self.window.save_config_file(target)
            restored = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(restored["v4"]["artifact_control"], {"edge_width": 3})
        self.assertEqual(restored["v4"]["mask_bands"], [1, 2, 3])
        self.assertEqual(restored["v4"]["requested_minerals"], ["calcite", "dolomite"])
        self.assertEqual(restored["desktop"]["output_root"], "C:/data/results")
        self.assertEqual(restored["v4"]["preprocessing"], {"sg_window": 13, "sg_polyorder": 3})

    def test_worker_emits_audit_run_progress_and_cancel_signals(self):
        from PySide6.QtTest import QSignalSpy

        def fake_audit(config, **kwargs):
            kwargs["progress"](ProgressEvent("audit", 0.5, "checked"))
            return {"ready": True}

        with patch.object(desktop, "audit_v4_project", side_effect=fake_audit):
            worker = desktop._Worker("audit", {})
            progress = QSignalSpy(worker.progress_event)
            completed = QSignalSpy(worker.completed)
            worker.start()
            if completed.count() == 0:
                completed.wait(3000)
            worker.wait()
            self.assertEqual(completed.count(), 1)
            self.assertGreaterEqual(progress.count(), 1)

        def fake_run(config, output_root, **kwargs):
            kwargs["progress"](ProgressEvent("classification", 0.75, "mapped"))
            return {"run_directory": output_root}

        with patch.object(desktop, "run_v4_project", side_effect=fake_run):
            worker = desktop._Worker("run", {}, "C:/results")
            progress = QSignalSpy(worker.progress_event)
            completed = QSignalSpy(worker.completed)
            worker.start()
            if completed.count() == 0:
                completed.wait(3000)
            worker.wait()
            self.assertEqual(completed.count(), 1)
            self.assertGreaterEqual(progress.count(), 1)

        with patch.object(desktop, "audit_v4_project", side_effect=RunCancelled("cancelled")):
            worker = desktop._Worker("audit", {})
            cancelled = QSignalSpy(worker.cancelled)
            worker.start()
            if cancelled.count() == 0:
                cancelled.wait(3000)
            worker.wait()
            self.assertEqual(cancelled.count(), 1)

        idle_worker = desktop._Worker("audit", {})
        idle_worker.cancel()
        self.assertTrue(idle_worker.token.cancelled)

    def test_duration_formatting(self):
        self.assertEqual(desktop._format_duration(None), "--:--")
        self.assertEqual(desktop._format_duration(65), "01:05")
        self.assertEqual(desktop._format_duration(3661), "1:01:01")

    def test_step_navigation_and_existing_run_reopen(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap

        self.window.steps.setCurrentRow(3)
        self.assertEqual(self.window.tabs.currentIndex(), 1)
        self.window.steps.setCurrentRow(6)
        self.assertEqual(self.window.tabs.currentIndex(), 2)
        self.window.tabs.setCurrentIndex(0)
        self.assertEqual(self.window.steps.currentRow(), 0)

        with TemporaryDirectory() as temporary:
            run = Path(temporary) / "example_run"
            previews = run / "previews"
            previews.mkdir(parents=True)
            preview = QPixmap(2, 2)
            preview.fill(Qt.GlobalColor.white)
            self.assertTrue(preview.save(str(previews / "comparison_balanced.png")))
            manifest = {
                "quality": {"quality_grade": "B", "status": "Warning"},
                "previews": {"comparison_balanced": "Z:/moved/comparison_balanced.png"},
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            opened = self.window.load_run_directory(run)

        self.assertEqual(opened.name, "example_run")
        self.assertEqual(self.window.tabs.currentIndex(), 2)
        self.assertIn("Quality B", self.window.quality_label.text())
        self.assertIsNotNone(self.window.preview_pixmap)
        self.assertTrue(self.window.open_output_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
