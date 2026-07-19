from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any
import json
import sys
import traceback

from .catalog import MineralCatalog
from .v4_models import CancellationToken, ProgressEvent, RunCancelled
from .v4_service import audit_v4_project, run_v4_project


try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QAction, QIcon, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QStyle,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - exercised only without desktop dependencies
    QApplication = None


def _logo_path() -> str:
    return str(files("corespec_mapper.resources").joinpath("corespec_logo.png"))


def _icon_path() -> str:
    return str(files("corespec_mapper.resources").joinpath("corespec_logo.ico"))


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


if QApplication is not None:
    class _Worker(QThread):
        progress_event = Signal(dict)
        completed = Signal(object)
        cancelled = Signal(str)
        failed = Signal(str)

        def __init__(self, action: str, config: dict[str, Any], output_root: str | None = None):
            super().__init__()
            self.action = action
            self.config = config
            self.output_root = output_root
            self.token = CancellationToken()

        def cancel(self) -> None:
            self.token.cancel()

        def run(self) -> None:
            try:
                if self.action == "audit":
                    result = audit_v4_project(
                        self.config,
                        scan_library=True,
                        progress=lambda event: self.progress_event.emit(event.to_dict()),
                        cancel_token=self.token,
                    )
                else:
                    assert self.output_root is not None
                    result = run_v4_project(
                        self.config,
                        self.output_root,
                        progress=lambda event: self.progress_event.emit(event.to_dict()),
                        cancel_token=self.token,
                    )
                self.completed.emit(result)
            except RunCancelled as exc:
                self.cancelled.emit(str(exc))
            except Exception as exc:
                self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


    class _PathRow(QWidget):
        def __init__(self, *, directory: bool = False, parent: QWidget | None = None):
            super().__init__(parent)
            self.directory = directory
            self.edit = QLineEdit()
            self.button = QToolButton()
            self.button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
            self.button.setToolTip("Browse")
            self.button.clicked.connect(self._browse)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.edit, 1)
            layout.addWidget(self.button)

        def _browse(self) -> None:
            if self.directory:
                value = QFileDialog.getExistingDirectory(self, "Select directory", self.edit.text())
            else:
                value, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select ENVI data or header",
                    self.edit.text(),
                    "ENVI data (*.dat *.img *.sli *.hdr);;All files (*)",
                )
            if value:
                self.edit.setText(value)

        def text(self) -> str:
            return self.edit.text().strip()

        def setText(self, value: str) -> None:
            self.edit.setText(value)


    class CoreSpecMainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.catalog = MineralCatalog.load()
            self.worker: _Worker | None = None
            self._base_config: dict[str, Any] = {}
            self._close_when_idle = False
            self.preview_pixmap: QPixmap | None = None
            self.preview_paths: dict[str, str] = {}
            self.current_run_directory: str | None = None
            self.setWindowTitle("CoreSpec Mapper V4")
            self.setWindowIcon(QIcon(_icon_path()))
            self.resize(1480, 900)
            self._build_toolbar()
            self._build_ui()
            self._populate_minerals()
            self._set_busy(False)

        def _build_toolbar(self) -> None:
            toolbar = self.addToolBar("Project")
            toolbar.setMovable(False)
            open_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Open", self)
            open_action.setToolTip("Open configuration")
            open_action.triggered.connect(self._open_config)
            save_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self)
            save_action.setToolTip("Save configuration")
            save_action.triggered.connect(self._save_config)
            open_run_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon), "Open run", self)
            open_run_action.setToolTip("Open an existing V4 run")
            open_run_action.triggered.connect(self._open_run_dialog)
            toolbar.addAction(open_action)
            toolbar.addAction(save_action)
            toolbar.addAction(open_run_action)
            toolbar.addSeparator()
            self.audit_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView), "Audit", self)
            self.audit_action.triggered.connect(self._start_audit)
            self.run_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Run", self)
            self.run_action.triggered.connect(self._start_run)
            self.cancel_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Cancel", self)
            self.cancel_action.triggered.connect(self._cancel)
            toolbar.addAction(self.audit_action)
            toolbar.addAction(self.run_action)
            toolbar.addAction(self.cancel_action)

        def _build_ui(self) -> None:
            root = QWidget()
            root_layout = QVBoxLayout(root)
            root_layout.setContentsMargins(8, 8, 8, 8)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            root_layout.addWidget(splitter, 1)

            sidebar = QWidget()
            sidebar.setMaximumWidth(190)
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(0, 0, 0, 0)
            sidebar_layout.setSpacing(8)

            logo = QLabel()
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo.setToolTip("CoreSpec Mapper")
            logo_pixmap = QPixmap(_logo_path())
            if logo_pixmap.isNull():
                logo.setText("CoreSpec")
            else:
                logo.setPixmap(
                    logo_pixmap.scaled(
                        118,
                        118,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            logo.setFixedHeight(126)
            sidebar_layout.addWidget(logo)

            self.steps = QListWidget()
            self.steps.addItems(["Project", "Data", "Mask", "Minerals", "Mode", "Review", "Run & Results"])
            self.steps.setCurrentRow(0)
            sidebar_layout.addWidget(self.steps, 1)
            splitter.addWidget(sidebar)

            self.tabs = QTabWidget()
            splitter.addWidget(self.tabs)
            splitter.setStretchFactor(1, 1)
            self.tabs.addTab(self._setup_tab(), "Setup")
            self.tabs.addTab(self._minerals_tab(), "Minerals")
            self.tabs.addTab(self._results_tab(), "Results")
            self.steps.currentRowChanged.connect(self._navigate_step)
            self.tabs.currentChanged.connect(self._sync_step_to_tab)

            status = QHBoxLayout()
            self.stage_label = QLabel("Ready")
            self.timing_label = QLabel("Elapsed 00:00  |  Remaining --:--")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
            self.cancel_button = QToolButton()
            self.cancel_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.cancel_button.setToolTip("Cancel current task")
            self.cancel_button.clicked.connect(self._cancel)
            status.addWidget(self.stage_label)
            status.addWidget(self.progress_bar, 1)
            status.addWidget(self.timing_label)
            status.addWidget(self.cancel_button)
            root_layout.addLayout(status)
            self.setCentralWidget(root)

        def _setup_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            inputs = QGroupBox("Project inputs")
            form = QFormLayout(inputs)
            self.project_name = QLineEdit("CoreSpec_Project")
            self.image_path = _PathRow()
            self.mask_path = _PathRow()
            self.library_path = _PathRow(directory=True)
            self.output_path = _PathRow(directory=True)
            self.physics = QComboBox()
            self.physics.addItems(["reflectance", "emissivity", "radiance", "brightness_temperature"])
            self.smoothed = QCheckBox("Input is already SG-smoothed")
            self.smoothed.setChecked(True)
            form.addRow("Project", self.project_name)
            form.addRow("Hyperspectral image", self.image_path)
            form.addRow("Core mask", self.mask_path)
            form.addRow("Spectral libraries", self.library_path)
            form.addRow("Output root", self.output_path)
            form.addRow("Data physics", self.physics)
            form.addRow("Preprocessing", self.smoothed)
            layout.addWidget(inputs)

            advanced = QGroupBox("Advanced automatic settings")
            advanced.setCheckable(True)
            advanced.setChecked(False)
            advanced_form = QFormLayout(advanced)
            self.sg_window = QSpinBox()
            self.sg_window.setRange(5, 31)
            self.sg_window.setSingleStep(2)
            self.sg_window.setValue(11)
            self.sg_order = QSpinBox()
            self.sg_order.setRange(1, 5)
            self.sg_order.setValue(2)
            self.sample_blocks = QSpinBox()
            self.sample_blocks.setRange(8, 16)
            self.sample_blocks.setValue(12)
            self.block_rows = QSpinBox()
            self.block_rows.setRange(16, 256)
            self.block_rows.setValue(64)
            self.max_references = QSpinBox()
            self.max_references.setRange(1, 6)
            self.max_references.setValue(6)
            advanced_form.addRow("SG window", self.sg_window)
            advanced_form.addRow("SG polynomial", self.sg_order)
            advanced_form.addRow("Depth blocks", self.sample_blocks)
            advanced_form.addRow("Rows per block", self.block_rows)
            advanced_form.addRow("Maximum references", self.max_references)
            layout.addWidget(advanced)
            controls = QHBoxLayout()
            self.audit_button = QPushButton("Audit")
            self.audit_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView))
            self.audit_button.clicked.connect(self._start_audit)
            self.run_button = QPushButton("Run V4")
            self.run_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.run_button.clicked.connect(self._start_run)
            controls.addStretch(1)
            controls.addWidget(self.audit_button)
            controls.addWidget(self.run_button)
            layout.addLayout(controls)
            layout.addStretch(1)
            return tab

        def _minerals_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            self.mineral_table = QTableWidget(0, 5)
            self.mineral_table.setHorizontalHeaderLabels(["Use", "Mineral", "Group", "Support", "Reason"])
            self.mineral_table.horizontalHeader().setStretchLastSection(True)
            self.mineral_table.setAlternatingRowColors(True)
            layout.addWidget(self.mineral_table)
            return tab

        def _results_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            controls = QHBoxLayout()
            self.preview_combo = QComboBox()
            self.preview_combo.currentTextChanged.connect(self._show_preview)
            self.open_run_button = QPushButton("Open run")
            self.open_run_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
            self.open_run_button.clicked.connect(self._open_run_dialog)
            self.open_output_button = QPushButton("Open output")
            self.open_output_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
            self.open_output_button.clicked.connect(self._open_output)
            self.open_output_button.setEnabled(False)
            controls.addWidget(self.preview_combo, 1)
            controls.addWidget(self.open_run_button)
            controls.addWidget(self.open_output_button)
            layout.addLayout(controls)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            self.preview_label = QLabel()
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setMinimumSize(640, 480)
            scroll.setWidget(self.preview_label)
            layout.addWidget(scroll, 1)
            self.quality_label = QLabel("No completed run")
            layout.addWidget(self.quality_label)
            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(150)
            layout.addWidget(self.log)
            return tab

        def _populate_minerals(self) -> None:
            rows = [mineral for mineral in self.catalog.minerals.values()]
            self.mineral_table.setRowCount(len(rows))
            self.mineral_rows: dict[str, int] = {}
            for row, mineral in enumerate(rows):
                self.mineral_rows[mineral.mineral_id] = row
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                use.setCheckState(Qt.CheckState.Checked if mineral.support_level == "validated_swir" else Qt.CheckState.Unchecked)
                name = QTableWidgetItem(f"{mineral.display_name_en}  {mineral.display_name_zh}")
                group = QTableWidgetItem(mineral.group_id)
                expert = self.catalog.experts[self.catalog.groups[mineral.group_id].expert_id]
                support = QTableWidgetItem("Pending audit" if expert.implemented else "Unsupported")
                reason = QTableWidgetItem("" if expert.implemented else f"{expert.display_name} is not implemented")
                if not expert.implemented:
                    use.setCheckState(Qt.CheckState.Unchecked)
                    use.setFlags(Qt.ItemFlag.ItemIsUserCheckable)
                for column, item in enumerate((use, name, group, support, reason)):
                    self.mineral_table.setItem(row, column, item)
            self.mineral_table.resizeColumnsToContents()

        def _config(self) -> dict[str, Any]:
            paths = {
                "analysis_image": self.image_path.text(),
                "analysis_mask": self.mask_path.text(),
                "spectral_library_root": self.library_path.text(),
            }
            missing = [name for name, value in paths.items() if not value]
            if missing:
                raise ValueError(f"Missing input: {', '.join(missing)}")
            selected = []
            for mineral_id, row in self.mineral_rows.items():
                if self.mineral_table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    selected.append(mineral_id)
            if not selected:
                raise ValueError("Select at least one supported mineral")
            if self.sg_window.value() % 2 == 0:
                raise ValueError("SG window must be an odd number")
            result = deepcopy(self._base_config)
            result.update({
                "analysis_image": paths["analysis_image"],
                "analysis_mask": paths["analysis_mask"],
                "analysis_input_is_smoothed": self.smoothed.isChecked(),
                "chunk_rows": int(result.get("chunk_rows", 64)),
                "sg_window": self.sg_window.value(),
                "sg_polyorder": self.sg_order.value(),
            })
            v4 = result.setdefault("v4", {})
            v4.update({
                "project_name": self.project_name.text().strip() or "CoreSpec_Project",
                "data_physics": self.physics.currentText(),
                "spectral_library_root": paths["spectral_library_root"],
                "requested_minerals": selected,
            })
            preprocessing = dict(v4.get("preprocessing", {}))
            preprocessing.update({"sg_window": self.sg_window.value(), "sg_polyorder": self.sg_order.value()})
            v4["preprocessing"] = preprocessing
            sampling = dict(v4.get("sampling", {}))
            sampling.update({
                "blocks": self.sample_blocks.value(),
                "block_rows": self.block_rows.value(),
                "minimum_valid_pixels_per_block": int(sampling.get("minimum_valid_pixels_per_block", 128)),
                "minimum_column_samples": int(sampling.get("minimum_column_samples", 20)),
            })
            v4["sampling"] = sampling
            library = dict(v4.get("library_ensemble", {}))
            library.update({
                "maximum_representatives_per_mineral": self.max_references.value(),
                "dedup_angle_rad": float(library.get("dedup_angle_rad", 0.02)),
            })
            v4["library_ensemble"] = library
            desktop = result.setdefault("desktop", {})
            desktop["output_root"] = self.output_path.text()
            return result

        def _start_audit(self) -> None:
            try:
                config = self._config()
            except Exception as exc:
                QMessageBox.warning(self, "CoreSpec Mapper", str(exc))
                return
            self._start_worker(_Worker("audit", config))
            self.stage_label.setText("Auditing")

        def _start_run(self) -> None:
            try:
                config = self._config()
                output = self.output_path.text()
                if not output:
                    raise ValueError("Select an output root")
            except Exception as exc:
                QMessageBox.warning(self, "CoreSpec Mapper", str(exc))
                return
            self._start_worker(_Worker("run", config, output))
            self.stage_label.setText("Running")

        def _start_worker(self, worker: _Worker) -> None:
            if self.worker is not None and self.worker.isRunning():
                return
            self.worker = worker
            worker.progress_event.connect(self._on_progress)
            worker.completed.connect(self._on_completed)
            worker.cancelled.connect(self._on_cancelled)
            worker.failed.connect(self._on_failed)
            worker.finished.connect(self._on_worker_finished)
            self._set_busy(True)
            self.progress_bar.setValue(0)
            self.timing_label.setText("Elapsed 00:00  |  Remaining --:--")
            self.log.clear()
            worker.start()

        def _on_progress(self, event: dict[str, Any]) -> None:
            self.progress_bar.setValue(int(float(event["overall_fraction"]) * 1000))
            self.stage_label.setText(f"{event['stage']}: {event['message']}")
            self.timing_label.setText(
                f"Elapsed {_format_duration(event.get('elapsed_seconds'))}  |  "
                f"Remaining {_format_duration(event.get('estimated_remaining_seconds'))}"
            )
            self.log.append(f"{float(event['overall_fraction']) * 100:6.2f}%  {event['message']}")

        def _on_completed(self, result: object) -> None:
            self._set_busy(False)
            self.progress_bar.setValue(1000)
            self.timing_label.setText(self.timing_label.text().split("  |", 1)[0] + "  |  Remaining 00:00")
            if isinstance(result, dict) and "run_directory" in result:
                quality = result["quality"]
                self._present_run(
                    Path(str(result["run_directory"])),
                    quality,
                    result["manifest"].get("previews", {}),
                )
                self.stage_label.setText(f"Completed: quality {quality['quality_grade']}")
            elif isinstance(result, dict):
                self._apply_audit(result)
                self.stage_label.setText("Audit complete")
                self.tabs.setCurrentIndex(1)

        def _on_cancelled(self, message: str) -> None:
            self._set_busy(False)
            self.stage_label.setText("Cancelled")
            self.timing_label.setText(self.timing_label.text().split("  |", 1)[0] + "  |  Remaining --:--")
            self.log.append(message or "Task cancelled")

        def _on_worker_finished(self) -> None:
            if self._close_when_idle:
                self._close_when_idle = False
                self.close()

        def _apply_audit(self, result: dict[str, Any]) -> None:
            support = result.get("mineral_support", {})
            for mineral_id, row in self.mineral_rows.items():
                record = support.get(mineral_id)
                if record is None:
                    continue
                level = record["level"]
                self.mineral_table.item(row, 3).setText(level.title())
                self.mineral_table.item(row, 4).setText("; ".join(record.get("reasons", [])))
                use = self.mineral_table.item(row, 0)
                if level == "unsupported":
                    use.setCheckState(Qt.CheckState.Unchecked)
                    use.setFlags(Qt.ItemFlag.ItemIsUserCheckable)
                else:
                    use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            library = result.get("library") or {}
            self.log.append(f"Selected references: {library.get('selected_spectra', 'not scanned')}")

        def _on_failed(self, message: str) -> None:
            self._set_busy(False)
            self.stage_label.setText("Failed")
            self.log.setPlainText(message)
            QMessageBox.critical(self, "CoreSpec Mapper", message.split("\n", 1)[0])

        def _cancel(self) -> None:
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel()
                self.stage_label.setText("Cancelling at the next block boundary")

        def _set_busy(self, busy: bool) -> None:
            self.audit_action.setEnabled(not busy)
            self.run_action.setEnabled(not busy)
            self.audit_button.setEnabled(not busy)
            self.run_button.setEnabled(not busy)
            self.cancel_action.setEnabled(busy)
            self.cancel_button.setEnabled(busy)

        def _show_preview(self, key: str) -> None:
            path = self.preview_paths.get(key)
            if not path:
                return
            pixmap = QPixmap(path)
            if pixmap.isNull():
                self.preview_pixmap = None
                self.preview_label.setText(f"Preview is unavailable:\n{path}")
                return
            self.preview_pixmap = pixmap
            self.preview_label.setText("")
            self.preview_label.setPixmap(pixmap)
            self.preview_label.adjustSize()

        def _navigate_step(self, row: int) -> None:
            if row == 3:
                self.tabs.setCurrentIndex(1)
            elif row >= 5:
                self.tabs.setCurrentIndex(2)
            else:
                self.tabs.setCurrentIndex(0)

        def _sync_step_to_tab(self, index: int) -> None:
            expected = {0: range(0, 3), 1: range(3, 4), 2: range(5, 7)}
            if self.steps.currentRow() not in expected.get(index, ()):
                self.steps.setCurrentRow({0: 0, 1: 3, 2: 6}.get(index, 0))

        def _present_run(
            self,
            run_directory: Path,
            quality: dict[str, Any],
            previews: dict[str, Any],
        ) -> None:
            run = run_directory.expanduser().resolve()
            resolved: dict[str, str] = {}
            for key, value in previews.items():
                configured = Path(str(value))
                candidates = [configured] if configured.is_absolute() else [Path.cwd() / configured, run / configured]
                candidates.append(run / "previews" / configured.name)
                selected = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
                resolved[str(key)] = str(selected.resolve())
            self.current_run_directory = str(run)
            self.quality_label.setText(
                f"Quality {quality.get('quality_grade', 'Unknown')}  |  {quality.get('status', 'Unknown')}"
            )
            self.preview_paths = resolved
            self.preview_combo.clear()
            self.preview_combo.addItems(sorted(self.preview_paths))
            preferred = "comparison_balanced" if "comparison_balanced" in resolved else self.preview_combo.currentText()
            if preferred:
                self.preview_combo.setCurrentText(preferred)
                self._show_preview(preferred)
            self.open_output_button.setEnabled(True)
            self.tabs.setCurrentIndex(2)

        def load_run_directory(self, path: str | Path) -> Path:
            run = Path(path).expanduser().resolve()
            manifest_path = run / "run_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"V4 run manifest was not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            quality = manifest.get("quality")
            if not isinstance(quality, dict):
                report = run / "reports" / "quality_report.json"
                quality = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
            previews = manifest.get("previews", {})
            if not isinstance(previews, dict) or not previews:
                raise ValueError("The selected V4 run does not contain preview records")
            self._present_run(run, quality, previews)
            self.stage_label.setText(f"Opened run: {run.name}")
            return run

        def _open_run_dialog(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Open V4 run", self.current_run_directory or "")
            if not path:
                return
            try:
                self.load_run_directory(path)
            except Exception as exc:
                QMessageBox.critical(self, "CoreSpec Mapper", str(exc))

        def _open_output(self) -> None:
            if not self.current_run_directory:
                return
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_run_directory))

        def save_config_file(self, path: str | Path) -> Path:
            output = Path(path)
            config = self._config()
            output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            self._base_config = deepcopy(config)
            return output

        def load_config_file(self, path: str | Path) -> dict[str, Any]:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
            self._apply_config(config)
            self._base_config = deepcopy(config)
            return config

        def _apply_config(self, config: dict[str, Any]) -> None:
            v4 = config.get("v4", {})
            self.image_path.setText(str(config.get("analysis_image", "")))
            self.mask_path.setText(str(config.get("analysis_mask", "")))
            self.library_path.setText(str(v4.get("spectral_library_root", "")))
            self.output_path.setText(str(config.get("desktop", {}).get("output_root", "")))
            self.project_name.setText(str(v4.get("project_name", "CoreSpec_Project")))
            self.smoothed.setChecked(bool(config.get("analysis_input_is_smoothed", False)))
            physics_index = self.physics.findText(str(v4.get("data_physics", "reflectance")))
            if physics_index >= 0:
                self.physics.setCurrentIndex(physics_index)
            preprocessing = v4.get("preprocessing", {})
            self.sg_window.setValue(int(preprocessing.get("sg_window", config.get("sg_window", 11))))
            self.sg_order.setValue(int(preprocessing.get("sg_polyorder", config.get("sg_polyorder", 2))))
            sampling = v4.get("sampling", {})
            self.sample_blocks.setValue(int(sampling.get("blocks", 12)))
            self.block_rows.setValue(int(sampling.get("block_rows", 64)))
            library = v4.get("library_ensemble", {})
            self.max_references.setValue(int(library.get("maximum_representatives_per_mineral", 6)))
            configured = v4.get("requested_minerals")
            selected = (
                set(configured)
                if configured is not None
                else {
                    mineral.mineral_id
                    for mineral in self.catalog.minerals.values()
                    if mineral.support_level == "validated_swir"
                }
            )
            for mineral_id, row in self.mineral_rows.items():
                item = self.mineral_table.item(row, 0)
                if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                    item.setCheckState(Qt.CheckState.Checked if mineral_id in selected else Qt.CheckState.Unchecked)

        def _save_config(self) -> None:
            try:
                config = self._config()
            except Exception as exc:
                QMessageBox.warning(self, "CoreSpec Mapper", str(exc))
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save V4 configuration", "corespec_v4.json", "JSON (*.json)")
            if path:
                self.save_config_file(path)

        def _open_config(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Open V4 configuration", "", "JSON (*.json)")
            if not path:
                return
            try:
                self.load_config_file(path)
            except Exception as exc:
                QMessageBox.critical(self, "CoreSpec Mapper", str(exc))

        def closeEvent(self, event) -> None:
            if self.worker is not None and self.worker.isRunning():
                answer = QMessageBox.question(
                    self,
                    "CoreSpec Mapper",
                    "Cancel the current task and close CoreSpec Mapper when it reaches a safe checkpoint?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self._close_when_idle = True
                    self._cancel()
                event.ignore()
                return
            event.accept()


def launch_desktop() -> int:
    if QApplication is None:
        raise RuntimeError("PySide6 is required for the CoreSpec Mapper desktop application")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CoreSpec Mapper")
    app.setOrganizationName("CoreSpec")
    app.setWindowIcon(QIcon(_icon_path()))
    window = CoreSpecMainWindow()
    window.show()
    return app.exec()


def main() -> int:
    return launch_desktop()


if __name__ == "__main__":
    raise SystemExit(main())
