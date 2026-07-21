from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Sequence
import json
import math
import sys
import traceback

import numpy as np

from .envi import EnviDataset
from .spectral_db import V5SpectralDatabase
from .spectral_v5 import load_v5_catalog
from .v4_models import CancellationToken, RunCancelled


try:  # The V5 service is intentionally allowed to land independently.
    from .v5_service import audit_v5_project, run_v5_project
except ImportError:  # pragma: no cover - resolved lazily or injected in tests
    audit_v5_project = None
    run_v5_project = None


try:
    from PySide6.QtCore import QSignalBlocker, Qt, QThread, QTimer, QUrl, Signal
    from PySide6.QtGui import QAction, QColor, QCloseEvent, QDesktopServices, QIcon, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QStyle,
        QStyleOptionButton,
        QStyleOptionGroupBox,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QToolButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - desktop is an optional dependency
    QApplication = None


ServiceFunction = Callable[..., Any]


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


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, Mapping) else {"value": converted}
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {"value": value}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_value(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    if isinstance(value, Mapping):
        return "；".join(
            f"{key}: {_display_value(item, precision=precision)}" for key, item in value.items()
        ) or "—"
    if isinstance(value, (list, tuple)):
        return ", ".join(_display_value(item, precision=precision) for item in value)
    return str(value)


def _progress_record(event: Any) -> dict[str, Any]:
    record = _to_mapping(event)
    fraction = record.get("overall_fraction", record.get("fraction"))
    if fraction is None and record.get("overall_percent") is not None:
        fraction = float(record["overall_percent"]) / 100.0
    record["overall_fraction"] = max(0.0, min(1.0, float(fraction or 0.0)))
    return record


def _resolve_v5_services() -> tuple[ServiceFunction | None, ServiceFunction | None]:
    global audit_v5_project, run_v5_project
    if audit_v5_project is not None and run_v5_project is not None:
        return audit_v5_project, run_v5_project
    try:
        service = import_module("corespec_mapper.v5_service")
    except ImportError:
        return audit_v5_project, run_v5_project
    audit_v5_project = getattr(service, "audit_v5_project", audit_v5_project)
    run_v5_project = getattr(service, "run_v5_project", run_v5_project)
    return audit_v5_project, run_v5_project


if QApplication is not None:
    class _V5Worker(QThread):
        progress_event = Signal(dict)
        completed = Signal(object)
        cancelled = Signal(str)
        failed = Signal(str)

        def __init__(
            self,
            action: str,
            config: dict[str, Any],
            output_root: str | None = None,
            *,
            audit_function: ServiceFunction | None = None,
            run_function: ServiceFunction | None = None,
        ) -> None:
            super().__init__()
            if action not in {"audit", "run"}:
                raise ValueError(f"Unsupported V5 worker action: {action}")
            self.action = action
            self.config = deepcopy(config)
            self.output_root = output_root
            self.audit_function = audit_function
            self.run_function = run_function
            self.token = CancellationToken()

        def cancel(self) -> None:
            self.token.cancel()

        def _emit_progress(self, event: Any) -> None:
            self.progress_event.emit(_progress_record(event))

        def run(self) -> None:
            audit_default, run_default = _resolve_v5_services()
            if self.action == "audit":
                service = self.audit_function or audit_default
            else:
                service = self.run_function or run_default
            if service is None:
                self.failed.emit("V5 服务尚未安装：缺少 audit_v5_project/run_v5_project。")
                return
            try:
                if self.action == "audit":
                    result = service(
                        self.config,
                        progress=self._emit_progress,
                        cancel_token=self.token,
                    )
                else:
                    if not self.output_root:
                        raise ValueError("运行任务缺少输出目录")
                    result = service(
                        self.config,
                        self.output_root,
                        progress=self._emit_progress,
                        cancel_token=self.token,
                    )
                self.completed.emit(result)
            except RunCancelled as exc:
                self.cancelled.emit(str(exc))
            except Exception as exc:  # pragma: no cover - exercised by integration failures
                self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


    class _PathPicker(QWidget):
        path_changed = Signal(str)

        def __init__(
            self,
            *,
            directory: bool = False,
            optional: bool = False,
            filter_text: str = "ENVI 数据 (*.dat *.img *.hdr);;所有文件 (*)",
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.directory = directory
            self.optional = optional
            self.filter_text = filter_text
            self.edit = QLineEdit()
            self.edit.setClearButtonEnabled(True)
            self.edit.setPlaceholderText("可选" if optional else "请选择文件")
            self.button = QToolButton()
            self.button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
            self.button.setToolTip("浏览")
            self.button.clicked.connect(self._browse)
            self.edit.textChanged.connect(self.path_changed)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(self.edit, 1)
            layout.addWidget(self.button)

        def _browse(self) -> None:
            if self.directory:
                value = QFileDialog.getExistingDirectory(self, "选择目录", self.text())
            else:
                value, _ = QFileDialog.getOpenFileName(self, "选择数据", self.text(), self.filter_text)
            if value:
                self.setText(value)

        def text(self) -> str:
            return self.edit.text().strip()

        def setText(self, value: str | Path | None) -> None:
            self.edit.setText("" if value is None else str(value))


    class _StatusPill(QLabel):
        _COLORS = {
            "neutral": ("#E8EEF5", "#41566D"),
            "good": ("#DDF5E8", "#176B45"),
            "warning": ("#FFF0CF", "#8A5B00"),
            "bad": ("#FCE2E4", "#A22A36"),
            "active": ("#DDEBFF", "#1558A6"),
        }

        def __init__(self, text: str = "待审计", state: str = "neutral", parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setMinimumWidth(86)
            self.set_state(text, state)

        def set_state(self, text: str, state: str = "neutral") -> None:
            background, foreground = self._COLORS.get(state, self._COLORS["neutral"])
            self.setText(text)
            self.setStyleSheet(
                f"QLabel {{ background:{background}; color:{foreground}; border-radius:10px; "
                "padding:4px 10px; font-weight:600; }}"
            )


    def _paint_visible_check_mark(widget: QWidget, rect) -> None:
        """Overlay a high-contrast tick when the platform style omits it."""

        if not rect.isValid() or rect.width() < 8 or rect.height() < 8:
            return
        painter = QPainter(widget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#1677B8" if widget.isEnabled() else "#8293A3")
        pen = QPen(color, max(2.0, min(rect.width(), rect.height()) / 6.5))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        x1 = rect.left() + round(rect.width() * 0.20)
        y1 = rect.top() + round(rect.height() * 0.52)
        x2 = rect.left() + round(rect.width() * 0.43)
        y2 = rect.top() + round(rect.height() * 0.75)
        x3 = rect.left() + round(rect.width() * 0.82)
        y3 = rect.top() + round(rect.height() * 0.25)
        painter.drawLine(x1, y1, x2, y2)
        painter.drawLine(x2, y2, x3, y3)


    class _VisibleCheckBox(QCheckBox):
        """QCheckBox with a deterministic checked-state mark on Windows."""

        def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
            super().paintEvent(event)
            if not self.isChecked():
                return
            option = QStyleOptionButton()
            self.initStyleOption(option)
            indicator = self.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, self)
            _paint_visible_check_mark(self, indicator)


    class _VisibleCheckableGroupBox(QGroupBox):
        """Checkable group box that keeps its title checkbox visibly checked."""

        def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
            super().paintEvent(event)
            if not self.isCheckable() or not self.isChecked():
                return
            option = QStyleOptionGroupBox()
            self.initStyleOption(option)
            indicator = self.style().subControlRect(
                QStyle.ComplexControl.CC_GroupBox,
                option,
                QStyle.SubControl.SC_GroupBoxCheckBox,
                self,
            )
            _paint_visible_check_mark(self, indicator)


    class SpectrumPlot(QWidget):
        """Small dependency-free spectral plot used by the evidence review page."""

        PALETTE = ("#177DDC", "#16A085", "#9B59B6", "#E67E22", "#D64545", "#2E86AB")
        LINE_STYLES = (
            Qt.PenStyle.SolidLine,
            Qt.PenStyle.DashLine,
            Qt.PenStyle.DotLine,
            Qt.PenStyle.DashDotLine,
            Qt.PenStyle.DashDotDotLine,
        )

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.curves: list[dict[str, Any]] = []
            self.highlighted_index: int | None = None
            self.setMinimumHeight(280)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setToolTip("审计后显示实际入选标准谱；曲线仅用于证据审查，不代表丰度。")

        def set_curves(self, curves: Iterable[Mapping[str, Any]]) -> None:
            parsed: list[dict[str, Any]] = []
            for index, curve in enumerate(curves):
                wavelengths = curve.get("wavelengths_nm", curve.get("wavelengths", curve.get("x", ())))
                values = curve.get("values", curve.get("reflectance", curve.get("y", ())))
                try:
                    x = [float(item) for item in wavelengths]
                    y = [float(item) for item in values]
                except (TypeError, ValueError):
                    continue
                pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
                if len(pairs) < 2:
                    continue
                parsed.append(
                    {
                        "name": str(curve.get("name", curve.get("raw_name", f"Reference {index + 1}"))),
                        "mineral_id": str(curve.get("mineral_id", curve.get("primary_phase_id", ""))),
                        "x": [item[0] for item in pairs],
                        "y": [item[1] for item in pairs],
                        "color": str(curve.get("color", self.PALETTE[index % len(self.PALETTE)])),
                        "line_style": self.LINE_STYLES[(index // len(self.PALETTE)) % len(self.LINE_STYLES)],
                    }
                )
            self.curves = parsed
            self.highlighted_index = 0 if parsed else None
            self.update()

        def set_highlighted_curve(self, index: int) -> None:
            self.highlighted_index = index if 0 <= index < len(self.curves) else None
            self.update()

        def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor("#FFFFFF"))
            left, top, right, bottom = 64, 46, 24, 48
            plot = self.rect().adjusted(left, top, -right, -bottom)
            painter.setPen(QPen(QColor("#DCE4EC"), 1))
            for index in range(6):
                x = plot.left() + round(plot.width() * index / 5)
                painter.drawLine(x, plot.top(), x, plot.bottom())
            for index in range(5):
                y = plot.top() + round(plot.height() * index / 4)
                painter.drawLine(plot.left(), y, plot.right(), y)
            painter.setPen(QPen(QColor("#7A8A9A"), 1))
            painter.drawRect(plot)
            if not self.curves:
                painter.setPen(QColor("#7A8A9A"))
                painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "完成项目审计后显示入选标准谱")
                return
            selected_name = ""
            if self.highlighted_index is not None:
                selected_name = self.curves[self.highlighted_index]["name"]
            painter.setPen(QColor("#526273"))
            header = f"实际入选参考谱：{len(self.curves)} 条"
            if selected_name:
                header += f"  ·  当前高亮：{selected_name}"
            painter.drawText(left, 28, self.fontMetrics().elidedText(header, Qt.TextElideMode.ElideRight, max(1, plot.width())))
            xs = [value for curve in self.curves for value in curve["x"]]
            ys = [value for curve in self.curves for value in curve["y"]]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            if xmax <= xmin:
                xmax = xmin + 1.0
            if ymax <= ymin:
                ymax = ymin + 1.0
            padding = 0.04 * (ymax - ymin)
            ymin -= padding
            ymax += padding
            for index, curve in enumerate(self.curves):
                highlighted = index == self.highlighted_index
                pen = QPen(QColor(curve["color"]), 3.2 if highlighted else 1.8)
                pen.setStyle(curve["line_style"])
                painter.setPen(pen)
                points = []
                for xvalue, yvalue in zip(curve["x"], curve["y"]):
                    px = plot.left() + (xvalue - xmin) / (xmax - xmin) * plot.width()
                    py = plot.bottom() - (yvalue - ymin) / (ymax - ymin) * plot.height()
                    points.append((round(px), round(py)))
                for first, second in zip(points, points[1:]):
                    painter.drawLine(first[0], first[1], second[0], second[1])
                if highlighted and points:
                    marker_step = max(1, len(points) // 14)
                    for px, py in points[::marker_step]:
                        painter.drawEllipse(px - 2, py - 2, 4, 4)
            painter.setPen(QColor("#526273"))
            painter.drawText(plot.left(), self.height() - 15, f"{xmin:.0f} nm")
            painter.drawText(plot.right() - 72, self.height() - 15, f"{xmax:.0f} nm")
            painter.save()
            painter.translate(18, plot.center().y() + 32)
            painter.rotate(-90)
            painter.drawText(0, 0, "反射率 / 归一化证据")
            painter.restore()


    class CoreSpecV5Window(QMainWindow):
        STEP_TITLES = (
            ("项目", "Project", "建立非覆盖式项目与输出边界"),
            ("数据", "Data", "选择主分析立方体并发现同批次多源影像"),
            ("掩膜", "Mask", "自动提取岩心或使用外部同网格掩膜"),
            ("矿物", "Minerals", "按一级矿物族和二级矿物选择目标"),
            ("标准谱", "References", "查看并确认本项目实际入选的标准光谱"),
            ("模式", "Mode", "默认自动平衡，必要时展开高级控制"),
            ("阈值试算", "Threshold Trial", "自动优选三档阈值，支持复核、调整与重算"),
            ("运行", "Run", "确认审计与掩膜批准状态后执行全景识别"),
            ("结果", "Results", "查看三档结果、质量状态与日志"),
        )

        def __init__(
            self,
            *,
            audit_function: ServiceFunction | None = None,
            run_function: ServiceFunction | None = None,
            catalog: Mapping[str, Any] | None = None,
            reference_counts: Mapping[str, int] | None = None,
        ) -> None:
            super().__init__()
            self.catalog = deepcopy(dict(catalog)) if catalog is not None else load_v5_catalog()
            self.reference_counts = dict(reference_counts) if reference_counts is not None else self._load_reference_counts()
            self.audit_function = audit_function
            self.run_function = run_function
            self.worker: _V5Worker | None = None
            self._base_config: dict[str, Any] = {}
            self._close_when_idle = False
            self.last_audit: dict[str, Any] = {}
            self.current_thresholds: Any = None
            self.current_run_directory: Path | None = None
            self.preview_paths: dict[str, str] = {}
            self.preview_pixmap: QPixmap | None = None
            self.step_states = ["pending"] * len(self.STEP_TITLES)
            self._last_progress_fraction = 0.0
            self._last_remaining_seconds: float | None = None
            self._task_started_at: float | None = None
            self._eta_seconds_per_fraction: float | None = None
            self._audit_target_step: int | None = None
            self._threshold_editors: dict[tuple[str, str], tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
            self._threshold_trial_dirty = False
            self.setWindowTitle("CoreSpec Mapper V5.3")
            self.setWindowIcon(QIcon(_icon_path()))
            self.resize(1480, 920)
            self.setMinimumSize(1120, 720)
            self._build_actions()
            self._build_ui()
            self._apply_style()
            self._populate_minerals()
            self._set_step(0)
            self._set_busy(False)

        @staticmethod
        def _load_reference_counts() -> dict[str, int]:
            try:
                with V5SpectralDatabase() as database:
                    return database.candidate_counts()
            except (FileNotFoundError, ValueError, OSError):
                return {}

        def _build_actions(self) -> None:
            toolbar = self.addToolBar("项目工具")
            toolbar.setMovable(False)
            self.open_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "打开配置", self)
            self.save_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "保存配置", self)
            self.audit_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView), "项目审计", self)
            self.run_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "运行 V5", self)
            self.cancel_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop), "安全取消", self)
            self.open_action.triggered.connect(self._open_config_dialog)
            self.save_action.triggered.connect(self._save_config_dialog)
            self.audit_action.triggered.connect(self._start_audit)
            self.run_action.triggered.connect(self._start_run)
            self.cancel_action.triggered.connect(self._cancel)
            toolbar.addActions((self.open_action, self.save_action))
            toolbar.addSeparator()
            toolbar.addActions((self.audit_action, self.run_action, self.cancel_action))

        def _build_ui(self) -> None:
            central = QWidget()
            outer = QVBoxLayout(central)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            body = QSplitter(Qt.Orientation.Horizontal)
            body.setChildrenCollapsible(False)
            body.addWidget(self._build_sidebar())
            body.addWidget(self._build_workspace())
            body.setStretchFactor(0, 0)
            body.setStretchFactor(1, 1)
            outer.addWidget(body, 1)
            outer.addWidget(self._build_status_bar())
            self.setCentralWidget(central)

        def _build_sidebar(self) -> QWidget:
            sidebar = QFrame()
            sidebar.setObjectName("sidebar")
            sidebar.setMinimumWidth(224)
            sidebar.setMaximumWidth(252)
            layout = QVBoxLayout(sidebar)
            layout.setContentsMargins(22, 22, 18, 18)
            layout.setSpacing(14)
            brand = QHBoxLayout()
            logo = QLabel()
            logo.setObjectName("brandLogo")
            pixmap = QPixmap(_logo_path())
            if pixmap.isNull():
                logo.setText("CS")
            else:
                logo.setPixmap(pixmap.scaled(54, 54, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            name = QLabel("CoreSpec\nMapper 5.3")
            name.setObjectName("brandName")
            brand.addWidget(logo)
            brand.addWidget(name, 1)
            layout.addLayout(brand)
            self.steps = QListWidget()
            self.steps.setObjectName("stepList")
            self.steps.setSpacing(3)
            self.steps.setWordWrap(True)
            self.steps.setTextElideMode(Qt.TextElideMode.ElideNone)
            self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            for index, (zh, en, _) in enumerate(self.STEP_TITLES, 1):
                self.steps.addItem(f"{index:02d}   {zh}\n       {en}")
            self.steps.currentRowChanged.connect(self._set_step)
            layout.addWidget(self.steps, 1)
            database_note = QLabel("标准谱库已就绪\n1,783 条测量 · 16 类常用目标")
            database_note.setObjectName("databaseNote")
            database_note.setWordWrap(True)
            layout.addWidget(database_note)
            return sidebar

        def _build_workspace(self) -> QWidget:
            workspace = QFrame()
            workspace.setObjectName("workspace")
            layout = QVBoxLayout(workspace)
            layout.setContentsMargins(28, 22, 28, 18)
            layout.setSpacing(14)
            header = QHBoxLayout()
            labels = QVBoxLayout()
            self.page_title = QLabel()
            self.page_title.setObjectName("pageTitle")
            self.page_subtitle = QLabel()
            self.page_subtitle.setObjectName("pageSubtitle")
            labels.addWidget(self.page_title)
            labels.addWidget(self.page_subtitle)
            self.project_state = _StatusPill("草稿", "neutral")
            header.addLayout(labels, 1)
            header.addWidget(self.project_state)
            layout.addLayout(header)
            self.pages = QStackedWidget()
            self.pages.setObjectName("v5Pages")
            self.pages.addWidget(self._page_project())
            self.pages.addWidget(self._page_data())
            self.pages.addWidget(self._page_mask())
            self.pages.addWidget(self._page_minerals())
            self.pages.addWidget(self._page_spectra())
            self.pages.addWidget(self._page_mode())
            self.pages.addWidget(self._page_threshold_trial())
            self.pages.addWidget(self._page_run_control())
            self.pages.addWidget(self._page_results())
            layout.addWidget(self.pages, 1)
            navigation = QHBoxLayout()
            self.back_button = QPushButton("← 上一步")
            self.next_button = QPushButton("下一步 →")
            self.back_button.clicked.connect(lambda: self._set_step(self.pages.currentIndex() - 1))
            self.next_button.clicked.connect(self._advance_step)
            navigation.addWidget(self.back_button)
            navigation.addStretch(1)
            navigation.addWidget(self.next_button)
            layout.addLayout(navigation)
            return workspace

        def _build_status_bar(self) -> QWidget:
            bar = QFrame()
            bar.setObjectName("taskBar")
            layout = QHBoxLayout(bar)
            layout.setContentsMargins(18, 9, 18, 9)
            self.stage_label = QLabel("就绪")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(False)
            self.timing_label = QLabel("已用 00:00  ·  剩余 估算中")
            self.cancel_button = QToolButton()
            self.cancel_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.cancel_button.setToolTip("在安全检查点取消")
            self.cancel_button.clicked.connect(self._cancel)
            layout.addWidget(self.stage_label)
            layout.addWidget(self.progress_bar, 1)
            layout.addWidget(self.timing_label)
            layout.addWidget(self.cancel_button)
            return bar

        @staticmethod
        def _page_shell(introduction: str) -> tuple[QWidget, QVBoxLayout]:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(14)
            lead = QLabel(introduction)
            lead.setObjectName("pageLead")
            lead.setWordWrap(True)
            layout.addWidget(lead)
            return page, layout

        def _page_project(self) -> QWidget:
            page, layout = self._page_shell("每次运行创建独立 run_id，不覆盖输入与历史成果。")
            box = QGroupBox("项目身份与成果位置")
            form = QFormLayout(box)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            self.project_name = QLineEdit("CoreSpec_Project")
            self.project_description = QLineEdit()
            self.project_description.setPlaceholderText("可选：钻孔、深度段或批次说明")
            self.output_path = _PathPicker(directory=True)
            form.addRow("项目名称", self.project_name)
            form.addRow("项目说明", self.project_description)
            form.addRow("输出根目录", self.output_path)
            layout.addWidget(box)
            protection = QFrame()
            protection.setObjectName("infoCard")
            protection_layout = QHBoxLayout(protection)
            protection_layout.addWidget(QLabel("✓"))
            text = QLabel("输入只读 · 运行成果独立保存 · 配置和审计快照可追溯")
            protection_layout.addWidget(text, 1)
            layout.addWidget(protection)
            layout.addStretch(1)
            return page

        def _page_data(self) -> QWidget:
            page, layout = self._page_shell(
                "以实际波长和数据物理判断能力；主分析立方体驱动像元级识别。RGB/NIR/SWIR 伴随文件用于能力与来源审计，"
                "不会在未配准时强行逐像元融合；联合分析应选择已配准的连续立方体作为主数据。"
            )
            primary = QGroupBox("主分析数据")
            form = QFormLayout(primary)
            self.analysis_path = _PathPicker()
            self.analysis_domain = QComboBox()
            self.analysis_domain.addItem("自动判断", "auto")
            self.analysis_domain.addItem("SWIR 反射率", "swir")
            self.analysis_domain.addItem("NIR/VNIR 反射率", "nir")
            self.analysis_domain.addItem("TIR 发射率", "tir")
            self.data_physics = QComboBox()
            self.data_physics.addItem("反射率 Reflectance", "reflectance")
            self.data_physics.addItem("发射率 Emissivity", "emissivity")
            self.data_physics.addItem("辐亮度 Radiance", "radiance")
            self.input_smoothed = _VisibleCheckBox("输入已经完成光谱 SG 平滑")
            self.auto_discover = _VisibleCheckBox("自动发现同目录 RGB / NIR / SWIR")
            self.auto_discover.setChecked(True)
            self.analysis_path.edit.editingFinished.connect(self._discover_companions)
            form.addRow("主分析立方体", self.analysis_path)
            form.addRow("波段域", self.analysis_domain)
            form.addRow("数据物理", self.data_physics)
            form.addRow("预处理", self.input_smoothed)
            form.addRow("同批次数据", self.auto_discover)
            layout.addWidget(primary)
            companions = QGroupBox("可选多源影像")
            companion_form = QFormLayout(companions)
            self.rgb_path = _PathPicker(optional=True)
            self.nir_path = _PathPicker(optional=True)
            self.swir_path = _PathPicker(optional=True)
            companion_form.addRow("RGB", self.rgb_path)
            companion_form.addRow("NIR / VNIR", self.nir_path)
            companion_form.addRow("SWIR", self.swir_path)
            layout.addWidget(companions)
            self.input_table = QTableWidget(0, 5)
            self.input_table.setHorizontalHeaderLabels(("数据", "尺寸", "波段/范围", "物理", "状态"))
            self.input_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.input_table.setMaximumHeight(176)
            layout.addWidget(self.input_table)
            self.data_feedback = QLabel("选择数据后，这里会立即显示已读取的数据与处理状态。")
            self.data_feedback.setObjectName("pageLead")
            self.data_feedback.setWordWrap(True)
            layout.addWidget(self.data_feedback)
            for picker in (self.analysis_path, self.rgb_path, self.nir_path, self.swir_path):
                picker.path_changed.connect(self._refresh_selected_inputs)
            self.input_smoothed.toggled.connect(self._on_data_option_changed)
            self.auto_discover.toggled.connect(self._on_data_option_changed)
            self.analysis_domain.currentIndexChanged.connect(self._refresh_selected_inputs)
            self.data_physics.currentIndexChanged.connect(self._refresh_selected_inputs)
            return page

        def _page_mask(self) -> QWidget:
            page, layout = self._page_shell(
                "自动模式使用应用内置的智能岩心前景模型。自动或外部掩膜都必须先预览、核对质量并人工批准。"
            )
            options = QGroupBox("掩膜来源")
            grid = QGridLayout(options)
            self.mask_group = QButtonGroup(self)
            self.mask_auto = QRadioButton("自动材料掩膜")
            self.mask_external = QRadioButton("外部 ENVI 掩膜")
            self.mask_auto.setChecked(True)
            self.mask_group.addButton(self.mask_auto)
            self.mask_group.addButton(self.mask_external)
            self.mask_path = _PathPicker(optional=True)
            self.mask_path.setEnabled(False)
            self.mask_external.toggled.connect(self._on_mask_mode_changed)
            self.mask_path.path_changed.connect(self._preview_external_mask)
            self.mask_min_component = QSpinBox()
            self.mask_min_component.setRange(1, 100000)
            self.mask_min_component.setValue(64)
            self.mask_edge_guard = QSpinBox()
            self.mask_edge_guard.setRange(0, 32)
            self.mask_edge_guard.setValue(2)
            self.mask_engine = QComboBox()
            self.mask_engine.addItem("智能岩心前景提取", "integrated_core_foreground_v2")
            self.mask_engine.setVisible(False)
            self.mask_model_state = _StatusPill("应用内置 · 已启用", "good")
            self.mask_interior_fraction = QSpinBox()
            self.mask_interior_fraction.setRange(0, 100)
            self.mask_interior_fraction.setValue(65)
            self.mask_interior_fraction.setSuffix(" %")
            self.mask_min_fraction = QSpinBox()
            self.mask_min_fraction.setRange(0, 100)
            self.mask_min_fraction.setValue(5)
            self.mask_min_fraction.setSuffix(" %")
            grid.addWidget(self.mask_auto, 0, 0)
            grid.addWidget(QLabel("自动分离岩心、托盘和无效背景"), 0, 1, 1, 2)
            grid.addWidget(self.mask_external, 1, 0)
            grid.addWidget(self.mask_path, 1, 1, 1, 2)
            grid.addWidget(QLabel("自动模型"), 2, 0)
            grid.addWidget(self.mask_model_state, 2, 1)
            model_note = QLabel("智能识别岩心实体；不再提供多引擎选择")
            model_note.setWordWrap(True)
            grid.addWidget(model_note, 2, 2, 1, 2)
            grid.addWidget(QLabel("最小连通域"), 3, 0)
            grid.addWidget(self.mask_min_component, 3, 1)
            grid.addWidget(QLabel("边缘保护像元"), 3, 2)
            grid.addWidget(self.mask_edge_guard, 3, 3)
            grid.addWidget(QLabel("内部填充率门槛"), 4, 0)
            grid.addWidget(self.mask_interior_fraction, 4, 1)
            grid.addWidget(QLabel("最小覆盖率"), 4, 2)
            grid.addWidget(self.mask_min_fraction, 4, 3)
            layout.addWidget(options)
            review = QSplitter(Qt.Orientation.Horizontal)
            preview_frame = QFrame()
            preview_frame.setObjectName("previewFrame")
            preview_layout = QVBoxLayout(preview_frame)
            status_row = QHBoxLayout()
            status_row.addWidget(QLabel("掩膜预览"))
            self.generate_mask_button = QPushButton("生成/刷新掩膜预览")
            self.generate_mask_button.clicked.connect(self._refresh_mask_preview)
            status_row.addWidget(self.generate_mask_button)
            self.mask_zoom = QComboBox()
            self.mask_zoom.addItem("适应窗口", 0.0)
            self.mask_zoom.addItem("50%", 0.5)
            self.mask_zoom.addItem("100%", 1.0)
            self.mask_zoom.addItem("200%", 2.0)
            self.mask_zoom.currentIndexChanged.connect(self._scale_mask_preview)
            status_row.addWidget(self.mask_zoom)
            status_row.addStretch(1)
            self.mask_status = _StatusPill()
            status_row.addWidget(self.mask_status)
            preview_layout.addLayout(status_row)
            self.mask_preview = QLabel("审计后显示掩膜覆盖范围")
            self.mask_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.mask_preview.setMinimumSize(440, 230)
            self.mask_preview.setObjectName("imageWell")
            self.mask_preview_pixmap: QPixmap | None = None
            self.mask_preview_scroll = QScrollArea()
            self.mask_preview_scroll.setWidgetResizable(False)
            self.mask_preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.mask_preview_scroll.setMinimumSize(440, 230)
            self.mask_preview_scroll.setWidget(self.mask_preview)
            preview_layout.addWidget(self.mask_preview_scroll, 1)
            review.addWidget(preview_frame)
            self.mask_metrics = QTableWidget(0, 2)
            self.mask_metrics.setHorizontalHeaderLabels(("质量指标", "值"))
            self.mask_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            review.addWidget(self.mask_metrics)
            review.setStretchFactor(0, 2)
            review.setStretchFactor(1, 1)
            layout.addWidget(review, 1)
            approval = QGroupBox("人工批准门")
            approval_form = QFormLayout(approval)
            self.mask_approved = _VisibleCheckBox("我已核对掩膜覆盖岩心实体，未见托盘轮廓化、标签大面积误入或明显漏提")
            self.mask_approved.toggled.connect(self._on_mask_approval_changed)
            self.mask_reviewer = QLineEdit()
            self.mask_reviewer.setPlaceholderText("审核人 / 试验批次")
            self.mask_notes = QLineEdit()
            self.mask_notes.setPlaceholderText("可选：修订、配准或人工编辑说明")
            approval_form.addRow("批准", self.mask_approved)
            approval_form.addRow("审核记录", self.mask_reviewer)
            approval_form.addRow("备注", self.mask_notes)
            layout.addWidget(approval)
            return page

        def _on_mask_approval_changed(self, approved: bool) -> None:
            if not hasattr(self, "run_page_start_button"):
                return
            idle = self.worker is None or not self.worker.isRunning()
            self.run_page_start_button.setEnabled(
                bool(approved)
                and idle
                and self.references_confirmed.isChecked()
                and not self._threshold_trial_dirty
            )
            self.run_mask_state.set_state(
                "已批准 · 运行时重新核验质量" if approved else "掩膜未批准",
                "warning" if approved else "neutral",
            )

        def _on_mask_mode_changed(self, external: bool) -> None:
            self.mask_path.setEnabled(external)
            self.mask_approved.setChecked(False)
            self.generate_mask_button.setText("刷新外部掩膜预览" if external else "生成/刷新掩膜预览")
            if external:
                self._preview_external_mask(self.mask_path.text())
            else:
                self.mask_preview_pixmap = None
                self.mask_preview.clear()
                self.mask_preview.setText("项目审计后显示自动提取的岩心范围")
                self.mask_preview.setMinimumSize(440, 230)
                self.mask_preview.resize(440, 230)
                self.mask_status.set_state("等待自动提取", "neutral")

        def _refresh_mask_preview(self) -> None:
            self.mask_approved.setChecked(False)
            if self.mask_external.isChecked():
                self._preview_external_mask(self.mask_path.text())
                return
            self.mask_status.set_state("正在生成", "active")
            self._start_audit(target_step=2)

        def _set_mask_preview_pixmap(self, pixmap: QPixmap) -> None:
            self.mask_preview_pixmap = pixmap
            self.mask_preview.clear()
            self._scale_mask_preview()

        def _scale_mask_preview(self, *_args) -> None:
            if self.mask_preview_pixmap is None or self.mask_preview_pixmap.isNull():
                return
            zoom = float(self.mask_zoom.currentData() or 0.0)
            if zoom <= 0.0:
                viewport = self.mask_preview_scroll.viewport().size()
                pixmap = self.mask_preview_pixmap.scaled(
                    max(1, viewport.width() - 12),
                    max(1, viewport.height() - 12),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            else:
                pixmap = self.mask_preview_pixmap.scaled(
                    max(1, round(self.mask_preview_pixmap.width() * zoom)),
                    max(1, round(self.mask_preview_pixmap.height() * zoom)),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self.mask_preview.setMinimumSize(pixmap.size())
            self.mask_preview.resize(pixmap.size())
            self.mask_preview.setPixmap(pixmap)

        def _preview_external_mask(self, value: str = "") -> None:
            if not hasattr(self, "mask_preview") or not self.mask_external.isChecked():
                return
            path = Path(value or self.mask_path.text())
            if not path.is_file():
                self.mask_preview_pixmap = None
                self.mask_preview.clear()
                self.mask_preview.setText("选择可读取的 ENVI 掩膜后立即显示预览")
                self.mask_preview.setMinimumSize(440, 230)
                self.mask_preview.resize(440, 230)
                self.mask_status.set_state("等待文件", "neutral")
                return
            try:
                with EnviDataset(path) as dataset:
                    values = np.asarray(dataset.read_rows(0, dataset.info.lines, bands=[0])[:, :, 0]).copy()
                mask = np.isfinite(values) & (np.abs(values) > 1e-10)
                rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                rgb[mask] = np.array([55, 188, 218], dtype=np.uint8)
                rgb[~mask] = np.array([24, 36, 48], dtype=np.uint8)
                image = QImage(
                    rgb.data,
                    rgb.shape[1],
                    rgb.shape[0],
                    int(rgb.strides[0]),
                    QImage.Format.Format_RGB888,
                ).copy()
                pixmap = QPixmap.fromImage(image)
                self._set_mask_preview_pixmap(pixmap)
                fraction = float(np.mean(mask))
                self.mask_status.set_state(f"已预览 · {fraction:.1%}", "warning")
                self.mask_metrics.setRowCount(3)
                for row, (name, result) in enumerate((
                    ("掩膜尺寸", f"{mask.shape[0]} × {mask.shape[1]}"),
                    ("前景像元", int(np.count_nonzero(mask))),
                    ("覆盖比例", f"{fraction:.2%}"),
                )):
                    self.mask_metrics.setItem(row, 0, QTableWidgetItem(str(name)))
                    self.mask_metrics.setItem(row, 1, QTableWidgetItem(str(result)))
                self.mask_approved.setChecked(False)
            except (FileNotFoundError, OSError, ValueError) as exc:
                self.mask_preview_pixmap = None
                self.mask_preview.clear()
                self.mask_preview.setText(f"无法预览掩膜：{exc}")
                self.mask_preview.setMinimumSize(440, 230)
                self.mask_preview.resize(440, 230)
                self.mask_status.set_state("无法读取", "bad")

        def _page_minerals(self) -> QWidget:
            page, layout = self._page_shell("一级为矿物族，二级为可识别矿物；支持级别由传感器审计和内置参考谱共同决定。")
            splitter = QSplitter(Qt.Orientation.Horizontal)
            self.mineral_tree = QTreeWidget()
            self.mineral_tree.setColumnCount(4)
            self.mineral_tree.setHeaderLabels(("矿物族 / 矿物", "支持级别", "关键波段", "参考数"))
            self.mineral_tree.setAlternatingRowColors(True)
            self.mineral_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.mineral_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.mineral_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.mineral_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self.mineral_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            self.mineral_tree.currentItemChanged.connect(self._show_mineral_details)
            self.mineral_tree.itemChanged.connect(self._on_mineral_selection_changed)
            splitter.addWidget(self.mineral_tree)
            detail = QFrame()
            detail.setObjectName("detailCard")
            detail_layout = QVBoxLayout(detail)
            self.mineral_detail_title = QLabel("选择一个矿物查看证据要求")
            self.mineral_detail_title.setObjectName("detailTitle")
            self.mineral_detail_text = QLabel()
            self.mineral_detail_text.setWordWrap(True)
            self.mineral_detail_text.setAlignment(Qt.AlignmentFlag.AlignTop)
            detail_layout.addWidget(self.mineral_detail_title)
            detail_layout.addWidget(self.mineral_detail_text, 1)
            splitter.addWidget(detail)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            layout.addWidget(splitter, 1)
            return page

        def _page_mode(self) -> QWidget:
            page, layout = self._page_shell("Automatic Balanced 是默认生产模式；三档结果共享同一套光谱证据。")
            mode = QGroupBox("识别策略")
            form = QFormLayout(mode)
            self.execution_engine = QComboBox()
            self.execution_engine.addItem("V5.3 自适应矿物证据引擎", "adaptive_v5")
            self.execution_engine.setVisible(False)
            engine_state = _StatusPill("应用内置 · 自动执行", "good")
            self.profile = QComboBox()
            self.profile.addItem("平衡 · Balanced（推荐）", "balanced")
            self.profile.addItem("严格 · Conservative", "conservative")
            self.profile.addItem("敏感 · Sensitive", "sensitive")
            self.quick_calibration = _VisibleCheckBox("使用项目 ROI / 点位真值执行 Quick Calibration")
            self.quick_calibration.setEnabled(False)
            self.quick_calibration.setVisible(False)
            self.write_all_profiles = _VisibleCheckBox("同时输出严格、平衡、敏感三档")
            self.write_all_profiles.setChecked(True)
            self.write_all_profiles.setVisible(False)
            output_state = _StatusPill("三档成果已启用", "good")
            form.addRow("识别引擎", engine_state)
            form.addRow("默认结果", self.profile)
            form.addRow("成果输出", output_state)
            layout.addWidget(mode)
            self.advanced_group = _VisibleCheckableGroupBox("高级自动设置 · 展开后覆盖自动建议")
            self.advanced_group.setCheckable(True)
            self.advanced_group.setChecked(False)
            advanced_outer = QVBoxLayout(self.advanced_group)
            self.advanced_body = QWidget()
            advanced = QFormLayout(self.advanced_body)
            self.sg_window = QSpinBox(); self.sg_window.setRange(5, 31); self.sg_window.setSingleStep(2); self.sg_window.setValue(11)
            self.sg_order = QSpinBox(); self.sg_order.setRange(1, 5); self.sg_order.setValue(2)
            self.sample_blocks = QSpinBox(); self.sample_blocks.setRange(8, 16); self.sample_blocks.setValue(12)
            self.block_rows = QSpinBox(); self.block_rows.setRange(16, 512); self.block_rows.setValue(64)
            self.chunk_rows = QSpinBox(); self.chunk_rows.setRange(16, 1024); self.chunk_rows.setValue(64)
            self.max_references = QSpinBox(); self.max_references.setRange(1, 12); self.max_references.setValue(6)
            self.artifact_edge_width = QSpinBox(); self.artifact_edge_width.setRange(0, 20); self.artifact_edge_width.setValue(2)
            advanced.addRow("SG 窗口", self.sg_window)
            advanced.addRow("SG 多项式", self.sg_order)
            advanced.addRow("全深度样本块", self.sample_blocks)
            advanced.addRow("每块行数", self.block_rows)
            advanced.addRow("推理分块行数", self.chunk_rows)
            advanced.addRow("每矿物最大参考谱", self.max_references)
            advanced.addRow("边缘风险宽度", self.artifact_edge_width)
            self.reset_advanced_button = QPushButton("恢复自动建议")
            self.reset_advanced_button.clicked.connect(self._reset_advanced_defaults)
            advanced.addRow("", self.reset_advanced_button)
            advanced_outer.addWidget(self.advanced_body)
            self.advanced_group.toggled.connect(self._toggle_advanced_settings)
            self.advanced_body.setVisible(False)
            layout.addWidget(self.advanced_group)
            layout.addStretch(1)
            return page

        def _toggle_advanced_settings(self, enabled: bool) -> None:
            self.advanced_body.setVisible(enabled)
            self.advanced_group.setTitle(
                "高级自动设置 · 当前使用人工覆盖值" if enabled else "高级自动设置 · 展开后覆盖自动建议"
            )

        def _reset_advanced_defaults(self) -> None:
            self.sg_window.setValue(11)
            self.sg_order.setValue(2)
            self.sample_blocks.setValue(12)
            self.block_rows.setValue(64)
            self.chunk_rows.setValue(64)
            self.max_references.setValue(6)
            self.artifact_edge_width.setValue(2)

        def _page_spectra(self) -> QWidget:
            page, layout = self._page_shell(
                "系统按所选矿物、传感器波长和参考谱质量自动入选标准谱；请先核对曲线与来源，再进入识别模式。"
            )
            top = QSplitter(Qt.Orientation.Horizontal)
            plot_card = QFrame(); plot_card.setObjectName("previewFrame")
            plot_layout = QVBoxLayout(plot_card)
            selector = QHBoxLayout()
            selector.addWidget(QLabel("目标矿物"))
            self.evidence_mineral = QComboBox()
            self.evidence_mineral.currentTextChanged.connect(self._filter_evidence_curves)
            selector.addWidget(self.evidence_mineral, 1)
            self.evidence_status = _StatusPill("尚未审计", "neutral")
            selector.addWidget(self.evidence_status)
            plot_layout.addLayout(selector)
            self.spectrum_plot = SpectrumPlot()
            plot_layout.addWidget(self.spectrum_plot, 1)
            self.reference_curve_summary = QLabel("每条实际入选参考谱均单独绘制；曲线非常接近时可能重叠，可点击右侧条目高亮。")
            self.reference_curve_summary.setWordWrap(True)
            self.reference_curve_summary.setObjectName("pageSubtitle")
            plot_layout.addWidget(self.reference_curve_summary)
            top.addWidget(plot_card)
            references_card = QFrame(); references_card.setObjectName("detailCard")
            references_layout = QVBoxLayout(references_card)
            references_layout.addWidget(QLabel("入选标准谱与来源"))
            self.reference_list = QListWidget()
            self.reference_list.currentRowChanged.connect(self.spectrum_plot.set_highlighted_curve)
            references_layout.addWidget(self.reference_list, 1)
            top.addWidget(references_card)
            top.setStretchFactor(0, 3); top.setStretchFactor(1, 2)
            layout.addWidget(top, 1)
            self.references_confirmed = _VisibleCheckBox("我已核对本项目入选的标准光谱曲线与来源")
            self.references_confirmed.toggled.connect(self._on_reference_confirmation_changed)
            layout.addWidget(self.references_confirmed)
            return page

        def _page_threshold_trial(self) -> QWidget:
            page, layout = self._page_shell(
                "进入本页会自动试算。参数只在标准谱 Catalog 安全范围内优选，并同时检查空间伪影和矿物特征可通过率。"
            )
            method_card = QFrame(); method_card.setObjectName("infoCard")
            method_layout = QHBoxLayout(method_card)
            self.threshold_method_label = QLabel("等待场景阈值试算")
            self.threshold_method_label.setWordWrap(True)
            method_layout.addWidget(self.threshold_method_label, 1)
            self.rerun_threshold_button = QPushButton("应用调整并重新试算")
            self.rerun_threshold_button.clicked.connect(self._rerun_threshold_trial)
            method_layout.addWidget(self.rerun_threshold_button)
            layout.addWidget(method_card)
            lower = QSplitter(Qt.Orientation.Horizontal)
            self.threshold_table = QTableWidget(0, 7)
            self.threshold_table.setHorizontalHeaderLabels(
                ("矿物组", "档位", "列分位", "SAM 角度", "样本候选", "特征可通过", "状态")
            )
            self.threshold_table.verticalHeader().setVisible(False)
            self.threshold_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.threshold_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.threshold_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            self.threshold_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            self.threshold_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            self.threshold_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            self.threshold_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
            threshold_card = QFrame(); threshold_card.setObjectName("detailCard")
            threshold_layout = QVBoxLayout(threshold_card)
            threshold_header = QHBoxLayout()
            threshold_header.addWidget(QLabel("三档自动优选参数（可编辑）"))
            threshold_header.addStretch(1)
            self.export_thresholds_button = QPushButton("导出阈值 JSON")
            self.export_thresholds_button.clicked.connect(self._export_thresholds_dialog)
            threshold_header.addWidget(self.export_thresholds_button)
            threshold_layout.addLayout(threshold_header)
            threshold_layout.addWidget(self.threshold_table)
            lower.addWidget(threshold_card)
            risk_card = QFrame(); risk_card.setObjectName("detailCard")
            risk_layout = QVBoxLayout(risk_card)
            risk_layout.addWidget(QLabel("风险摘要"))
            self.risk_table = QTableWidget(0, 2)
            self.risk_table.setHorizontalHeaderLabels(("风险 / 检查", "状态"))
            self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            risk_layout.addWidget(self.risk_table)
            self.audit_summary = QLabel("进入本页后自动执行场景阈值试算")
            self.audit_summary.setWordWrap(True)
            risk_layout.addWidget(self.audit_summary)
            lower.addWidget(risk_card)
            lower.setStretchFactor(0, 3); lower.setStretchFactor(1, 2)
            layout.addWidget(lower, 1)
            return page

        def _page_run_control(self) -> QWidget:
            page, layout = self._page_shell(
                "确认三项准备状态后开始识别；详细技术记录会自动写入成果目录。"
            )
            checklist = QGroupBox("运行准备")
            checklist_layout = QVBoxLayout(checklist)
            self.run_audit_state = _StatusPill("尚未审计", "neutral")
            self.run_mask_state = _StatusPill("掩膜未批准", "neutral")
            self.run_threshold_state = _StatusPill("阈值待审查", "neutral")
            for label, pill in (
                ("数据可用", self.run_audit_state),
                ("掩膜已核对", self.run_mask_state),
                ("标准谱和阈值已确认", self.run_threshold_state),
            ):
                row = QHBoxLayout()
                row.addWidget(QLabel(label))
                row.addStretch(1)
                row.addWidget(pill)
                checklist_layout.addLayout(row)
            layout.addWidget(checklist)
            controls = QHBoxLayout()
            self.run_page_audit_button = QPushButton("返回阈值试算")
            self.run_page_audit_button.clicked.connect(lambda: self._set_step(6))
            self.run_page_start_button = QPushButton("开始识别")
            self.run_page_start_button.clicked.connect(self._start_run)
            controls.addWidget(self.run_page_audit_button)
            controls.addStretch(1)
            controls.addWidget(self.run_page_start_button)
            layout.addLayout(controls)
            layout.addStretch(1)
            return page

        def _page_results(self) -> QWidget:
            page, layout = self._page_shell("默认解释平衡版；严格版用于高可信核对，敏感版用于检查潜在漏识别。")
            controls = QHBoxLayout()
            self.preview_combo = QComboBox()
            self.preview_combo.currentIndexChanged.connect(
                lambda _index: self._show_preview(self.preview_combo.currentData())
            )
            self.open_output_button = QPushButton("打开成果目录")
            self.open_output_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
            self.open_output_button.clicked.connect(self._open_output)
            self.open_output_button.setEnabled(False)
            controls.addWidget(QLabel("结果视图"))
            controls.addWidget(self.preview_combo, 1)
            self.preview_zoom = QComboBox()
            self.preview_zoom.addItem("适应窗口", 0.0)
            self.preview_zoom.addItem("50%", 0.5)
            self.preview_zoom.addItem("100% · 1:1", 1.0)
            self.preview_zoom.addItem("200%", 2.0)
            self.preview_zoom.currentIndexChanged.connect(self._scale_result_preview)
            controls.addWidget(QLabel("缩放"))
            controls.addWidget(self.preview_zoom)
            controls.addWidget(self.open_output_button)
            layout.addLayout(controls)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            image_frame = QFrame(); image_frame.setObjectName("previewFrame")
            image_layout = QVBoxLayout(image_frame)
            self.result_scroll = QScrollArea(); self.result_scroll.setWidgetResizable(False)
            self.result_preview = QLabel("完成运行后显示分类与风险预览")
            self.result_preview.setObjectName("imageWell")
            self.result_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_preview.setMinimumSize(680, 480)
            self.result_scroll.setWidget(self.result_preview)
            image_layout.addWidget(self.result_scroll)
            splitter.addWidget(image_frame)
            side = QFrame(); side.setObjectName("detailCard")
            side_layout = QVBoxLayout(side)
            self.quality_status = _StatusPill("无运行", "neutral")
            self.quality_text = QLabel("尚无质量报告")
            self.quality_text.setWordWrap(True)
            self.run_log = QTextEdit(); self.run_log.setReadOnly(True)
            side_layout.addWidget(self.quality_status)
            side_layout.addWidget(self.quality_text)
            side_layout.addWidget(QLabel("运行日志"))
            side_layout.addWidget(self.run_log, 1)
            splitter.addWidget(side)
            splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 1)
            layout.addWidget(splitter, 1)
            return page

        def _apply_style(self) -> None:
            self.setStyleSheet(
                """
                QMainWindow, QWidget { background:#F5F7FA; color:#203040; font-family:'Microsoft YaHei UI','Segoe UI'; font-size:13px; }
                QToolBar { background:#FFFFFF; border:0; border-bottom:1px solid #DCE4EC; spacing:5px; padding:5px 10px; }
                QToolBar QToolButton { padding:6px 9px; }
                QFrame#sidebar { background:#102A43; border:0; }
                QLabel#brandLogo { background:#F7FBFF; border:1px solid #7DD8E8; border-radius:10px; padding:4px; }
                QLabel#brandName { background:transparent; color:#F4FBFF; font-size:18px; font-weight:700; }
                QLabel#databaseNote { background:#123A58; color:#D8F4FA; border:1px solid #245A78; border-radius:8px; padding:10px; line-height:1.4; }
                QListWidget#stepList { background:transparent; border:0; color:#C9D6E3; outline:0; }
                QListWidget#stepList::item { border-radius:7px; padding:9px 8px; margin:1px 0; }
                QListWidget#stepList::item:selected { background:#1677B8; color:#FFFFFF; border-left:3px solid #58D5E8; }
                QListWidget#stepList::item:hover:!selected { background:#173B5E; }
                QFrame#workspace { background:#F5F7FA; }
                QLabel#pageTitle { font-size:24px; font-weight:700; color:#102A43; }
                QLabel#pageSubtitle { color:#65788B; }
                QLabel#pageLead { background:#EAF2FA; color:#315A78; border-left:3px solid #2684C7; padding:10px 12px; }
                QGroupBox { background:#FFFFFF; border:1px solid #DDE5ED; border-radius:8px; margin-top:14px; padding:14px 12px 10px 12px; font-weight:600; }
                QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; color:#234E70; }
                QLineEdit, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTreeWidget, QListWidget { background:#FFFFFF; border:1px solid #CCD8E3; border-radius:5px; padding:5px; selection-background-color:#D9EAFE; selection-color:#163B5C; }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border:1px solid #2684C7; }
                QPushButton { background:#FFFFFF; border:1px solid #B8C7D5; border-radius:6px; padding:7px 14px; }
                QPushButton:hover { background:#EDF5FC; border-color:#2684C7; }
                QPushButton:disabled { color:#98A8B7; background:#EEF1F4; }
                QFrame#infoCard, QFrame#detailCard, QFrame#previewFrame { background:#FFFFFF; border:1px solid #DDE5ED; border-radius:8px; }
                QLabel#detailTitle { color:#173B5E; font-size:17px; font-weight:700; }
                QLabel#imageWell { background:#EEF2F6; color:#6F8191; border:1px dashed #AFC0CF; }
                QFrame#taskBar { background:#FFFFFF; border-top:1px solid #DCE4EC; }
                QProgressBar { background:#E7EDF3; border:0; border-radius:4px; height:8px; }
                QProgressBar::chunk { background:#2684C7; border-radius:4px; }
                QHeaderView::section { background:#EDF2F7; color:#38556D; border:0; border-right:1px solid #D7E0E8; border-bottom:1px solid #D7E0E8; padding:7px; font-weight:600; }
                """
            )

        def _populate_minerals(self) -> None:
            self.mineral_tree.clear()
            self.mineral_items: dict[str, QTreeWidgetItem] = {}
            minerals = self.catalog.get("minerals", {})
            groups = self.catalog.get("taxonomy_groups", {})
            for group_id, group in groups.items():
                root = QTreeWidgetItem(
                    [
                        f"{group.get('display_name_zh', group_id)} / {group.get('display_name_en', '')}",
                        "",
                        "",
                        "",
                    ]
                )
                root.setData(0, Qt.ItemDataRole.UserRole, {"group_id": group_id})
                root.setFlags(root.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
                root.setCheckState(0, Qt.CheckState.Unchecked)
                self.mineral_tree.addTopLevelItem(root)
                members = group.get("members", ())
                for mineral_id in members:
                    record = minerals.get(mineral_id)
                    if record is None:
                        continue
                    windows = self._format_windows(record.get("required_windows_nm", ()))
                    support = self._support_label(record.get("support_level", "unknown"))
                    count = int(self.reference_counts.get(mineral_id, 0))
                    child = QTreeWidgetItem(
                        [
                            f"{record.get('display_name_zh', '')}  {record.get('display_name_en', mineral_id)}",
                            support,
                            windows,
                            str(count),
                        ]
                    )
                    child.setData(0, Qt.ItemDataRole.UserRole, {"mineral_id": mineral_id})
                    enabled = bool(record.get("recognition_enabled", True))
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    default = enabled and record.get("support_level") == "validated_swir"
                    child.setCheckState(0, Qt.CheckState.Checked if default else Qt.CheckState.Unchecked)
                    if not enabled:
                        child.setDisabled(True)
                        child.setToolTip(0, str(record.get("limitation", "当前数据能力不支持")))
                    root.addChild(child)
                    self.mineral_items[mineral_id] = child
                root.setExpanded(True)
            self.mineral_tree.resizeColumnToContents(1)
            self.mineral_tree.resizeColumnToContents(2)

        @staticmethod
        def _support_label(level: str) -> str:
            return {
                "validated_swir": "已验证 SWIR",
                "experimental_swir": "实验级 SWIR",
                "family_only_rgb_nir": "仅族级 / 锁定",
                "unsupported_no_tir": "缺少 TIR / 锁定",
                "supported": "当前传感器支持",
                "conditional": "当前传感器条件支持",
                "unsupported": "当前传感器不支持 / 锁定",
                "unknown": "当前传感器能力未知",
            }.get(str(level), str(level).replace("_", " "))

        @staticmethod
        def _format_windows(windows: Sequence[Sequence[Any]]) -> str:
            values = []
            for window in windows:
                if len(window) >= 2:
                    values.append(f"{float(window[0]):.0f}–{float(window[1]):.0f}")
            return ", ".join(values) + (" nm" if values else "")

        def _show_mineral_details(self, current: QTreeWidgetItem | None, previous: QTreeWidgetItem | None = None) -> None:
            if current is None:
                return
            data = current.data(0, Qt.ItemDataRole.UserRole) or {}
            mineral_id = data.get("mineral_id")
            if not mineral_id:
                self.mineral_detail_title.setText(current.text(0))
                self.mineral_detail_text.setText("展开矿物族并选择二级矿物查看诊断特征。")
                return
            record = self.catalog["minerals"][mineral_id]
            features = []
            for feature in record.get("key_features_nm", ()):
                if feature.get("center") is not None:
                    features.append(f"{feature['center']} nm {feature.get('kind', '')}".strip())
                elif feature.get("range"):
                    features.append(self._format_windows((feature["range"],)))
            self.mineral_detail_title.setText(
                f"{record.get('display_name_zh', '')} · {record.get('display_name_en', mineral_id)}"
            )
            lines = [
                f"化学式：{record.get('formula', '—')}",
                f"光谱族：{record.get('spectral_family', '—')}",
                f"必需波段：{self._format_windows(record.get('required_windows_nm', ()))}",
                f"关键特征：{', '.join(features) or '—'}",
                f"易混矿物：{', '.join(record.get('confusers', ())) or '—'}",
                f"内置候选参考谱：{self.reference_counts.get(mineral_id, 0)} 条",
                f"蚀变角色：{', '.join(record.get('alteration_roles', ())) or '—'}",
            ]
            if record.get("limitation"):
                lines.append(f"限制：{record['limitation']}")
            self.mineral_detail_text.setText("\n\n".join(lines))

        def _set_step(self, index: int) -> None:
            if not hasattr(self, "pages"):
                return
            index = max(0, min(len(self.STEP_TITLES) - 1, int(index)))
            with QSignalBlocker(self.steps):
                self.steps.setCurrentRow(index)
            self.pages.setCurrentIndex(index)
            zh, en, subtitle = self.STEP_TITLES[index]
            self.page_title.setText(f"{zh}  ·  {en}")
            self.page_subtitle.setText(subtitle)
            self.back_button.setEnabled(index > 0)
            self.next_button.setEnabled(index < len(self.STEP_TITLES) - 1)
            self.next_button.setText(
                "开始识别 →" if index == 7 and self.current_run_directory is None
                else "查看结果 →" if index == 7
                else "自动试算阈值 →" if index == 5
                else "优选标准谱 →" if index == 3
                else "下一步 →"
            )

        def _set_step_state(self, index: int, state: str) -> None:
            if not 0 <= index < len(self.STEP_TITLES):
                return
            self.step_states[index] = state
            marker = {"pending": "○", "active": "◉", "complete": "✓", "warning": "!", "blocked": "×"}.get(state, "○")
            zh, en, _ = self.STEP_TITLES[index]
            self.steps.item(index).setText(f"{index + 1:02d} {marker} {zh}\n       {en}")

        def _advance_step(self) -> None:
            index = self.pages.currentIndex()
            if index == 3:
                self._start_audit(target_step=4)
                return
            if index == 4:
                if not self.references_confirmed.isChecked():
                    QMessageBox.information(self, "标准谱确认", "请核对入选标准光谱后勾选确认。")
                    return
                self._set_step(5)
                return
            if index == 5:
                self._start_audit(target_step=6)
                return
            if index == 7 and self.current_run_directory is None:
                self._start_run()
                return
            self._set_step(index + 1)

        def _on_reference_confirmation_changed(self, confirmed: bool) -> None:
            self._set_step_state(4, "complete" if confirmed else "warning")
            self.run_threshold_state.set_state(
                "标准谱已确认 · 阈值待试算" if confirmed else "标准谱待确认",
                "warning" if confirmed else "neutral",
            )
            if hasattr(self, "run_page_start_button"):
                idle = self.worker is None or not self.worker.isRunning()
                self.run_page_start_button.setEnabled(
                    confirmed and self.mask_approved.isChecked() and not self._threshold_trial_dirty and idle
                )

        def _export_thresholds_dialog(self) -> None:
            thresholds = self.current_thresholds
            if thresholds is None:
                QMessageBox.information(self, "导出阈值", "请先完成项目审计或加载运行结果。")
                return
            path, _ = QFileDialog.getSaveFileName(self, "导出阈值", "corespec_v5_3_thresholds.json", "JSON (*.json)")
            if not path:
                return
            Path(path).write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

        def _discover_companions(self) -> None:
            if not self.auto_discover.isChecked() or not self.analysis_path.text():
                return
            source = Path(self.analysis_path.text())
            parent = source.parent
            if not parent.is_dir():
                return
            files_in_directory = [item for item in parent.iterdir() if item.is_file()]
            for prefix, picker in (("rgb", self.rgb_path), ("nir", self.nir_path), ("swir", self.swir_path)):
                if picker.text():
                    continue
                candidates = [
                    item for item in files_in_directory
                    if item.name.casefold().startswith(prefix) and item.suffix.casefold() in {".dat", ".img", ".hdr"}
                ]
                candidates.sort(key=lambda item: ({".dat": 0, ".img": 1, ".hdr": 2}.get(item.suffix.casefold(), 3), item.name))
                if candidates:
                    picker.setText(candidates[0])
            lower_name = source.name.casefold()
            if lower_name.startswith("rgb") and not self.rgb_path.text():
                self.rgb_path.setText(source)
            elif lower_name.startswith("nir") and not self.nir_path.text():
                self.nir_path.setText(source)
            elif lower_name.startswith("swir") and not self.swir_path.text():
                self.swir_path.setText(source)
            self._refresh_selected_inputs()

        def _on_data_option_changed(self, *_args) -> None:
            if self.auto_discover.isChecked():
                self._discover_companions()
            processing = "保留已完成的 SG 平滑，不再重复平滑" if self.input_smoothed.isChecked() else "运行时自动执行 SG 平滑"
            discovery = "自动发现同目录伴随影像" if self.auto_discover.isChecked() else "伴随影像由用户手动指定"
            self.data_feedback.setText(f"{processing}；{discovery}。")
            self._refresh_selected_inputs()

        def _refresh_selected_inputs(self, *_args) -> None:
            if not hasattr(self, "input_table"):
                return
            rows: list[tuple[str, str, str, str, str]] = []
            entries = (
                ("主分析", self.analysis_path.text(), self.data_physics.currentText()),
                ("RGB", self.rgb_path.text(), "辅助影像"),
                ("NIR/VNIR", self.nir_path.text(), "辅助影像"),
                ("SWIR", self.swir_path.text(), "辅助影像"),
            )
            for role, value, physics in entries:
                if not value:
                    continue
                path = Path(value)
                try:
                    with EnviDataset(path) as dataset:
                        shape = f"{dataset.info.lines} × {dataset.info.samples} × {dataset.info.bands}"
                        wavelengths = dataset.info.wavelengths_nm
                        if wavelengths is not None and wavelengths.size:
                            spectral = f"{float(np.nanmin(wavelengths)):.1f}–{float(np.nanmax(wavelengths)):.1f} nm"
                        else:
                            spectral = f"{dataset.info.bands} 波段"
                        status = "已读取"
                except (FileNotFoundError, OSError, ValueError) as exc:
                    shape = "—"
                    spectral = "—"
                    status = "路径待完善" if not path.is_file() else f"无法读取：{str(exc).splitlines()[0]}"
                rows.append((role, shape, spectral, physics, status))
            self.input_table.setRowCount(len(rows))
            for row, values in enumerate(rows):
                for column, value in enumerate(values):
                    self.input_table.setItem(row, column, QTableWidgetItem(str(value)))
            if rows:
                readable = sum(item[-1] == "已读取" for item in rows)
                processing = "跳过重复 SG 平滑" if self.input_smoothed.isChecked() else "运行时执行 SG 平滑"
                self.data_feedback.setText(f"已选择 {len(rows)} 组数据，其中 {readable} 组可读取；{processing}。")

        def _selected_minerals(self) -> list[str]:
            return [
                mineral_id for mineral_id, item in self.mineral_items.items()
                if not item.isDisabled() and item.checkState(0) == Qt.CheckState.Checked
            ]

        def _on_mineral_selection_changed(self, *_args) -> None:
            if hasattr(self, "references_confirmed"):
                self.references_confirmed.setChecked(False)
            self.current_thresholds = None
            self._threshold_editors.clear()

        def build_config(self) -> dict[str, Any]:
            image = self.analysis_path.text()
            if not image:
                raise ValueError("请选择主分析立方体")
            requested = self._selected_minerals()
            if not requested:
                raise ValueError("至少选择一个当前可用矿物")
            if self.mask_external.isChecked() and not self.mask_path.text():
                raise ValueError("外部掩膜模式需要选择 ENVI 掩膜")
            if self.mask_auto.isChecked() and not self.rgb_path.text():
                raise ValueError("自动掩膜需要在数据页选择 RGB 影像")
            if self.sg_window.value() % 2 == 0:
                raise ValueError("SG 窗口必须为奇数")
            result = deepcopy(self._base_config)
            result["schema_version"] = 3
            result["application_version"] = "5.3.0"
            project = result.setdefault("project", {})
            project.update(
                {
                    "name": self.project_name.text().strip() or "CoreSpec_Project",
                    "description": self.project_description.text().strip(),
                    "output_root": self.output_path.text(),
                    "non_destructive": True,
                }
            )
            inputs = result.setdefault("inputs", {})
            inputs.pop("spectral_library_root", None)
            inputs.update(
                {
                    "analysis_image": image,
                    "analysis_domain": self.analysis_domain.currentData(),
                    "data_physics": self.data_physics.currentData(),
                    "input_is_smoothed": self.input_smoothed.isChecked(),
                    "auto_discover_companions": self.auto_discover.isChecked(),
                    "rgb": self.rgb_path.text() or None,
                    "nir": self.nir_path.text() or None,
                    "swir": self.swir_path.text() or None,
                }
            )
            mask = result.setdefault("mask", {})
            existing_approval = mask.get("approval", {}) if isinstance(mask.get("approval"), Mapping) else {}
            mask.update(
                {
                    "mode": "external" if self.mask_external.isChecked() else "automatic",
                    "engine": "integrated_core_foreground_v2",
                    "path": self.mask_path.text() or None,
                    "minimum_component_pixels": self.mask_min_component.value(),
                    "minimum_mask_fraction": self.mask_min_fraction.value() / 100.0,
                    "minimum_interior_pixel_fraction": self.mask_interior_fraction.value() / 100.0,
                    "edge_guard_pixels": self.mask_edge_guard.value(),
                    "approval": {
                        "required": True,
                        "approved": self.mask_approved.isChecked(),
                        "reviewer": self.mask_reviewer.text().strip() or None,
                        "reviewed_at": existing_approval.get("reviewed_at"),
                        "notes": self.mask_notes.text().strip() or None,
                    },
                }
            )
            mask.pop("geocore_module_root", None)
            mask.pop("model_package", None)
            minerals = result.setdefault("minerals", {})
            minerals.update({"requested": requested, "include_internal_confusers": True})
            mode = result.setdefault("mode", {})
            mode["engine"] = "adaptive_v5"
            mode.update(
                {
                    "profile": self.profile.currentData(),
                    "automatic": True,
                    "quick_calibration": False,
                    "write_all_profiles": True,
                }
            )
            advanced = result.setdefault("advanced", {})
            advanced.update(
                {
                    "user_overrides_enabled": self.advanced_group.isChecked(),
                    "preprocessing": {"sg_window": self.sg_window.value(), "sg_polyorder": self.sg_order.value()},
                    "sampling": {"blocks": self.sample_blocks.value(), "block_rows": self.block_rows.value()},
                    "execution": {"chunk_rows": self.chunk_rows.value()},
                    "library_ensemble": {"maximum_representatives_per_mineral": self.max_references.value()},
                    "artifact_control": {"edge_width": self.artifact_edge_width.value()},
                }
            )
            desktop = result.setdefault("desktop", {})
            desktop.update(
                {
                    "last_step": self.pages.currentIndex(),
                    "preview_key": self.preview_combo.currentData(),
                    "references_confirmed": self.references_confirmed.isChecked(),
                }
            )
            if self._threshold_editors:
                trial = result.setdefault("threshold_trial", {})
                trial.update(
                    {
                        "method": "catalog_bounded_scene_proxy_plus_feature_support_v5_3",
                        "overrides": self._threshold_overrides_from_editors(),
                    }
                )
            result.pop("spectral_library_root", None)
            if isinstance(result.get("v5"), dict):
                result["v5"].pop("spectral_library_root", None)
            return result

        def apply_config(self, config: Mapping[str, Any]) -> None:
            value = deepcopy(dict(config))
            self._base_config = value
            project = value.get("project", {})
            self.project_name.setText(str(project.get("name", value.get("project_name", "CoreSpec_Project"))))
            self.project_description.setText(str(project.get("description", "")))
            self.output_path.setText(project.get("output_root", value.get("desktop", {}).get("output_root", "")))
            inputs = value.get("inputs", {})
            self.analysis_path.setText(inputs.get("analysis_image", value.get("analysis_image", "")))
            self._set_combo_data(self.analysis_domain, inputs.get("analysis_domain", "auto"))
            self._set_combo_data(self.data_physics, inputs.get("data_physics", "reflectance"))
            self.input_smoothed.setChecked(bool(inputs.get("input_is_smoothed", value.get("analysis_input_is_smoothed", False))))
            self.auto_discover.setChecked(bool(inputs.get("auto_discover_companions", True)))
            self.rgb_path.setText(inputs.get("rgb")); self.nir_path.setText(inputs.get("nir")); self.swir_path.setText(inputs.get("swir"))
            mask = value.get("mask", {})
            external = mask.get("mode") == "external"
            self.mask_external.setChecked(external); self.mask_auto.setChecked(not external)
            self.mask_path.setText(mask.get("path"))
            self._set_combo_data(self.mask_engine, "integrated_core_foreground_v2")
            self.mask_min_component.setValue(int(mask.get("minimum_component_pixels", 64)))
            self.mask_min_fraction.setValue(round(100 * float(mask.get("minimum_mask_fraction", 0.05))))
            self.mask_interior_fraction.setValue(round(100 * float(mask.get("minimum_interior_pixel_fraction", 0.65))))
            self.mask_edge_guard.setValue(int(mask.get("edge_guard_pixels", 2)))
            approval = mask.get("approval", {}) if isinstance(mask.get("approval"), Mapping) else {}
            self.mask_approved.setChecked(bool(approval.get("approved", False)))
            self.mask_reviewer.setText(str(approval.get("reviewer") or ""))
            self.mask_notes.setText(str(approval.get("notes") or ""))
            requested = set(value.get("minerals", {}).get("requested", ()))
            if requested:
                for mineral_id, item in self.mineral_items.items():
                    item.setCheckState(0, Qt.CheckState.Checked if mineral_id in requested and not item.isDisabled() else Qt.CheckState.Unchecked)
            mode = value.get("mode", {})
            self._set_combo_data(self.execution_engine, "adaptive_v5")
            self._set_combo_data(self.profile, mode.get("profile", "balanced"))
            validation_available = isinstance(value.get("advanced", {}).get("validation"), Mapping)
            self.quick_calibration.setEnabled(validation_available)
            self.quick_calibration.setChecked(bool(mode.get("quick_calibration", validation_available)))
            self.write_all_profiles.setChecked(True)
            advanced = value.get("advanced", {})
            self.advanced_group.setChecked(bool(advanced.get("user_overrides_enabled", False)))
            pre = advanced.get("preprocessing", {}); sampling = advanced.get("sampling", {})
            execution = advanced.get("execution", {}); library = advanced.get("library_ensemble", {}); artifact = advanced.get("artifact_control", {})
            self.sg_window.setValue(int(pre.get("sg_window", 11))); self.sg_order.setValue(int(pre.get("sg_polyorder", 2)))
            self.sample_blocks.setValue(int(sampling.get("blocks", 12))); self.block_rows.setValue(int(sampling.get("block_rows", 64)))
            self.chunk_rows.setValue(int(execution.get("chunk_rows", 64))); self.max_references.setValue(int(library.get("maximum_representatives_per_mineral", 6)))
            self.artifact_edge_width.setValue(int(artifact.get("edge_width", 2)))
            self.references_confirmed.setChecked(bool(value.get("desktop", {}).get("references_confirmed", False)))
            self._set_step(int(value.get("desktop", {}).get("last_step", 0)))

        @staticmethod
        def _set_combo_data(combo: QComboBox, value: Any) -> None:
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)

        def load_config_file(self, path: str | Path) -> dict[str, Any]:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("V5 配置根节点必须是 JSON 对象")
            self.apply_config(value)
            return value

        def save_config_file(self, path: str | Path) -> Path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.build_config(), ensure_ascii=False, indent=2), encoding="utf-8")
            return target

        def _open_config_dialog(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "打开 V5 配置", "", "CoreSpec JSON (*.json);;所有文件 (*)")
            if not path:
                return
            try:
                self.load_config_file(path)
            except Exception as exc:
                QMessageBox.warning(self, "CoreSpec Mapper V5", str(exc))

        def _save_config_dialog(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "保存 V5 配置", "corespec_v5.json", "CoreSpec JSON (*.json)")
            if not path:
                return
            try:
                self.save_config_file(path)
            except Exception as exc:
                QMessageBox.warning(self, "CoreSpec Mapper V5", str(exc))

        def _start_audit(self, _checked: bool = False, *, target_step: int | None = None) -> None:
            try:
                config = self.build_config()
            except Exception as exc:
                QMessageBox.warning(self, "项目审计", str(exc))
                return
            self._audit_target_step = 6 if target_step is None else int(target_step)
            if self._audit_target_step == 4:
                self.references_confirmed.setChecked(False)
            self._start_worker(
                _V5Worker("audit", config, audit_function=self.audit_function, run_function=self.run_function)
            )
            self.project_state.set_state("审计中", "active")
            self.stage_label.setText("正在审计输入与标准谱")

        def _start_run(self) -> None:
            try:
                config = self.build_config()
                output = self.output_path.text()
                if not output:
                    raise ValueError("请选择输出根目录")
                approval = config.get("mask", {}).get("approval", {})
                if approval.get("required") and not approval.get("approved"):
                    raise ValueError("请先在掩膜页核对预览并勾选人工批准")
                if not self.references_confirmed.isChecked():
                    raise ValueError("请先核对并确认本项目入选的标准光谱")
                if self._threshold_trial_dirty:
                    raise ValueError("阈值参数已经调整，请先重新试算")
                trial = _to_mapping(self.last_audit.get("threshold_trial"))
                if trial and trial.get("status") != "resolved":
                    raise ValueError("阈值试算尚未通过，请处理风险提示后重新试算")
            except Exception as exc:
                QMessageBox.warning(self, "运行 V5", str(exc))
                return
            self._start_worker(
                _V5Worker("run", config, output, audit_function=self.audit_function, run_function=self.run_function)
            )
            self.project_state.set_state("运行中", "active")
            self.stage_label.setText("正在运行 V5")

        def _start_worker(self, worker: _V5Worker) -> None:
            if self.worker is not None and self.worker.isRunning():
                return
            self.worker = worker
            worker.progress_event.connect(self._on_progress)
            worker.completed.connect(self._on_completed)
            worker.cancelled.connect(self._on_cancelled)
            worker.failed.connect(self._on_failed)
            worker.finished.connect(self._worker_finished)
            self.progress_bar.setValue(0)
            self._last_progress_fraction = 0.0
            self._last_remaining_seconds = None
            self._task_started_at = monotonic()
            self._eta_seconds_per_fraction = None
            self.timing_label.setText("已用 00:00  ·  剩余 估算中")
            self._set_busy(True)
            worker.start()

        def _on_progress(self, event: Mapping[str, Any]) -> None:
            fraction = float(event.get("overall_fraction", 0.0))
            self.progress_bar.setValue(round(1000 * max(0.0, min(1.0, fraction))))
            self.stage_label.setText(str(event.get("message", event.get("stage", "处理中"))))
            elapsed = _safe_float(event.get("elapsed_seconds"))
            remaining = _safe_float(event.get("estimated_remaining_seconds"))
            if remaining is not None and remaining <= 0.0 and fraction < 0.995:
                remaining = None
            if elapsed is None and self._task_started_at is not None:
                elapsed = monotonic() - self._task_started_at
            if elapsed is not None and fraction >= 0.01:
                observed_rate = elapsed / max(fraction, 1e-6)
                self._eta_seconds_per_fraction = (
                    observed_rate
                    if self._eta_seconds_per_fraction is None
                    else 0.78 * self._eta_seconds_per_fraction + 0.22 * observed_rate
                )
                derived_remaining = self._eta_seconds_per_fraction * max(0.0, 1.0 - fraction)
                if fraction < 0.995:
                    derived_remaining = max(1.0, derived_remaining)
                if remaining is None:
                    remaining = derived_remaining
                else:
                    remaining = 0.65 * remaining + 0.35 * derived_remaining
            if remaining is not None:
                if fraction >= self._last_progress_fraction and self._last_remaining_seconds is not None:
                    remaining = min(remaining, self._last_remaining_seconds)
                self._last_remaining_seconds = remaining
            self._last_progress_fraction = max(self._last_progress_fraction, fraction)
            remaining_text = "估算中" if remaining is None and fraction < 0.995 else _format_duration(remaining)
            self.timing_label.setText(f"已用 {_format_duration(elapsed)}  ·  剩余 {remaining_text}")
            message = str(event.get("message", ""))
            if message:
                self.run_log.append(message)

        def _on_completed(self, result: Any) -> None:
            action = self.worker.action if self.worker is not None else ""
            if action == "audit":
                record = self.load_audit_result(result)
                self._set_step(self._audit_target_step if self._audit_target_step is not None else 6)
                self._audit_target_step = None
                ready = bool(record.get("ready", False))
                self.project_state.set_state("可审查" if ready else "待批准", "good" if ready else "warning")
                self.stage_label.setText("审计完成")
            else:
                record = self.load_run_result(result)
                self._set_step(8)
                quality = _to_mapping(record.get("quality"))
                publishable = bool(quality.get("publishable", False))
                grade = str(quality.get("quality_grade", record.get("quality_grade", "—"))).upper()
                self.project_state.set_state(
                    "已完成 · 可发布" if publishable else f"已完成 · 质量 {grade}",
                    "good" if publishable else "warning",
                )
                self.stage_label.setText("运行完成")
            self.progress_bar.setValue(1000)

        def _on_cancelled(self, message: str) -> None:
            self.project_state.set_state("已取消", "warning")
            self.stage_label.setText("任务已安全取消")
            self.run_log.append(message or "任务已取消；未发布不完整成果。")

        def _on_failed(self, message: str) -> None:
            self.project_state.set_state("失败", "bad")
            self.stage_label.setText("任务失败")
            self.run_log.append(message)
            QMessageBox.critical(self, "CoreSpec Mapper V5", message)

        def _worker_finished(self) -> None:
            self._set_busy(False)
            self.worker = None
            if self._close_when_idle:
                self._close_when_idle = False
                self.close()

        def _cancel(self) -> None:
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel()
                self.stage_label.setText("正在等待安全检查点…")
                self.cancel_action.setEnabled(False)
                self.cancel_button.setEnabled(False)

        def _set_busy(self, busy: bool) -> None:
            self.audit_action.setEnabled(not busy)
            self.run_action.setEnabled(not busy)
            self.cancel_action.setEnabled(busy)
            self.cancel_button.setEnabled(busy)
            self.open_action.setEnabled(not busy)
            self.save_action.setEnabled(not busy)
            if hasattr(self, "run_page_audit_button"):
                self.run_page_audit_button.setEnabled(not busy)
                self.run_page_start_button.setEnabled(
                    not busy
                    and self.mask_approved.isChecked()
                    and self.references_confirmed.isChecked()
                    and not self._threshold_trial_dirty
                )
            if hasattr(self, "generate_mask_button"):
                self.generate_mask_button.setEnabled(not busy)

        def load_audit_result(self, result: Any) -> dict[str, Any]:
            record = _to_mapping(result)
            self.last_audit = record
            self._load_input_audit(record)
            self._load_mask_audit(record)
            self._load_mineral_support(record)
            self._load_references(record)
            self._load_thresholds(record)
            self._load_risks(record)
            status = str(record.get("status", record.get("state", "Ready")))
            ready = bool(record.get("ready", status.casefold() not in {"blocked", "failed"}))
            review_required = status.casefold() == "reviewrequired"
            self.evidence_status.set_state(
                "审计通过" if ready else "待人工批准" if review_required else "审计阻断",
                "good" if ready else "warning" if review_required else "bad",
            )
            self.run_audit_state.set_state("通过" if ready else "待批准" if review_required else "阻断", "good" if ready else "warning" if review_required else "bad")
            mask = _to_mapping(record.get("mask"))
            approval = _to_mapping(mask.get("approval"))
            mask_ready = not bool(mask.get("blocked", False))
            approved = bool(approval.get("approved", self.mask_approved.isChecked()))
            self.run_mask_state.set_state(
                "质量通过 · 已批准" if mask_ready and approved else "质量通过 · 待批准" if mask_ready else "质量阻断",
                "good" if mask_ready and approved else "warning" if mask_ready else "bad",
            )
            trial = _to_mapping(record.get("threshold_trial"))
            threshold_ready = trial.get("status", "resolved" if record.get("thresholds") else "blocked") == "resolved"
            references_ready = bool(self._reference_records(record))
            confirmed = self.references_confirmed.isChecked()
            self.run_threshold_state.set_state(
                "已确认 · 可运行" if threshold_ready and confirmed else
                "阈值已优选 · 标准谱待确认" if threshold_ready else
                "阈值试算需处理",
                "good" if threshold_ready and confirmed else "warning" if threshold_ready else "bad",
            )
            self.run_page_start_button.setEnabled(ready and approved and threshold_ready and confirmed)
            self._set_step_state(4, "complete" if references_ready and confirmed else "warning" if references_ready else "blocked")
            self._set_step_state(6, "complete" if threshold_ready else "blocked")
            summary = record.get("summary", record.get("message", "输入、矿物能力和参考谱审计已完成。"))
            self.audit_summary.setText(str(summary))
            return record

        def _load_input_audit(self, audit: Mapping[str, Any]) -> None:
            values = audit.get("inputs", audit.get("datasets", ()))
            if isinstance(values, Mapping):
                rows = [(key, _to_mapping(value)) for key, value in values.items()]
            elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                rows = [(str(_to_mapping(value).get("role", "data")), _to_mapping(value)) for value in values]
            else:
                capability = _to_mapping(audit.get("capability", audit.get("sensor_capability_card")))
                rows = [("analysis", capability)] if capability else []
            self.input_table.setRowCount(len(rows))
            for row, (role, item) in enumerate(rows):
                shape = item.get("shape")
                if shape is None and item.get("lines") is not None:
                    shape = [item.get("lines"), item.get("samples"), item.get("bands")]
                wavelength = item.get("wavelength_range_nm")
                if wavelength is None and item.get("wavelength_min") is not None:
                    wavelength = [item.get("wavelength_min"), item.get("wavelength_max")]
                fields = (
                    role.upper(),
                    _display_value(shape),
                    _display_value(wavelength),
                    _display_value(item.get("data_physics", item.get("physics"))),
                    _display_value(item.get("status", "已读取")),
                )
                for column, text in enumerate(fields):
                    self.input_table.setItem(row, column, QTableWidgetItem(text))

        def _load_mask_audit(self, audit: Mapping[str, Any]) -> None:
            mask = _to_mapping(audit.get("mask", audit.get("mask_audit")))
            if not mask:
                return
            flags = mask.get("quality_flags", mask.get("flags", ()))
            blocked = bool(mask.get("blocked", False)) or any(
                str(_to_mapping(flag).get("severity", "")).casefold() in {"error", "blocked"} for flag in flags
            )
            self.mask_status.set_state("阻断" if blocked else "质量通过", "bad" if blocked else "good")
            metrics = [
                ("有效像元", mask.get("valid_pixels", mask.get("pixel_count"))),
                ("覆盖比例", mask.get("valid_fraction", mask.get("coverage_fraction"))),
                ("连通域", mask.get("component_count")),
                ("最大连通域", mask.get("largest_component_pixels")),
                ("内部填充率", mask.get("interior_pixel_fraction")),
                ("边缘风险", mask.get("edge_fraction")),
                ("托盘/背景泄漏", mask.get("background_leakage_fraction")),
            ]
            engine = _to_mapping(mask.get("engine"))
            if engine:
                metrics.append(("掩膜引擎", engine.get("used", engine.get("requested"))))
            metrics = [(key, value) for key, value in metrics if value is not None]
            self.mask_metrics.setRowCount(len(metrics))
            for row, (key, value) in enumerate(metrics):
                self.mask_metrics.setItem(row, 0, QTableWidgetItem(key))
                self.mask_metrics.setItem(row, 1, QTableWidgetItem(_display_value(value)))
            preview = mask.get("preview", mask.get("preview_path"))
            if preview and Path(str(preview)).is_file():
                pixmap = QPixmap(str(preview))
                if not pixmap.isNull():
                    self._set_mask_preview_pixmap(pixmap)
            approval = _to_mapping(mask.get("approval"))
            if approval:
                self.mask_approved.setChecked(bool(approval.get("approved", False)))
                self.mask_reviewer.setText(str(approval.get("reviewer") or self.mask_reviewer.text()))
                self.mask_notes.setText(str(approval.get("notes") or self.mask_notes.text()))

        def _load_mineral_support(self, audit: Mapping[str, Any]) -> None:
            support = audit.get("mineral_support", audit.get("observability", ()))
            if isinstance(support, Mapping):
                rows = [(key, _to_mapping(value)) for key, value in support.items()]
            elif isinstance(support, Sequence) and not isinstance(support, (str, bytes)):
                rows = [(str(_to_mapping(value).get("mineral_id", "")), _to_mapping(value)) for value in support]
            else:
                rows = []
            for mineral_id, item in rows:
                tree_item = self.mineral_items.get(mineral_id)
                if tree_item is None:
                    continue
                level = str(item.get("support_level", item.get("level", tree_item.text(1))))
                tree_item.setText(1, self._support_label(level))
                if item.get("reference_count") is not None:
                    tree_item.setText(3, str(item["reference_count"]))
                reasons = item.get("reasons", item.get("reason"))
                if reasons:
                    tree_item.setToolTip(1, _display_value(reasons))
                if level.casefold() in {"unsupported", "blocked"}:
                    tree_item.setCheckState(0, Qt.CheckState.Unchecked)
                    tree_item.setDisabled(True)

        @staticmethod
        def _reference_records(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
            candidates: Any = audit.get("selected_references", audit.get("reference_spectra"))
            if candidates is None:
                ensemble = _to_mapping(audit.get("library_ensemble"))
                candidates = ensemble.get("selected", ensemble.get("references", ()))
            if isinstance(candidates, Mapping):
                flattened = []
                for mineral_id, values in candidates.items():
                    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                        for value in values:
                            item = _to_mapping(value); item.setdefault("mineral_id", mineral_id); flattened.append(item)
                return flattened
            if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                return [_to_mapping(value) for value in candidates]
            return []

        def _load_references(self, audit: Mapping[str, Any]) -> None:
            self._all_reference_records = self._reference_records(audit)
            minerals = sorted(
                {str(item.get("mineral_id", item.get("primary_phase_id", ""))) for item in self._all_reference_records if item.get("mineral_id", item.get("primary_phase_id"))}
            )
            with QSignalBlocker(self.evidence_mineral):
                self.evidence_mineral.clear()
                self.evidence_mineral.addItem("全部入选标准谱", "")
                for mineral_id in minerals:
                    record = self.catalog.get("minerals", {}).get(mineral_id, {})
                    self.evidence_mineral.addItem(f"{record.get('display_name_zh', '')} {record.get('display_name_en', mineral_id)}", mineral_id)
            self._filter_evidence_curves()

        def _filter_evidence_curves(self, *_args) -> None:
            mineral_id = self.evidence_mineral.currentData() or ""
            records = getattr(self, "_all_reference_records", [])
            selected = [
                item for item in records
                if not mineral_id or str(item.get("mineral_id", item.get("primary_phase_id", ""))) == mineral_id
            ]
            self.spectrum_plot.set_curves(selected)
            self.reference_list.clear()
            for index, item in enumerate(selected):
                name = item.get("name", item.get("raw_name", item.get("sample_id", "标准谱")))
                source = item.get("source", item.get("source_id", "内置库"))
                score = item.get("score", item.get("reference_score"))
                suffix = f" · score {_display_value(score)}" if score is not None else ""
                icon_pixmap = QPixmap(12, 12)
                color = self.spectrum_plot.curves[index]["color"] if index < len(self.spectrum_plot.curves) else SpectrumPlot.PALETTE[index % len(SpectrumPlot.PALETTE)]
                icon_pixmap.fill(QColor(color))
                row_item = QListWidgetItem(QIcon(icon_pixmap), f"{name}\n{source}{suffix}")
                row_item.setToolTip("点击高亮该条实际入选标准谱")
                self.reference_list.addItem(row_item)
            if self.reference_list.count():
                self.reference_list.setCurrentRow(0)
            self.reference_curve_summary.setText(
                f"当前显示 {len(self.spectrum_plot.curves)} 条实际入选参考谱（不是合并谱）；"
                "颜色、线型和点击高亮用于辨识，曲线非常接近时仍可能局部重叠。"
            )

        def _load_thresholds(self, audit: Mapping[str, Any]) -> None:
            raw = audit.get("thresholds", audit.get("calibration", ()))
            self.current_thresholds = deepcopy(raw)
            self._threshold_editors.clear()
            rows: list[tuple[str, str, dict[str, Any]]] = []
            if isinstance(raw, Mapping):
                for group_id, policies in raw.items():
                    if not isinstance(policies, Mapping):
                        continue
                    for policy in ("conservative", "balanced", "sensitive"):
                        if isinstance(policies.get(policy), Mapping):
                            rows.append((str(group_id), policy, dict(policies[policy])))
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for index, value in enumerate(raw):
                    item = _to_mapping(value)
                    rows.append((str(item.get("group", item.get("group_id", f"组 {index + 1}"))), "balanced", item))

            group_names = {
                "carbonate_2300": "碳酸盐 2300",
                "calcium_sulfates": "钙硫酸盐",
                "white_mica_illite": "白云母/伊利石",
                "smectites": "蒙脱石族",
                "kaolin_2170_2205": "高岭石族",
            }
            policy_names = {"conservative": "严格", "balanced": "平衡", "sensitive": "敏感"}
            self.threshold_table.setRowCount(len(rows))
            for row, (group_id, policy, item) in enumerate(rows):
                self.threshold_table.setItem(row, 0, QTableWidgetItem(group_names.get(group_id, group_id)))
                self.threshold_table.item(row, 0).setToolTip(group_id)
                self.threshold_table.setItem(row, 1, QTableWidgetItem(policy_names.get(policy, policy)))
                final = item.get("final", item.get("resolved", item.get("value")))
                candidate_range = _to_mapping(item.get("candidate_range", item.get("search_range")))
                if isinstance(final, Mapping):
                    percentile = _safe_float(final.get("column_percentile"))
                    absolute = _safe_float(final.get("absolute_sam_rad", final.get("absolute_sam_threshold_rad")))
                else:
                    percentile = None
                    absolute = _safe_float(final)
                percentile_bounds = candidate_range.get("column_percentile", (0.001, 0.30))
                absolute_bounds = candidate_range.get("absolute_sam_rad", (0.01, 0.30))
                if percentile is not None and absolute is not None:
                    percentile_editor = QDoubleSpinBox()
                    percentile_editor.setDecimals(3)
                    percentile_editor.setSuffix(" %")
                    percentile_editor.setRange(100.0 * float(percentile_bounds[0]), 100.0 * float(percentile_bounds[1]))
                    percentile_editor.setSingleStep(0.25)
                    percentile_editor.setValue(100.0 * percentile)
                    percentile_editor.setMinimumWidth(108)
                    absolute_editor = QDoubleSpinBox()
                    absolute_editor.setDecimals(4)
                    absolute_editor.setSuffix(" rad")
                    absolute_editor.setRange(float(absolute_bounds[0]), float(absolute_bounds[1]))
                    absolute_editor.setSingleStep(0.0025)
                    absolute_editor.setValue(absolute)
                    absolute_editor.setMinimumWidth(118)
                    percentile_editor.valueChanged.connect(self._mark_threshold_dirty)
                    absolute_editor.valueChanged.connect(self._mark_threshold_dirty)
                    self.threshold_table.setCellWidget(row, 2, percentile_editor)
                    self.threshold_table.setCellWidget(row, 3, absolute_editor)
                    self._threshold_editors[(group_id, policy)] = (percentile_editor, absolute_editor)
                else:
                    self.threshold_table.setItem(row, 2, QTableWidgetItem("—"))
                    self.threshold_table.setItem(row, 3, QTableWidgetItem("—"))
                metrics = _to_mapping(item.get("selected_proxy_metrics"))
                candidates = metrics.get("candidate_pixels", candidate_range.get("evaluated_candidates", "—"))
                ready_pixels = metrics.get("feature_ready_pixels", "—")
                self.threshold_table.setItem(row, 4, QTableWidgetItem(_display_value(candidates)))
                self.threshold_table.setItem(row, 5, QTableWidgetItem(_display_value(ready_pixels)))
                status = str(item.get("status", "resolved"))
                source = str(item.get("source", "场景联合优选"))
                self.threshold_table.setItem(
                    row,
                    6,
                    QTableWidgetItem("已优选" if status == "resolved" else "需处理"),
                )
                self.threshold_table.item(row, 6).setToolTip(
                    f"{source}\n{item.get('resolved_reason', '')}"
                )
            trial = _to_mapping(audit.get("threshold_trial"))
            method = trial.get("method") or (
                "Catalog 安全边界 + 场景分层 + 空间伪影 + 矿物特征联合优选"
                if rows
                else "等待场景阈值试算"
            )
            warnings = trial.get("warnings", ())
            warning_count = len(warnings) if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)) else 0
            self.threshold_method_label.setText(
                f"优选方法：{method}。已生成 {len(rows)} 组档位参数"
                + (f"，其中 {warning_count} 项需要复核。" if warning_count else "。")
            )
            self._threshold_trial_dirty = False

        def _mark_threshold_dirty(self, *_args) -> None:
            self._threshold_trial_dirty = True
            self.threshold_method_label.setText("参数已调整；点击“应用调整并重新试算”后再运行。")
            self.run_threshold_state.set_state("阈值已调整 · 待重算", "warning")

        def _threshold_overrides_from_editors(self) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for (group_id, policy), (percentile, absolute) in self._threshold_editors.items():
                result.setdefault(group_id, {})[policy] = {
                    "column_percentile": percentile.value() / 100.0,
                    "absolute_sam_threshold_rad": absolute.value(),
                }
            return result

        def _rerun_threshold_trial(self) -> None:
            self._start_audit(target_step=6)

        def _load_risks(self, audit: Mapping[str, Any]) -> None:
            raw = audit.get("risk_summary", audit.get("risks", {}))
            rows: list[tuple[str, Any]] = []
            if isinstance(raw, Mapping):
                rows.extend((str(key), value) for key, value in raw.items())
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for item in raw:
                    record = _to_mapping(item)
                    rows.append((str(record.get("name", record.get("code", "风险"))), record.get("status", record.get("value", record.get("message")))))
            flags = audit.get("quality_flags", ())
            if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
                for flag in flags:
                    record = _to_mapping(flag)
                    rows.append((str(record.get("code", "质量提示")), record.get("message", record.get("severity"))))
            waterfall = audit.get("rejection_waterfall", {})
            if isinstance(waterfall, Mapping):
                for policy, groups in waterfall.items():
                    if not isinstance(groups, Mapping):
                        continue
                    for group, record in groups.items():
                        item = _to_mapping(record)
                        rows.append((
                            f"{policy}/{group} 拒绝瀑布",
                            f"候选 {item.get('sam_or_column_candidates', '—')} → "
                            f"证据后 {item.get('classified_before_spatial', '—')} → "
                            f"最终 {item.get('classified_final', '—')}",
                        ))
            self.risk_table.setRowCount(len(rows))
            for row, (name, value) in enumerate(rows):
                self.risk_table.setItem(row, 0, QTableWidgetItem(name))
                self.risk_table.setItem(row, 1, QTableWidgetItem(_display_value(value)))

        def load_run_result(self, result: Any) -> dict[str, Any]:
            record = _to_mapping(result)
            run_directory = record.get("run_directory", record.get("output_directory"))
            if run_directory:
                self.current_run_directory = Path(str(run_directory))
            self._apply_run_record(record)
            return record

        def load_run_directory(self, path: str | Path) -> Path:
            run = Path(path)
            manifest_path = run / "run_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["run_directory"] = str(run)
            self.current_run_directory = run
            self._apply_run_record(record)
            self._set_step(8)
            return run

        def _apply_run_record(self, record: Mapping[str, Any]) -> None:
            if record.get("thresholds"):
                self._load_thresholds(record)
            if record.get("rejection_waterfall"):
                self._load_risks(record)
            if record.get("selected_references") or record.get("library_ensemble"):
                self._load_references(record)
            quality = _to_mapping(record.get("quality", record.get("quality_report")))
            grade = quality.get("quality_grade", quality.get("grade", record.get("quality_grade", "—")))
            status = str(quality.get("status", record.get("status", "Completed")))
            bad = status.casefold() in {"failed", "blocked", "d"}
            warning = status.casefold() in {"warning", "c"} or str(grade).upper() == "C"
            self.quality_status.set_state(f"质量 {grade} · {status}", "bad" if bad else "warning" if warning else "good")
            issues = quality.get("issues", quality.get("warnings", ()))
            self.quality_text.setText(_display_value(issues) if issues else "成果完整性与发布门禁已完成。")
            previews = record.get("previews", {})
            if not isinstance(previews, Mapping):
                previews = {}
            self.preview_paths = {}
            run = self.current_run_directory
            for key, value in previews.items():
                path = Path(str(value))
                if run is not None and not path.is_file():
                    alternatives = (run / path, run / "previews" / path.name)
                    path = next((candidate for candidate in alternatives if candidate.is_file()), path)
                if path.is_file():
                    self.preview_paths[str(key)] = str(path)
            if run is not None and (run / "previews").is_dir():
                for path in sorted((run / "previews").glob("*.png")):
                    if str(path) not in self.preview_paths.values():
                        self.preview_paths.setdefault(path.stem, str(path))
            with QSignalBlocker(self.preview_combo):
                self.preview_combo.clear()
                for key in self.preview_paths:
                    self.preview_combo.addItem(self._preview_label(key), key)
            if self.preview_combo.count():
                preferred = f"comparison_{self.profile.currentData() or 'balanced'}"
                preferred_index = self.preview_combo.findData(preferred)
                self.preview_combo.setCurrentIndex(preferred_index if preferred_index >= 0 else 0)
                self._show_preview(self.preview_combo.currentData())
            self.open_output_button.setEnabled(run is not None and run.exists())
            log = record.get("log", record.get("messages"))
            if log:
                self.run_log.append(_display_value(log))

        @staticmethod
        def _preview_label(key: str) -> str:
            exact = {
                "background": "背景影像",
                "swir_background_1600nm": "SWIR 1600 nm 背景",
                "comparison_conservative": "严格档结果对比",
                "comparison_balanced": "平衡档结果对比（推荐）",
                "comparison_sensitive": "敏感档结果对比",
                "comparison_three_profiles": "严格 / 平衡 / 敏感三档对比",
                "material_mask": "岩心前景掩膜",
                "stripe_diagnosis": "固定列与条带风险诊断",
                "aloh_wavelength_subtype": "Al-OH 吸收位置分型",
            }
            if key in exact:
                return exact[key]
            if key.startswith("group_final_"):
                value = key.removeprefix("group_final_")
                for policy, label in (
                    ("_conservative", "严格档"),
                    ("_balanced", "平衡档"),
                    ("_sensitive", "敏感档"),
                ):
                    if value.endswith(policy):
                        group = value[: -len(policy)]
                        group_name = {
                            "carbonate_2300": "碳酸盐",
                            "calcium_sulfates": "钙硫酸盐",
                            "white_mica_illite": "白云母/伊利石",
                            "smectites": "蒙脱石族",
                            "kaolin_2170_2205": "高岭石族",
                        }.get(group, group.replace("_", " "))
                        return f"{group_name} · {label}"
            return key.replace("_", " ")

        def _show_preview(self, key: str | None) -> None:
            if not key:
                return
            path = self.preview_paths.get(key)
            if not path:
                return
            pixmap = QPixmap(path)
            if pixmap.isNull():
                self.result_preview.setText(f"无法读取预览：{path}")
                return
            self.preview_pixmap = pixmap
            self._scale_result_preview()

        def _scale_result_preview(self) -> None:
            if self.preview_pixmap is None:
                return
            zoom = float(self.preview_zoom.currentData() or 0.0)
            if zoom <= 0.0:
                viewport = self.result_scroll.viewport().size()
                pixmap = self.preview_pixmap.scaled(
                    max(1, viewport.width() - 12),
                    max(1, viewport.height() - 12),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            else:
                pixmap = self.preview_pixmap.scaled(
                    max(1, round(self.preview_pixmap.width() * zoom)),
                    max(1, round(self.preview_pixmap.height() * zoom)),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self.result_preview.setMinimumSize(pixmap.size())
            self.result_preview.resize(pixmap.size())
            self.result_preview.setPixmap(pixmap)

        def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
            super().resizeEvent(event)
            if hasattr(self, "preview_zoom") and float(self.preview_zoom.currentData() or 0.0) <= 0.0:
                self._scale_result_preview()
            if hasattr(self, "mask_zoom") and float(self.mask_zoom.currentData() or 0.0) <= 0.0:
                self._scale_mask_preview()

        def _open_output(self) -> None:
            if self.current_run_directory is not None and self.current_run_directory.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_run_directory.resolve())))

        def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
            if self.worker is not None and self.worker.isRunning():
                self._close_when_idle = True
                self._cancel()
                event.ignore()
                return
            event.accept()


    # Compatibility-friendly name for callers that expect the former module pattern.
    CoreSpecMainWindowV5 = CoreSpecV5Window


    def launch_desktop_v5() -> int:
        app = QApplication.instance() or QApplication(sys.argv)
        app.setApplicationName("CoreSpec Mapper V5")
        window = CoreSpecV5Window()
        window.show()
        return app.exec()


    def main() -> int:
        return launch_desktop_v5()


else:
    class CoreSpecV5Window:  # pragma: no cover - only used to provide a clear optional-dependency error
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 is required for the CoreSpec Mapper V5 desktop application")


    CoreSpecMainWindowV5 = CoreSpecV5Window


    def launch_desktop_v5() -> int:  # pragma: no cover
        raise RuntimeError("PySide6 is required for the CoreSpec Mapper V5 desktop application")


    def main() -> int:  # pragma: no cover
        return launch_desktop_v5()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
