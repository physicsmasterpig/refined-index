"""app/datahub/generate_tab.py — Generate Tab for Data Hub.

BLUEPRINT §11.3.

Left pane:  task builder (type selector + parameter form + "Add to Queue").
Right pane: task queue list + Start/Pause/Resume/Cancel/Clear buttons
            + local cache summary table.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QRadioButton, QButtonGroup,
    QSizePolicy, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from manifold_index.app.widgets.no_scroll_spin import NoScrollSpinBox as QSpinBox

from manifold_index.services.datahub_service import DataHubService
from manifold_index.app.workers.generate_worker import GenerateWorker


class GenerateTab(QWidget):
    """Tab ②: Build kernels / I^ref cache / NC cache locally.

    Signals
    -------
    task_finished(str)  — human-readable completion message
    """

    task_finished = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: GenerateWorker | None = None
        self._task_queue: list[dict] = []
        self._pause_event = threading.Event()  # set = paused

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # ── Left pane: Task builder ───────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Data Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Filling Kernels", "I^ref Cache", "NC Cycle Cache"])
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo)
        type_row.addStretch(1)
        left_layout.addLayout(type_row)

        # Stacked parameter forms
        self._param_stack = QStackedWidget()

        # ── Kernels form ──────────────────────────────────────────────
        kernel_form = QWidget()
        kf = QVBoxLayout(kernel_form)
        kf.setSpacing(6)

        mode_row = QHBoxLayout()
        self._k_single = QRadioButton("Single slope")
        self._k_range  = QRadioButton("Range")
        self._k_single.setChecked(True)
        mode_grp = QButtonGroup(kernel_form)
        mode_grp.addButton(self._k_single)
        mode_grp.addButton(self._k_range)
        self._k_single.toggled.connect(self._on_kernel_mode_changed)
        mode_row.addWidget(self._k_single)
        mode_row.addWidget(self._k_range)
        mode_row.addStretch(1)
        kf.addLayout(mode_row)

        # Single slope row
        self._k_single_row = QWidget()
        sr = QHBoxLayout(self._k_single_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(QLabel("P ="))
        self._k_P = QSpinBox(); self._k_P.setRange(-50, 50); self._k_P.setValue(1); self._k_P.setFixedWidth(55)
        sr.addWidget(self._k_P)
        sr.addWidget(QLabel("Q ="))
        self._k_Q = QSpinBox(); self._k_Q.setRange(0, 50); self._k_Q.setValue(0); self._k_Q.setFixedWidth(55)
        sr.addWidget(self._k_Q)
        sr.addStretch(1)
        kf.addWidget(self._k_single_row)

        # Range row
        self._k_range_row = QWidget()
        rr = QHBoxLayout(self._k_range_row)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.addWidget(QLabel("P ∈ ["))
        self._k_P_lo = QSpinBox(); self._k_P_lo.setRange(-50, 0); self._k_P_lo.setValue(-3); self._k_P_lo.setFixedWidth(50)
        rr.addWidget(self._k_P_lo)
        rr.addWidget(QLabel(","))
        self._k_P_hi = QSpinBox(); self._k_P_hi.setRange(0, 50); self._k_P_hi.setValue(3); self._k_P_hi.setFixedWidth(50)
        rr.addWidget(self._k_P_hi)
        rr.addWidget(QLabel("]  Q ∈ [0,"))
        self._k_Q_hi = QSpinBox(); self._k_Q_hi.setRange(0, 50); self._k_Q_hi.setValue(3); self._k_Q_hi.setFixedWidth(50)
        rr.addWidget(self._k_Q_hi)
        rr.addWidget(QLabel("]"))
        rr.addStretch(1)
        kf.addWidget(self._k_range_row)
        self._k_range_row.setVisible(False)

        qq_row = QHBoxLayout()
        qq_row.addWidget(QLabel("qq order:"))
        self._k_qq = QSpinBox(); self._k_qq.setRange(4, 200); self._k_qq.setValue(50); self._k_qq.setFixedWidth(65)
        qq_row.addWidget(self._k_qq)
        self._k_coprime = QCheckBox("Coprime only"); self._k_coprime.setChecked(True)
        qq_row.addWidget(self._k_coprime)
        self._k_skip = QCheckBox("Skip cached ≥ qq"); self._k_skip.setChecked(True)
        qq_row.addWidget(self._k_skip)
        qq_row.addStretch(1)
        kf.addLayout(qq_row)
        kf.addStretch(1)
        self._param_stack.addWidget(kernel_form)  # index 0

        # ── I^ref form ────────────────────────────────────────────────
        iref_form = QWidget()
        irf = QVBoxLayout(iref_form)
        irf.setSpacing(6)
        census_row = QHBoxLayout()
        census_row.addWidget(QLabel("Census:"))
        self._ir_census_from = QSpinBox(); self._ir_census_from.setRange(3, 9999); self._ir_census_from.setValue(3); self._ir_census_from.setFixedWidth(65)
        census_row.addWidget(self._ir_census_from)
        census_row.addWidget(QLabel("to"))
        self._ir_census_to = QSpinBox(); self._ir_census_to.setRange(3, 9999); self._ir_census_to.setValue(50); self._ir_census_to.setFixedWidth(65)
        census_row.addWidget(self._ir_census_to)
        census_row.addStretch(1)
        irf.addLayout(census_row)
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("qq:"))
        self._ir_qq = QSpinBox(); self._ir_qq.setRange(4, 200); self._ir_qq.setValue(20); self._ir_qq.setFixedWidth(55)
        param_row.addWidget(self._ir_qq)
        param_row.addWidget(QLabel("m ±"))
        self._ir_m = QSpinBox(); self._ir_m.setRange(1, 50); self._ir_m.setValue(20); self._ir_m.setFixedWidth(50)
        param_row.addWidget(self._ir_m)
        param_row.addWidget(QLabel("e ±"))
        self._ir_e = QSpinBox(); self._ir_e.setRange(1, 50); self._ir_e.setValue(20); self._ir_e.setFixedWidth(50)
        param_row.addWidget(self._ir_e)
        self._ir_skip = QCheckBox("Skip existing"); self._ir_skip.setChecked(True)
        param_row.addWidget(self._ir_skip)
        param_row.addStretch(1)
        irf.addLayout(param_row)
        irf.addStretch(1)
        self._param_stack.addWidget(iref_form)  # index 1

        # ── NC form ───────────────────────────────────────────────────
        nc_form = QWidget()
        ncf = QVBoxLayout(nc_form)
        ncf.setSpacing(6)
        nc_census_row = QHBoxLayout()
        nc_census_row.addWidget(QLabel("Census:"))
        self._nc_from = QSpinBox(); self._nc_from.setRange(3, 9999); self._nc_from.setValue(3); self._nc_from.setFixedWidth(65)
        nc_census_row.addWidget(self._nc_from)
        nc_census_row.addWidget(QLabel("to"))
        self._nc_to = QSpinBox(); self._nc_to.setRange(3, 9999); self._nc_to.setValue(412); self._nc_to.setFixedWidth(65)
        nc_census_row.addWidget(self._nc_to)
        nc_census_row.addStretch(1)
        ncf.addLayout(nc_census_row)
        nc_param_row = QHBoxLayout()
        nc_param_row.addWidget(QLabel("qq:"))
        self._nc_qq = QSpinBox(); self._nc_qq.setRange(4, 200); self._nc_qq.setValue(20); self._nc_qq.setFixedWidth(55)
        nc_param_row.addWidget(self._nc_qq)
        nc_param_row.addWidget(QLabel("|P| ≤"))
        self._nc_p_max = QSpinBox(); self._nc_p_max.setRange(1, 50); self._nc_p_max.setValue(10); self._nc_p_max.setFixedWidth(50)
        nc_param_row.addWidget(self._nc_p_max)
        nc_param_row.addWidget(QLabel("Q ≤"))
        self._nc_q_max = QSpinBox(); self._nc_q_max.setRange(0, 50); self._nc_q_max.setValue(10); self._nc_q_max.setFixedWidth(50)
        nc_param_row.addWidget(self._nc_q_max)
        self._nc_skip = QCheckBox("Skip existing"); self._nc_skip.setChecked(True)
        nc_param_row.addWidget(self._nc_skip)
        nc_param_row.addStretch(1)
        ncf.addLayout(nc_param_row)
        ncf.addStretch(1)
        self._param_stack.addWidget(nc_form)  # index 2

        left_layout.addWidget(self._param_stack, 1)

        worker_row = QHBoxLayout()
        worker_row.addWidget(QLabel("Workers:"))
        self._n_workers = QSpinBox(); self._n_workers.setRange(1, 32); self._n_workers.setValue(4); self._n_workers.setFixedWidth(55)
        worker_row.addWidget(self._n_workers)
        self._add_btn = QPushButton("+ Add to Queue")
        self._add_btn.setProperty("class", "secondary")
        self._add_btn.clicked.connect(self._on_add_task)
        worker_row.addWidget(self._add_btn)
        worker_row.addStretch(1)
        left_layout.addLayout(worker_row)

        splitter.addWidget(left)

        # ── Right pane: queue + cache summary ─────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        queue_box = QGroupBox("Task Queue")
        qbl = QVBoxLayout(queue_box)

        self._queue_table = QTableWidget(0, 4)
        self._queue_table.setHorizontalHeaderLabels(["Type", "Parameters", "Status", "Progress"])
        self._queue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._queue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._queue_table.horizontalHeader().setStretchLastSection(True)
        self._queue_table.setColumnWidth(0, 90)
        self._queue_table.setColumnWidth(1, 200)
        self._queue_table.setColumnWidth(2, 80)
        qbl.addWidget(self._queue_table)

        self._queue_progress = QProgressBar()
        self._queue_progress.setValue(0)
        self._queue_progress.setVisible(False)
        qbl.addWidget(self._queue_progress)

        ctrl_row = QHBoxLayout()
        self._start_btn  = QPushButton("Start")
        self._pause_btn  = QPushButton("▐▐  Pause")
        self._resume_btn = QPushButton("▶  Resume")
        self._cancel_btn = QPushButton("Cancel")
        self._clear_btn  = QPushButton("Clear completed")
        for btn in (self._start_btn, self._pause_btn, self._resume_btn,
                    self._cancel_btn, self._clear_btn):
            btn.setProperty("class", "secondary")
            ctrl_row.addWidget(btn)
        ctrl_row.addStretch(1)
        self._start_btn.clicked.connect(self._on_start)
        self._pause_btn.clicked.connect(self._on_pause)
        self._resume_btn.clicked.connect(self._on_resume)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._clear_btn.clicked.connect(self._on_clear_completed)
        self._pause_btn.setEnabled(False)
        self._resume_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        qbl.addLayout(ctrl_row)

        self._queue_status = QLabel()
        self._queue_status.setProperty("class", "muted")
        self._queue_status.setVisible(False)
        qbl.addWidget(self._queue_status)

        right_layout.addWidget(queue_box, 2)

        # Cache summary
        cache_box = QGroupBox("Local Cache")
        cbl = QVBoxLayout(cache_box)
        self._cache_table = QTableWidget(3, 4)
        self._cache_table.setHorizontalHeaderLabels(["Type", "Count", "Size", "Last built"])
        self._cache_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._cache_table.verticalHeader().setVisible(False)
        self._cache_table.horizontalHeader().setStretchLastSection(True)
        for i, label in enumerate(["Kernels", "I^ref", "NC"]):
            self._cache_table.setItem(i, 0, QTableWidgetItem(label))
        cbl.addWidget(self._cache_table)

        refresh_row = QHBoxLayout()
        refresh_row.addStretch(1)
        self._cache_refresh_btn = QPushButton("Refresh")
        self._cache_refresh_btn.setProperty("class", "secondary")
        self._cache_refresh_btn.clicked.connect(self._refresh_cache_summary)
        refresh_row.addWidget(self._cache_refresh_btn)
        cbl.addLayout(refresh_row)

        right_layout.addWidget(cache_box, 1)
        splitter.addWidget(right)
        splitter.setSizes([400, 500])

        self._refresh_cache_summary()

    # ------------------------------------------------------------------
    # Type selector
    # ------------------------------------------------------------------

    def _on_type_changed(self, idx: int) -> None:
        self._param_stack.setCurrentIndex(idx)

    def _on_kernel_mode_changed(self, single: bool) -> None:
        self._k_single_row.setVisible(single)
        self._k_range_row.setVisible(not single)

    # ------------------------------------------------------------------
    # Task queue
    # ------------------------------------------------------------------

    def _on_add_task(self) -> None:
        idx = self._type_combo.currentIndex()
        task: dict = {"type_idx": idx, "status": "queued"}

        if idx == 0:  # Kernels
            if self._k_single.isChecked():
                P, Q = self._k_P.value(), self._k_Q.value()
                task.update({
                    "task": "kernels",
                    "slopes": [(P, Q)],
                    "qq": self._k_qq.value(),
                    "skip": self._k_skip.isChecked(),
                    "n_workers": self._n_workers.value(),
                    "label": f"Kernels ({P},{Q}) qq={self._k_qq.value()}",
                })
            else:
                slopes = [
                    (P, Q)
                    for P in range(self._k_P_lo.value(), self._k_P_hi.value() + 1)
                    for Q in range(0, self._k_Q_hi.value() + 1)
                    if not (P == 0 and Q == 0)
                ]
                if self._k_coprime.isChecked():
                    from math import gcd
                    slopes = [(P, Q) for P, Q in slopes if gcd(abs(P), abs(Q)) == 1]
                task.update({
                    "task": "kernels",
                    "slopes": slopes,
                    "qq": self._k_qq.value(),
                    "skip": self._k_skip.isChecked(),
                    "n_workers": self._n_workers.value(),
                    "label": f"Kernels P∈[{self._k_P_lo.value()},{self._k_P_hi.value()}] Q∈[0,{self._k_Q_hi.value()}] qq={self._k_qq.value()}",
                })

        elif idx == 1:  # I^ref
            names = [
                f"m{n:03d}"
                for n in range(self._ir_census_from.value(), self._ir_census_to.value() + 1)
            ]
            task.update({
                "task": "iref",
                "manifold_names": names,
                "qq": self._ir_qq.value(),
                "m_max": self._ir_m.value(),
                "e_max": self._ir_e.value(),
                "skip": self._ir_skip.isChecked(),
                "n_workers": self._n_workers.value(),
                "label": f"I^ref m{self._ir_census_from.value():03d}–m{self._ir_census_to.value():03d} qq={self._ir_qq.value()}",
            })

        else:  # NC
            names = [
                f"m{n:03d}"
                for n in range(self._nc_from.value(), self._nc_to.value() + 1)
            ]
            task.update({
                "task": "nc",
                "manifold_names": names,
                "qq": self._nc_qq.value(),
                "p_max": self._nc_p_max.value(),
                "q_max": self._nc_q_max.value(),
                "skip": self._nc_skip.isChecked(),
                "n_workers": self._n_workers.value(),
                "label": f"NC m{self._nc_from.value():03d}–m{self._nc_to.value():03d} qq={self._nc_qq.value()}",
            })

        self._task_queue.append(task)
        self._add_queue_row(task)
        self._start_btn.setEnabled(True)

    def _add_queue_row(self, task: dict) -> None:
        row = self._queue_table.rowCount()
        self._queue_table.insertRow(row)
        self._queue_table.setItem(row, 0, QTableWidgetItem(task.get("task", "?")))
        self._queue_table.setItem(row, 1, QTableWidgetItem(task.get("label", "")))
        self._queue_table.setItem(row, 2, QTableWidgetItem(task.get("status", "queued")))
        self._queue_table.setItem(row, 3, QTableWidgetItem(""))
        task["_row"] = row

    # ------------------------------------------------------------------
    # Worker control
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        queued = [t for t in self._task_queue if t["status"] == "queued"]
        if not queued:
            return

        task = queued[0]
        task["status"] = "running"
        self._set_row_status(task["_row"], "running")
        self._queue_progress.setVisible(True)
        self._queue_progress.setValue(0)
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)

        self._worker = GenerateWorker(
            task         = task["task"],
            params       = task,
            pause_event  = self._pause_event,
            parent       = self,
        )
        self._worker.status.connect(lambda s: self._update_run_status(task, s))
        self._worker.progress.connect(lambda d, t: self._queue_progress.setValue(
            int(100 * d / t) if t > 0 else 0
        ))
        self._worker.finished.connect(lambda r: self._on_worker_finished(task, r))
        self._worker.error.connect(lambda e: self._on_worker_error(task, e))
        self._worker.start()

    def _on_pause(self) -> None:
        self._pause_event.set()
        self._pause_btn.setEnabled(False)
        self._resume_btn.setEnabled(True)
        self._queue_status.setText("Paused.")
        self._queue_status.setVisible(True)

    def _on_resume(self) -> None:
        self._pause_event.clear()
        self._pause_btn.setEnabled(True)
        self._resume_btn.setEnabled(False)
        self._queue_status.setVisible(False)

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._pause_event.clear()
        self._pause_btn.setEnabled(False)
        self._resume_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)

    def _on_clear_completed(self) -> None:
        completed = [t for t in self._task_queue if t["status"] in ("done", "error", "cancelled")]
        for task in completed:
            self._task_queue.remove(task)
        self._rebuild_queue_table()

    def _update_run_status(self, task: dict, msg: str) -> None:
        self._queue_status.setText(msg)
        self._queue_status.setVisible(True)

    def _on_worker_finished(self, task: dict, results: list) -> None:
        # build_kernels returns (P, Q, status) 3-tuples; iref/nc return (key, status) 2-tuples
        def _status_str(item: tuple) -> str:
            return item[-1]  # last element is always the status string

        done = sum(
            1 for item in results
            if not _status_str(item).startswith("error")
            and _status_str(item) != "cancelled"
        )
        task["status"] = "done"
        self._set_row_status(task["_row"], f"done ({done})")
        self._queue_progress.setVisible(False)
        self._pause_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._start_btn.setEnabled(bool([t for t in self._task_queue if t["status"] == "queued"]))
        self._refresh_cache_summary()
        self.task_finished.emit(f"{task.get('label','Task')} complete ({done} items)")

    def _on_worker_error(self, task: dict, msg: str) -> None:
        task["status"] = "error"
        self._set_row_status(task["_row"], "error")
        self._queue_progress.setVisible(False)
        self._queue_status.setText(f"Error: {msg}")
        self._queue_status.setVisible(True)
        self._pause_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        # Re-enable Start for any remaining queued tasks
        self._start_btn.setEnabled(True)

    def _set_row_status(self, row: int, status: str) -> None:
        item = self._queue_table.item(row, 2)
        if item:
            item.setText(status)

    def _rebuild_queue_table(self) -> None:
        self._queue_table.setRowCount(0)
        for task in self._task_queue:
            self._add_queue_row(task)
            self._set_row_status(task["_row"], task["status"])

    # ------------------------------------------------------------------
    # Cache summary
    # ------------------------------------------------------------------

    def _refresh_cache_summary(self) -> None:
        try:
            cache = DataHubService.list_local_cache()
        except Exception:
            return

        for i, key in enumerate(["kernels", "iref", "nc"]):
            info = cache.get(key, {})
            count = info.get("count", 0)
            size_b = info.get("size_bytes", 0)
            size_str = f"{size_b / 1e6:.1f} MB" if size_b else "—"
            self._cache_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self._cache_table.setItem(i, 2, QTableWidgetItem(size_str))
            self._cache_table.setItem(i, 3, QTableWidgetItem("—"))

