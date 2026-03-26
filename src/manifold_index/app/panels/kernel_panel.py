"""
app/panels/kernel_panel.py — Kernel Database Builder tab.

Lets users pre-compute filling kernels for selected slopes or slope
ranges at a chosen qq order and view / manage the cached kernel DB.
"""

from __future__ import annotations

import os
import time
from math import gcd

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionTitle")
    return lbl


class KernelPanel(QWidget):
    """Full-page tab for building and inspecting the kernel cache."""

    # Emitted when builds finish so the status bar can update
    build_finished = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker = None
        self._queue: list[tuple[int, int, int]] = []
        self._queue_idx = 0
        self._build_t0 = 0.0
        self._setup_ui()
        self._refresh_table()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(16)

        # ── LEFT: controls ────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        t = QLabel("Kernel Database Builder")
        t.setObjectName("panelTitle")
        left.addWidget(t)

        sub = QLabel(
            "Pre-compute filling kernels for selected slopes so that "
            "Dehn filling runs instantly from cache."
        )
        sub.setObjectName("panelSubtitle")
        sub.setWordWrap(True)
        left.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        left.addWidget(sep)

        # ── Slope selection mode ──────────────────────────────
        left.addWidget(_section_label("Slope Selection"))

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(8)
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Single slope (P, Q)", "Slope range"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        left.addLayout(mode_row)

        # Single slope inputs
        self._single_row = QWidget()
        sh = QHBoxLayout(self._single_row)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.setSpacing(6)
        sh.addWidget(QLabel("P ="))
        self._p_single = QSpinBox()
        self._p_single.setRange(-50, 50)
        self._p_single.setValue(1)
        self._p_single.setFixedWidth(60)
        sh.addWidget(self._p_single)
        sh.addSpacing(10)
        sh.addWidget(QLabel("Q ="))
        self._q_single = QSpinBox()
        self._q_single.setRange(-50, 50)
        self._q_single.setValue(0)
        self._q_single.setFixedWidth(60)
        sh.addWidget(self._q_single)
        sh.addStretch()
        left.addWidget(self._single_row)

        # Range inputs
        self._range_box = QWidget()
        rv = QVBoxLayout(self._range_box)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)

        rp = QHBoxLayout()
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(6)
        rp.addWidget(QLabel("P ∈"))
        self._p_min = QSpinBox(); self._p_min.setRange(-50, 50); self._p_min.setValue(-3); self._p_min.setFixedWidth(60)
        rp.addWidget(self._p_min)
        rp.addWidget(QLabel("to"))
        self._p_max = QSpinBox(); self._p_max.setRange(-50, 50); self._p_max.setValue(3); self._p_max.setFixedWidth(60)
        rp.addWidget(self._p_max)
        rp.addStretch()
        rv.addLayout(rp)

        rq = QHBoxLayout()
        rq.setContentsMargins(0, 0, 0, 0)
        rq.setSpacing(6)
        rq.addWidget(QLabel("Q ∈"))
        self._q_min = QSpinBox(); self._q_min.setRange(-50, 50); self._q_min.setValue(-3); self._q_min.setFixedWidth(60)
        rq.addWidget(self._q_min)
        rq.addWidget(QLabel("to"))
        self._q_max = QSpinBox(); self._q_max.setRange(-50, 50); self._q_max.setValue(3); self._q_max.setFixedWidth(60)
        rq.addWidget(self._q_max)
        rq.addStretch()
        rv.addLayout(rq)

        self._coprime_cb = QCheckBox("Coprime only (gcd(P,Q) = 1)")
        self._coprime_cb.setChecked(True)
        rv.addWidget(self._coprime_cb)

        self._skip_cached_cb = QCheckBox("Skip slopes already cached at ≥ this order")
        self._skip_cached_cb.setChecked(True)
        rv.addWidget(self._skip_cached_cb)

        self._range_box.hide()
        left.addWidget(self._range_box)

        # ── qq order ──────────────────────────────────────────
        left.addWidget(_section_label("Truncation Order"))

        qq_row = QHBoxLayout()
        qq_row.setContentsMargins(0, 0, 0, 0)
        qq_row.setSpacing(6)
        qq_row.addWidget(QLabel("qq order:"))
        self._qq_spin = QSpinBox()
        self._qq_spin.setRange(8, 400)
        self._qq_spin.setValue(50)
        self._qq_spin.setFixedWidth(70)
        qq_row.addWidget(self._qq_spin)
        qq_row.addStretch()
        left.addLayout(qq_row)

        # ── Workers ───────────────────────────────────────────
        nw_row = QHBoxLayout()
        nw_row.setContentsMargins(0, 0, 0, 0)
        nw_row.setSpacing(6)
        nw_row.addWidget(QLabel("Workers:"))
        self._workers_spin = QSpinBox()
        cpu = os.cpu_count() or 4
        self._workers_spin.setRange(0, cpu)
        self._workers_spin.setValue(max(1, cpu - 2))
        self._workers_spin.setFixedWidth(60)
        self._workers_spin.setToolTip("0 = serial (no multiprocessing)")
        nw_row.addWidget(self._workers_spin)
        nw_row.addWidget(QLabel(f"(max {cpu})"))
        nw_row.addStretch()
        left.addLayout(nw_row)

        # ── Buttons ───────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(8)

        self._build_btn = QPushButton("Build Kernels  ▶")
        self._build_btn.setObjectName("primary")
        self._build_btn.setFixedHeight(36)
        self._build_btn.clicked.connect(self._on_build_clicked)
        btn_row.addWidget(self._build_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("secondary")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._cancel_btn)

        btn_row.addStretch()
        left.addLayout(btn_row)

        # ── Progress ──────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(14)
        self._progress.hide()
        left.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 11px;")
        self._status.setWordWrap(True)
        left.addWidget(self._status)

        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMaximumWidth(420)
        root.addWidget(left_w)

        # ── RIGHT: kernel DB table ────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        right.addWidget(_section_label("Cached Kernels"))

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["P", "Q", "qq order", "Source"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        right.addWidget(self._table, 1)

        tbl_btns = QHBoxLayout()
        tbl_btns.setContentsMargins(0, 0, 0, 0)
        tbl_btns.setSpacing(8)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("secondary")
        self._refresh_btn.clicked.connect(self._refresh_table)
        tbl_btns.addWidget(self._refresh_btn)

        self._clear_btn = QPushButton("Clear User Cache")
        self._clear_btn.setObjectName("secondary")
        self._clear_btn.clicked.connect(self._on_clear_cache)
        tbl_btns.addWidget(self._clear_btn)

        tbl_btns.addStretch()

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        tbl_btns.addWidget(self._count_label)

        right.addLayout(tbl_btns)

        right_w = QWidget()
        right_w.setLayout(right)
        root.addWidget(right_w, 1)

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_mode_changed(self, idx: int) -> None:
        self._single_row.setVisible(idx == 0)
        self._range_box.setVisible(idx == 1)

    # ------------------------------------------------------------------
    # Build slopes list
    # ------------------------------------------------------------------

    def _collect_slopes(self) -> list[tuple[int, int]]:
        """Return the list of (P, Q) slopes to build."""
        if self._mode_combo.currentIndex() == 0:
            p, q = self._p_single.value(), self._q_single.value()
            if p == 0 and q == 0:
                return []
            return [(p, q)]

        p_lo, p_hi = self._p_min.value(), self._p_max.value()
        q_lo, q_hi = self._q_min.value(), self._q_max.value()
        coprime_only = self._coprime_cb.isChecked()

        slopes: list[tuple[int, int]] = []
        for p in range(p_lo, p_hi + 1):
            for q in range(q_lo, q_hi + 1):
                if p == 0 and q == 0:
                    continue
                if coprime_only and gcd(abs(p), abs(q)) != 1:
                    continue
                slopes.append((p, q))
        return slopes

    # ------------------------------------------------------------------
    # Build action
    # ------------------------------------------------------------------

    @Slot()
    def _on_build_clicked(self) -> None:
        from manifold_index.app.workers import KernelBuilderWorker
        from manifold_index.core.kernel_cache import list_cached_kernels

        slopes = self._collect_slopes()
        if not slopes:
            self._status.setText("⚠ No valid slopes selected.")
            return

        qq = self._qq_spin.value()

        # Optionally skip already-cached slopes
        if self._skip_cached_cb.isChecked() and self._mode_combo.currentIndex() == 1:
            existing = set(list_cached_kernels())
            filtered: list[tuple[int, int]] = []
            for p, q in slopes:
                # Skip if any cached kernel for (p, q) with qq_order >= qq
                if any(ep == p and eq == q and eqq >= qq for ep, eq, eqq in existing):
                    continue
                filtered.append((p, q))
            skipped = len(slopes) - len(filtered)
            slopes = filtered
            if skipped:
                self._status.setText(f"Skipped {skipped} already-cached slope(s).")
            if not slopes:
                self._status.setText(
                    f"All {skipped} slope(s) already cached at qq ≥ {qq}. Nothing to do."
                )
                return

        # Build the queue: list of (P, Q, qq)
        self._queue = [(p, q, qq) for p, q in slopes]
        self._queue_idx = 0
        self._build_t0 = time.time()

        self._set_building(True)
        self._progress.setRange(0, len(self._queue))
        self._progress.setValue(0)
        self._progress.show()
        self._status.setText(
            f"Building {len(self._queue)} kernel(s) at qq = {qq}…"
        )

        self._launch_next()

    def _launch_next(self) -> None:
        """Launch the worker for the next slope in the queue."""
        from manifold_index.app.workers import KernelBuilderWorker

        if self._queue_idx >= len(self._queue):
            # All done
            elapsed = time.time() - self._build_t0
            n = len(self._queue)
            msg = f"✓  Built {n} kernel(s) in {elapsed:.1f} s"
            self._status.setText(msg)
            self._set_building(False)
            self._progress.hide()
            self._refresh_table()
            self.build_finished.emit(msg)
            return

        P, Q, qq = self._queue[self._queue_idx]
        n_workers = self._workers_spin.value()
        total = len(self._queue)
        self._status.setText(
            f"[{self._queue_idx + 1}/{total}]  Building kernel P={P}, Q={Q}, qq={qq}…"
        )

        worker = KernelBuilderWorker(P, Q, qq, n_workers or None)
        worker.status.connect(self._on_worker_status)
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        self._worker = worker
        worker.start()

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        total = len(self._queue)
        self._status.setText(f"[{self._queue_idx + 1}/{total}]  {msg}")

    @Slot()
    def _on_worker_finished(self) -> None:
        self._queue_idx += 1
        self._progress.setValue(self._queue_idx)
        self._launch_next()

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        self._status.setText(f"✗ Error at queue item {self._queue_idx + 1}: {msg}")
        # Continue with next item
        self._queue_idx += 1
        self._progress.setValue(self._queue_idx)
        self._launch_next()

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    @Slot()
    def _on_cancel_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel_requested = True
            self._worker.quit()
            self._worker.wait(3000)
        self._queue = []
        self._queue_idx = 0
        self._set_building(False)
        self._progress.hide()
        self._status.setText("Cancelled.")

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------

    def _set_building(self, busy: bool) -> None:
        self._build_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        self._mode_combo.setEnabled(not busy)
        self._qq_spin.setEnabled(not busy)
        self._workers_spin.setEnabled(not busy)

    # ------------------------------------------------------------------
    # Kernel DB table
    # ------------------------------------------------------------------

    @Slot()
    def _refresh_table(self) -> None:
        """Reload and display all cached kernels."""
        from manifold_index.core.kernel_cache import (
            list_cached_kernels,
            _DEFAULT_CACHE_DIR,
            _BUNDLED_KERNEL_DIR,
        )

        entries = list_cached_kernels()

        # Determine source for each entry
        user_dir = _DEFAULT_CACHE_DIR
        bundled_dir = _BUNDLED_KERNEL_DIR

        self._table.setRowCount(len(entries))
        for row, (p, q, qq) in enumerate(entries):
            self._table.setItem(row, 0, self._centered_item(str(p)))
            self._table.setItem(row, 1, self._centered_item(str(q)))
            self._table.setItem(row, 2, self._centered_item(str(qq)))

            # Check source
            from manifold_index.core.kernel_cache import _kernel_filename
            fname = _kernel_filename(p, q, qq)
            if user_dir.exists() and (user_dir / fname).exists():
                src = "User"
            elif bundled_dir.exists() and (bundled_dir / fname).exists():
                src = "Bundled"
            else:
                src = "?"
            self._table.setItem(row, 3, self._centered_item(src))

        self._count_label.setText(f"{len(entries)} kernel(s)")

    def _centered_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item

    # ------------------------------------------------------------------
    # Clear user cache
    # ------------------------------------------------------------------

    @Slot()
    def _on_clear_cache(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        from manifold_index.core.kernel_cache import (
            _DEFAULT_CACHE_DIR,
            clear_kernel_cache,
        )

        d = _DEFAULT_CACHE_DIR
        if not d.exists():
            self._status.setText("User cache directory does not exist.")
            return

        files = list(d.glob("kernel_*.pkl.gz"))
        if not files:
            self._status.setText("No user-cached kernels to clear.")
            return

        reply = QMessageBox.question(
            self,
            "Clear User Cache",
            f"Delete {len(files)} kernel file(s) from\n{d}?\n\n"
            "Bundled kernels shipped with the package will NOT be removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = 0
        for f in files:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass

        clear_kernel_cache()
        self._status.setText(f"Removed {removed} cached kernel file(s).")
        self._refresh_table()
