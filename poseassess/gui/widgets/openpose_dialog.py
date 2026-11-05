"""\"Get OpenPose…\" dialog of "1. Project": set up a local OpenPose BODY_25B install.

OpenPose (CMU) is licensed for academic / non-profit noncommercial research only and
must not be redistributed, so PoseAssess never bundles it. This dialog helps the user

(a) point PoseAssess at an OpenPose folder they already have,
(b) download the OFFICIAL OpenPose 1.7.0 GPU release from CMU's GitHub onto this PC,
    after accepting the licence (`openpose_setup.download_openpose`),
(c) import their own BODY_25B weights (``.caffemodel``) into it; the matching official
    ``pose_deploy.prototxt`` is downloaded (`openpose_setup.install_caffemodel`).

Downloads and copies run on a QThread: the GUI thread never waits for them.

Also holds the status helpers shared with the project page (`openpose_report`,
`status_html`).
"""
from __future__ import annotations

import html
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from poseassess.core import openpose_setup as ops

GOOD, WARN, MUTED = "#3ad35a", "#e0a94c", "#8a8f98"
#: Status line of "1. Project" when no usable OpenPose install is found.
ENTER_FOLDER = "Enter the OpenPose folder (e.g. C:\\openpose) or use Get OpenPose…"
#: Download size shown on the button (MB = 2**20 bytes, like Explorer).
DOWNLOAD_MB = round(ops.OFFICIAL_ZIP_SIZE / 2**20)

#: (QThread, worker) of downloads still running after their dialog was closed (they were
#: cancelled); kept alive until they end, Python must never destroy a running QThread.
_ORPHANS: list[tuple[QThread, QObject]] = []

INTRO_HTML = f"""
<p><b>OpenPose · BODY_25B</b> runs from your own OpenPose installation on this PC.
It needs:</p>
<ul>
<li>an <b>NVIDIA graphics card</b> with driver
{'.'.join(map(str, ops.MIN_DRIVER))} or newer (RTX 50 series / Blackwell cards do not
work with OpenPose {ops.OFFICIAL_VERSION});</li>
<li>the OpenPose {ops.OFFICIAL_VERSION} <b>GPU build</b> for Windows: download it below,
or use a folder you already have (e.g. C:\\openpose);</li>
<li>the <b>BODY_25B weights</b> (pose_iter_….caffemodel), which <b>you supply</b>: CMU no
longer offers a download. Importing them also downloads the matching official
pose_deploy.prototxt.</li>
</ul>
<p><b>License:</b> OpenPose (Carnegie Mellon University) may be used for academic or
non-profit <b>noncommercial research only</b> and must not be redistributed, so it is not
included in PoseAssess. <a href="{ops.LICENSE_URL}">Read the OpenPose license</a>.</p>
"""


# --------------------------------------------------------------------------- #
# Status helpers (Qt-free; shared with the project page)
# --------------------------------------------------------------------------- #
def breakable(text: str) -> str:
    """HTML-escaped `text` with zero-width spaces after path separators, so a word-wrapped
    QLabel wraps long paths instead of growing wider."""
    return (html.escape(text, quote=False)
            .replace("\\", "\\\u200b").replace("/", "/\u200b"))


def status_html(lines: list[tuple[str, str]]) -> str:
    """Rich text for a QLabel: one ``(text, colour)`` per line."""
    return "<br>".join(f"<span style='color:{c}'>{breakable(t)}</span>" for t, c in lines)


def install_lines(st: ops.InstallStatus, *, all_errors: bool = False) -> list[tuple[str, str]]:
    """``(text, colour)`` lines for `st`: its summary, then (optionally) the other errors and
    the warnings."""
    lines = [(st.summary(), GOOD if st.ok else WARN)]
    if all_errors:
        lines += [(f"Also: {e}", WARN) for e in st.errors[1:]]
    lines += [(f"Warning: {w}", WARN) for w in st.warnings]
    return lines


def fix_hint(st: ops.InstallStatus) -> str | None:
    """How to fix the first problem of a found but incomplete install (None: nothing to add,
    e.g. the error already says it, or the install is fine)."""
    if st.ok or not st.errors:
        return None
    first = st.errors[0]
    if "CPU-only" in first:
        return (f"Fix: use the OpenPose {ops.OFFICIAL_VERSION} GPU build (Get OpenPose… "
                "downloads it).")
    if "rename" in first:
        return ("Fix: rename the file, or use Get OpenPose… > Import BODY_25B model file "
                "(.caffemodel)…")
    if ops.CAFFEMODEL_NAME in first or ops.PROTOTXT_NAME in first:
        return ("Fix: Get OpenPose… > Import BODY_25B model file (.caffemodel)… (copies your "
                "weights under the right name and gets the matching prototxt).")
    return None


@dataclass
class OpenPoseReport:
    """The OpenPose install a run would use (`openpose_report`)."""
    ok: bool
    root: Path | None
    lines: list[tuple[str, str]] = field(default_factory=list)   # (text, colour)


def _origin_note(origin: str) -> str:
    return "auto-detected" if origin.startswith("found at") else origin


def openpose_report(setting: str) -> OpenPoseReport:
    """Status of the install a run would use for the folder `setting` (empty: auto-detect).

    Same search as the run (`resolve_openpose_dir`); only looks at files (no GPU check).
    It looks into several folders, some may be on slow or disconnected drives: call it
    off the GUI thread. Not ok (no install found, or the one found is incomplete): the
    lines end with `ENTER_FOLDER`.
    """
    text = (setting or "").strip().strip('"').strip()
    root, origin = ops.resolve_openpose_dir(text)
    if root is None:
        where = ((f"{ops.EXE_NAME} not found in {text}", WARN) if text
                 else (origin, MUTED))   # where auto-detect looked
        return OpenPoseReport(False, None, [(ENTER_FOLDER, WARN), where])
    lines: list[tuple[str, str]] = []
    if text and ops.find_root(text) is None:
        lines.append((f"{ops.EXE_NAME} not found in {text}; using {root} instead "
                      f"({_origin_note(origin)})", WARN))
    st = ops.check_install(root)
    found = install_lines(st)
    if not text:
        summary, colour = found[0]
        found[0] = (f"{summary} - {_origin_note(origin)}", colour)
    if not st.ok:
        hint = fix_hint(st)
        found[1:1] = [(hint, WARN)] if hint else []
        found.append((ENTER_FOLDER, WARN))
    return OpenPoseReport(st.ok, root, lines + found)


def default_download_parent() -> Path:
    """Where the download goes by default: ``C:\\`` (gives ``C:\\openpose``, which
    PoseAssess finds by itself) on Windows, else the home folder."""
    return Path("C:\\") if os.name == "nt" else Path.home()


def stop_jobs(timeout_ms: int = 10000) -> None:
    """Cancel downloads left running by closed dialogs and wait for them (app shutdown)."""
    for thread, job in list(_ORPHANS):
        job.cancel()
        thread.quit()
        thread.wait(timeout_ms)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class _Job(QObject):
    """Runs ``fn(progress, cancelled)`` on a QThread; reports progress and the result."""
    progress = Signal(int, int)    # done, total (bytes)
    done = Signal(object, str)     # result, error text ("" = success)

    def __init__(self, fn: Callable[[Callable[[int, int], None], Callable[[], bool]], object]):
        super().__init__()
        self._fn = fn
        self._cancel = threading.Event()
        self._last = 0.0

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _progress(self, done: int, total: int) -> None:
        now = time.monotonic()  # at most ~10 updates / s (one call per 64 KB chunk)
        if done >= total or now - self._last >= 0.1:
            self._last = now
            self.progress.emit(int(done), int(total))

    def run(self) -> None:
        try:
            result = self._fn(self._progress, self.cancelled)
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.done.emit(None, str(e) or type(e).__name__)
            return
        self.done.emit(result, "")


# --------------------------------------------------------------------------- #
# Dialog
# --------------------------------------------------------------------------- #
class OpenPoseDialog(QDialog):
    """Find, download or complete a local OpenPose BODY_25B install.

    `root`: the install known so far (e.g. the auto-detected one), or None.
    After the dialog closed, `result_root` is the install the user chose, downloaded
    or imported weights into (None when they did none of that).
    """

    def __init__(self, parent=None, root: str | Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Get OpenPose (BODY_25B)")
        self.setMinimumWidth(520)
        self._root: Path | None = ops.find_root(root) if root else None
        self._picked = False
        self._thread: QThread | None = None
        self._job: _Job | None = None
        self._kind = ""   # "download" | "import" while a job runs

        lay = QVBoxLayout(self)
        intro = QLabel(INTRO_HTML)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        intro.setOpenExternalLinks(True)
        lay.addWidget(intro)

        self.accept_box = QCheckBox(
            "I have read the OpenPose license and use it for noncommercial research only")
        self.accept_box.toggled.connect(self._update_buttons)
        lay.addWidget(self.accept_box)

        self.b_existing = QPushButton("Use an existing OpenPose folder…")
        self.b_existing.clicked.connect(self._use_existing)
        self.b_download = QPushButton(
            f"Download official OpenPose {ops.OFFICIAL_VERSION} ({DOWNLOAD_MB} MB)…")
        self.b_download.setToolTip(
            f"Downloads {ops.OFFICIAL_ZIP_NAME} from CMU's GitHub releases and unpacks it into "
            "a folder 'openpose' inside the folder you choose (C:\\ gives C:\\openpose). "
            "About 1.5 GB of free disk space is needed. Accept the license first.")
        self.b_download.clicked.connect(self._download)
        get_row = QHBoxLayout()
        get_row.addWidget(self.b_existing)
        get_row.addWidget(self.b_download)
        get_row.addStretch(1)
        lay.addLayout(get_row)

        model_box = QGroupBox("BODY_25B model")
        mb = QVBoxLayout(model_box)
        self.variant_combo = QComboBox()
        for key, v in ops.BODY25B_VARIANTS.items():
            self.variant_combo.addItem(v.title, key)
        self.variant_combo.setCurrentIndex(self.variant_combo.findData(ops.DEFAULT_VARIANT))
        self.variant_combo.setToolTip(
            "Which official BODY_25B network your .caffemodel belongs to: the prototxt "
            "must match the weights.")
        self.b_import = QPushButton("Import BODY_25B model file (.caffemodel)…")
        self.b_import.setToolTip(
            f"Copies the file as {ops.MODEL_SUBDIR / ops.CAFFEMODEL_NAME} (that exact name) "
            "and downloads the matching official pose_deploy.prototxt.")
        self.b_import.clicked.connect(self._import_model)
        var_row = QHBoxLayout()
        var_row.addWidget(QLabel("Variant"))
        var_row.addWidget(self.variant_combo, 1)
        mb.addLayout(var_row)
        note = QLabel("Choose the network your .caffemodel belongs to: the weights must match "
                      "the prototxt. The variant of an installed official prototxt is "
                      "preselected.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED};")
        mb.addWidget(note)
        imp_row = QHBoxLayout()
        imp_row.addWidget(self.b_import)
        imp_row.addStretch(1)
        mb.addLayout(imp_row)
        lay.addWidget(model_box)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.setToolTip("Stop the download (the partial files are removed).")
        self.b_cancel.setVisible(False)
        self.b_cancel.clicked.connect(self._cancel_download)
        prog_row = QHBoxLayout()
        prog_row.addWidget(self.progress, 1)
        prog_row.addWidget(self.b_cancel)
        lay.addLayout(prog_row)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText)
        sp = self.status.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Ignored)  # long paths wrap, never widen the dialog
        self.status.setSizePolicy(sp)
        lay.addWidget(self.status)
        lay.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        if self._root is not None:
            self._set_root(self._root, picked=False)
        else:
            self._show_lines([("No OpenPose folder chosen yet: use an existing one or download "
                               "OpenPose, then import the BODY_25B model file.", MUTED)])
        self._update_buttons()

    # ---- public ----
    def result_root(self) -> Path | None:
        """The install chosen, downloaded or completed in this dialog, else None."""
        return self._root if self._picked else None

    def busy(self) -> bool:
        return self._thread is not None

    # ---- state ----
    def _update_buttons(self) -> None:
        idle = not self.busy()
        self.b_existing.setEnabled(idle)
        self.b_download.setEnabled(idle and self.accept_box.isChecked())
        self.b_import.setEnabled(idle and self._root is not None)
        self.variant_combo.setEnabled(idle)
        self.accept_box.setEnabled(idle)

    def _show_lines(self, lines: list[tuple[str, str]]) -> None:
        self.status.setText(status_html(lines))

    def _set_root(self, root: Path, *, picked: bool, note: str = "") -> None:
        self._root = Path(root)
        self._picked = self._picked or picked
        st = ops.check_install(self._root)
        if st.variant:
            self.variant_combo.setCurrentIndex(self.variant_combo.findData(st.variant))
        lines = [(note, MUTED)] if note else []
        lines += install_lines(st, all_errors=True)
        if not st.ok and any(ops.CAFFEMODEL_NAME in e or ops.PROTOTXT_NAME in e
                             for e in st.errors):
            lines.append(("Next: import your BODY_25B weights with \"Import BODY_25B model "
                          "file\" above.", MUTED))
        self._show_lines(lines)
        self._update_buttons()

    # ---- (a) existing folder ----
    def _use_existing(self) -> None:
        start = str(self._root or default_download_parent())
        d = QFileDialog.getExistingDirectory(
            self, f"Select the OpenPose folder (with bin\\{ops.EXE_NAME})", start)
        if not d:
            return
        root = ops.find_root(d)
        if root is None:
            QMessageBox.warning(self, "OpenPose", f"{ops.EXE_NAME} was not found in\n{d}\n\n"
                                f"Choose the OpenPose folder: it contains bin\\{ops.EXE_NAME} "
                                "and models\\.")
            return
        self._set_root(root, picked=True)

    # ---- (b) download ----
    def _download(self) -> None:
        if not self.accept_box.isChecked() or self.busy():
            return
        parent = QFileDialog.getExistingDirectory(
            self, "Download OpenPose into this folder (a folder 'openpose' is created in it)",
            str(default_download_parent()))
        if not parent:
            return
        existing = ops.find_root(Path(parent) / "openpose")
        if existing is not None:
            if QMessageBox.question(
                    self, "Download OpenPose", f"There already is an OpenPose in\n{existing}\n\n"
                    "Use it instead of downloading?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
                self._set_root(existing, picked=True)
            return
        self._show_lines([(f"Downloading OpenPose {ops.OFFICIAL_VERSION} into "
                           f"{Path(parent) / 'openpose'}…", MUTED)])
        self._start("download", lambda progress, cancelled: ops.download_openpose(
            parent, progress=progress, cancelled=cancelled))

    def _cancel_download(self) -> None:
        if self._job is not None and self._kind == "download":
            self._job.cancel()
            self.b_cancel.setEnabled(False)
            self._show_lines([("Cancelling…", MUTED)])

    def _on_progress(self, done: int, total: int) -> None:
        if self._job is None or self._job.cancelled():
            return  # late update of a cancelled download: keep "Cancelling…"
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        if done >= total:
            self.progress.setRange(0, 0)  # unpacking: no progress information
            self.progress.setFormat("Unpacking…")
            self._show_lines([(f"Unpacking… (download complete; about "
                               f"{ops.OFFICIAL_UNPACKED_SIZE / 1e9:.0f} GB to unpack)", MUTED)])
            return
        self.progress.setRange(0, 1000)
        self.progress.setValue(int(done * 1000 // total))
        self.progress.setFormat(f"%p%  ({done / 2**20:.0f} of {total / 2**20:.0f} MB)")

    # ---- (c) import weights ----
    def _import_model(self) -> None:
        root = self._root
        if root is None or self.busy():
            return
        src, _ = QFileDialog.getOpenFileName(
            self, "Select the BODY_25B weights (.caffemodel)", "",
            "Caffe model (*.caffemodel);;All files (*)")
        if not src:
            return
        variant = self.variant_combo.currentData()
        self._show_lines([(f"Installing {Path(src).name} into "
                           f"{ops.model_dir(root)}…", MUTED)])
        self._start("import", lambda progress, cancelled: ops.install_caffemodel(
            root, src, variant=variant))

    # ---- jobs ----
    def _start(self, kind: str, fn) -> None:
        self._kind = kind
        self._job = _Job(fn)
        self._thread = QThread()
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        # direct: quits even when no GUI event loop runs (shutdown)
        self._job.done.connect(self._thread.quit, Qt.DirectConnection)
        self._job.progress.connect(self._on_progress)
        self._job.done.connect(self._on_job_done)
        self._thread.finished.connect(self._on_thread_finished)
        self.progress.setRange(0, 0)
        self.progress.setFormat("%p%")
        self.progress.setVisible(True)
        self.b_cancel.setVisible(kind == "download")
        self.b_cancel.setEnabled(kind == "download")
        self._update_buttons()
        self._thread.start()

    def _on_job_done(self, result, error: str) -> None:
        kind = self._kind
        cancelled = self._job is not None and self._job.cancelled()
        if error:
            if kind == "download" and cancelled:
                self._show_lines([("Download cancelled; the partial files were removed.",
                                   MUTED)])
            else:
                title = "Download OpenPose" if kind == "download" else "Import BODY_25B model"
                QMessageBox.warning(self, title, error)
                if self._root is not None:
                    self._set_root(self._root, picked=False)
                else:
                    self._show_lines([(f"{title} failed: {error}", WARN)])
            return
        if kind == "download":
            self._set_root(result, picked=True,
                           note=f"OpenPose {ops.OFFICIAL_VERSION} downloaded to {result}.")
        else:
            root = result.root if result is not None and result.root else self._root
            self._set_root(root, picked=True, note="BODY_25B model file installed.")

    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.wait()
        self._thread = None
        self._job = None
        self._kind = ""
        self.progress.setVisible(False)
        self.b_cancel.setVisible(False)
        self._update_buttons()

    # ---- closing ----
    def done(self, r: int) -> None:  # Close button, Esc, window close
        if self.busy():
            if self._kind != "download":
                QMessageBox.information(self, "Get OpenPose",
                                        "Please wait until the model file is installed.")
                return
            if QMessageBox.question(
                    self, "Get OpenPose", "Stop the OpenPose download?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            self._orphan_job()
        super().done(r)

    def _orphan_job(self) -> None:
        """Cancel the download and let it end on its own (it removes its partial files):
        its thread is kept alive in `_ORPHANS` until then, its signals no longer reach
        this dialog."""
        thread, job = self._thread, self._job
        job.cancel()
        for signal, slot in ((job.done, self._on_job_done), (job.progress, self._on_progress),
                             (thread.finished, self._on_thread_finished)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        _ORPHANS[:] = [(t, j) for t, j in _ORPHANS if t.isRunning()]
        _ORPHANS.append((thread, job))
        self._thread = None
        self._job = None
        self._kind = ""
