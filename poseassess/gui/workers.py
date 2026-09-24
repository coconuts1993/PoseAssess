"""Background workers so long pipeline stages never freeze the UI.

`PipelineWorker` runs one or more pipeline stages on a QThread and streams log
lines + per-stage results back to the GUI thread via signals.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from poseassess.core.pipeline import Pipeline, StageResult


def silence_matplotlib_gui() -> None:
    """Force matplotlib to a headless backend and disable blocking windows.

    Pose2Sim stages (synchronization, filtering) may create figures / call
    plt.show(); from a Qt worker thread that raises "GUI outside main thread"
    and can crash the app. Our own plots use FigureCanvasQTAgg directly, so
    switching pyplot's backend to Agg is safe for the GUI. Call once per worker.
    """
    try:
        os.environ["MPLBACKEND"] = "Agg"  # also catch modules importing plt later
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None
        plt.pause = lambda *a, **k: None
        plt.figure = _noop_figure(plt.figure)
    except Exception:  # noqa: BLE001 - never let this break a run
        pass


def _noop_figure(orig):
    """Keep plt.figure working but never let it try to raise a GUI window."""
    def _fig(*a, **k):
        k.pop("num", None)
        return orig(*a, **k)
    return _fig


class _LogStream:
    """File-like object that forwards captured stdout/stderr to a Qt log signal.

    Splits on newlines AND carriage returns (tqdm progress bars use `\\r`), and
    throttles to the latest line every `min_interval` seconds so a per-frame
    stage (OpenSim IK prints one line/frame) shows live progress without
    flooding the GUI's signal queue.
    """
    def __init__(self, emit, min_interval: float = 0.15):
        self._emit = emit
        self._buf = ""
        self._last = 0.0
        self._min = min_interval

    def write(self, s):  # noqa: D401
        if not s:
            return
        try:
            self._buf += s
        except Exception:  # noqa: BLE001
            return
        if "\n" not in s and "\r" not in s:
            return
        parts = self._buf.replace("\r", "\n").split("\n")
        self._buf = parts[-1]
        lines = [p.rstrip() for p in parts[:-1] if p.strip()]
        if not lines:
            return
        now = time.time()
        if now - self._last >= self._min:
            self._last = now
            try:
                self._emit(lines[-1])
            except Exception:  # noqa: BLE001
                pass

    def flush(self):
        pass


class PipelineWorker(QObject):
    """Runs pipeline stages off the GUI thread."""
    log = Signal(str)
    stage_done = Signal(object)   # StageResult
    finished = Signal(list)       # list[StageResult]
    failed = Signal(str)

    def __init__(self, project, stages: Optional[list[str]] = None,
                 stop_on_error: bool = True):
        super().__init__()
        self.project = project
        self.stages = stages
        self.stop_on_error = stop_on_error
        self._cancel = False
        self._proc = None

    def cancel(self) -> None:
        """Terminate the running pipeline subprocess — this actually stops a
        long native stage mid-way (a between-stages flag never could)."""
        self._cancel = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()          # SIGTERM / TerminateProcess
            except Exception:  # noqa: BLE001
                pass

    def run(self) -> None:
        import subprocess
        import time
        from pathlib import Path
        import poseassess

        stages = list(self.stages or Pipeline.STAGES)
        pkg_parent = str(Path(poseassess.__file__).resolve().parent.parent)
        args = [sys.executable, "-u", "-m", "poseassess.core.pipeline_cli",
                str(self.project.root), ",".join(stages)]
        if not self.stop_on_error:
            args.append("--continue")
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"
        # no console window on Windows
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        results: list[StageResult] = []
        last_emit = 0.0
        try:
            self._proc = subprocess.Popen(
                args, cwd=pkg_parent, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=flags)
            for raw in self._proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                if line.startswith("@@STAGE_START "):
                    self.log.emit(f"=== {line[len('@@STAGE_START '):]} ===")
                elif line.startswith("@@STAGE_RESULT "):
                    rest = line[len("@@STAGE_RESULT "):]
                    name, status, msg = (rest.split(" ", 2) + ["", ""])[:3]
                    res = StageResult(name, status == "OK", msg)
                    results.append(res)
                    self.stage_done.emit(res)
                elif line.startswith("@@DONE"):
                    break
                else:
                    # throttle high-frequency progress (IK prints ~1 line/frame)
                    now = time.time()
                    if now - last_emit >= 0.15:
                        last_emit = now
                        self.log.emit(line)
            self._proc.wait()
            if self._cancel:
                self.log.emit("[cancelled]")
            self.finished.emit(results)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self._proc = None


def run_in_thread(worker: QObject) -> QThread:
    """Move `worker` (which must have a `run` slot) onto a new QThread and start.

    Returns the QThread; caller should keep a reference until finished.
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    # clean up when the worker signals completion
    if hasattr(worker, "finished"):
        worker.finished.connect(thread.quit)
    if hasattr(worker, "failed"):
        worker.failed.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
