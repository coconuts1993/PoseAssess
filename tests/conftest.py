"""Shared pytest setup for PoseAssess.

* puts the repository root on sys.path (``import poseassess``, ``from tests.synth import ...``);
* headless Qt (``QT_QPA_PLATFORM=offscreen``; OpenGL rendering needs xvfb, see ``gl`` marker);
* ``POSEASSESS_WII=0``: the app never starts the Wii auto-connect thread in tests unless a test
  asks for it;
* fixtures: ``qapp``, ``project`` (empty 3-camera project), ``demo_trial`` (``synth.make_demo_trial``).

Tests that need real OpenGL rendering are marked ``@pytest.mark.gl`` and run only with
``POSEASSESS_TEST_GL=1`` under ``xvfb-run`` (QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("POSEASSESS_WII", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "gl: needs OpenGL rendering (xvfb); POSEASSESS_TEST_GL=1")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("POSEASSESS_TEST_GL") == "1":
        return
    skip = pytest.mark.skip(reason="OpenGL test: set POSEASSESS_TEST_GL=1 and run under xvfb")
    for item in items:
        if "gl" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def project(tmp_path):
    """An empty, created 3-camera / 30 fps project (no Config.toml)."""
    from poseassess.core.project import Project

    p = Project(tmp_path / "proj")
    p.config.name = "test"
    p.config.num_cameras = 3
    p.config.frame_rate = 30
    return p.create()


@pytest.fixture
def demo_trial(tmp_path):
    """``synth.make_demo_trial`` in a temporary folder (Z-up world, 3 cameras)."""
    from tests.synth import make_demo_trial

    return make_demo_trial(tmp_path / "demo")


@pytest.fixture(autouse=True)
def _fresh_pyqtgraph_gl_programs(monkeypatch):
    """With real OpenGL (xvfb runs), pyqtgraph's shader programs are compiled once per class /
    globally, i.e. for the GL context of the first test that drew: later tests (new contexts)
    would log GLErrors. Force recompiling them per test. No-op for offscreen runs (no GL)."""
    if os.environ.get("QT_QPA_PLATFORM", "offscreen") == "offscreen":
        yield
        return
    try:
        from pyqtgraph.opengl import shaders
        from pyqtgraph.opengl.items.GLLinePlotItem import GLLinePlotItem
        from pyqtgraph.opengl.items.GLScatterPlotItem import GLScatterPlotItem
    except Exception:  # noqa: BLE001  (no pyqtgraph / PyOpenGL)
        yield
        return
    monkeypatch.setattr(GLLinePlotItem, "_shaderProgram", None, raising=False)
    monkeypatch.setattr(GLScatterPlotItem, "_shaderProgram", None, raising=False)
    for prog in getattr(shaders, "Shaders", []):
        monkeypatch.setattr(prog, "prog", None, raising=False)
    yield
