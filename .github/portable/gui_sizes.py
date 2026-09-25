"""Minimum width of the main window and of every page (printed by CI on Windows).

``--max N``: exit with 1 when the main window's minimum width exceeds N px (CI checks the native
Windows platform; Qt's offscreen platform has no real fonts on Windows)."""
import argparse
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.getcwd())
ap = argparse.ArgumentParser()
ap.add_argument("scale", nargs="?", type=float, default=1.0, help="font size factor")
ap.add_argument("--max", type=int, default=None, help="fail above this minimum width (px)")
opts = ap.parse_args()
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

app = QApplication([])
f = app.font()
f.setPointSizeF(f.pointSizeF() * opts.scale)
app.setFont(f)
max_w = opts.max
from poseassess.gui.main_window import MainWindow
w=MainWindow(); w.show(); app.processEvents()
print("platform", app.platformName(), "dpi", app.primaryScreen().logicalDotsPerInch(), app.primaryScreen().devicePixelRatio(), "font", app.font().family(), app.font().pointSizeF(), "MainWindow min", w.minimumSizeHint().width())
for p in w.pages:
    print(f"  {p.nav_title:16s} {p.minimumSizeHint().width():5d}")
# widest descendants of the widest page
wp=max(w.pages,key=lambda p:p.minimumSizeHint().width())
items=sorted(((c.minimumSizeHint().width(), type(c).__name__, (c.text()[:50] if hasattr(c,'text') and callable(c.text) else c.objectName())) for c in wp.findChildren(QWidget)), reverse=True)[:12]
print("widest page:", wp.nav_title)
for it in items: print("   ", it)
w.close()
if max_w is not None and w.minimumSizeHint().width() > max_w:
    print(f"FAIL: main window minimum width {w.minimumSizeHint().width()} px > {max_w} px")
    sys.exit(1)
