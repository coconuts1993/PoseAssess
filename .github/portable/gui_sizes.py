"""Diagnostic: minimum width of the main window and of every page (CI prints it on Windows)."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.getcwd())
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QFont
app=QApplication([])
scale=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
f=app.font(); f.setPointSizeF(f.pointSizeF()*scale); app.setFont(f)
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
