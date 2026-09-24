"""Embedded matplotlib canvas for joint-angle curves."""
from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")  # bind to whatever Qt binding is active (PySide6)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def plot_bilateral(self, motion, base: str, label: str) -> None:
        """Plot the right and left curve of a bilateral joint over time."""
        self.ax.clear()
        df = motion.df
        t = df.index.to_numpy()
        plotted = False
        for suffix, color, name in (("_r", "#d1495b", "Right"), ("_l", "#3d8bfd", "Left")):
            col = base + suffix
            if col in df.columns:
                self.ax.plot(t, df[col].to_numpy(), color=color, label=name, linewidth=1.6)
                plotted = True
        if not plotted and base in df.columns:  # unilateral coordinate
            self.ax.plot(t, df[base].to_numpy(), color="#2a9d8f", label=label, linewidth=1.6)
        self.ax.set_title(label)
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("angle (deg)" if motion.in_degrees else "angle (rad)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="best", fontsize=8)
        self.draw()

    def clear(self) -> None:
        self.ax.clear()
        self.draw()
