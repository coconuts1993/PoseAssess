"""``BalancePlotCanvas``: multi-panel matplotlib canvas for the balance data (force, COP x/y, COM
x/y vs .trc time, event markers). Owner: GUI-VIEW.

Uses ``Figure`` + ``FigureCanvasQTAgg`` directly (never pyplot), like ``plot_canvas.PlotCanvas``.

Three drawings, one per canvas instance:
* ``plot_fused``      load (kg) + ML + AP (COP and COM, board frame, mm) vs .trc time, marked /
                      detected events, the analysis window shaded;
* ``plot_path``       sway path (AP vs ML) of the COP and the COM inside the window, with the COP
                      95 % confidence ellipse;
* ``plot_alignment``  force vs foot / COM height on the .trc time axis, for checking a time
                      alignment (sync events, manual offset).
Board frame: x = medio-lateral (ML, + = subject's right), y = antero-posterior (AP, + = front).
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

LOAD_COLOR = "#1a9a3a"
COP_COLOR = "#e0303a"
COM_COLOR = "#8a2be2"
EVENT_COLOR = "#3d8bfd"
DETECTED_COLOR = "#8a8f98"
WINDOW_COLOR = "#3d8bfd"
FOOT_COLOR = "#2f6fbf"
EVENT_STYLE = {  # detected event type -> (colour, short label)
    "takeoff": ("#2f6fbf", "takeoff"), "landing": ("#d1495b", "landing"),
    "stomp": ("#e08a1e", "stomp"), "step_on": ("#8a8f98", "step on"),
    "step_off": ("#8a8f98", "step off")}


def _finite_range(*arrays) -> tuple[float, float] | None:
    vals = [np.asarray(a, float)[np.isfinite(np.asarray(a, float))] for a in arrays if a is not None]
    vals = [v for v in vals if v.size]
    if not vals:
        return None
    lo = min(float(v.min()) for v in vals)
    hi = max(float(v.max()) for v in vals)
    return lo, hi


def ellipse95(x: np.ndarray, y: np.ndarray, n: int = 100) -> np.ndarray | None:
    """(n,2) outline of the 95 % confidence ellipse of the points (same units), or None."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return None
    x, y = x[ok], y[ok]
    cov = np.cov(np.vstack([x, y]))
    w, v = np.linalg.eigh(cov)
    w = np.sqrt(np.maximum(w, 0) * 5.991)  # chi2(0.95, 2)
    a = np.linspace(0, 2 * np.pi, n)
    circ = np.column_stack([np.cos(a) * w[0], np.sin(a) * w[1]])
    return circ @ v.T + np.array([x.mean(), y.mean()])


class BalancePlotCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 6, height: float = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.panels: dict[str, object] = {}
        self._auto_xlim: tuple[float, float] | None = None

    # ---------------------------------------------------------------- helpers
    def show_message(self, text: str) -> None:
        """Empty canvas with a centred message."""
        self.fig.clear()
        self.panels = {}
        self.fig.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=9,
                      color="#666666")
        self.draw_idle()

    def clear(self) -> None:
        self.fig.clear()
        self.panels = {}
        self._auto_xlim = None
        self.draw_idle()

    def reset_view(self) -> None:
        """Forget the user's zoom: the next ``plot_alignment`` shows its automatic view."""
        self._auto_xlim = None

    @staticmethod
    def _style(ax, ylabel: str, title: str = "") -> None:
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title, fontsize=8, loc="left")

    # ------------------------------------------------------------ plot_fused
    def plot_fused(self, fused, t_from: float | None = None, t_to: float | None = None, *,
                   force_events: list[dict] | None = None,
                   events: list[dict] | None = None) -> None:
        """Draw ``FusedTrial`` panels; shade [t_from, t_to].

        ``events`` (default ``fused.events``): marked events with ``trc_time`` and ``label``;
        ``force_events``: detected force events with ``trc_time`` and ``type`` (optional).
        The full-rate Wii columns (``fused.wii``) are used when present, else the per-frame
        values."""
        self.fig.clear()
        self.panels = {}
        if fused is None or not len(fused.trc_time):
            self.draw_idle()
            return
        tt = np.asarray(fused.trc_time, float)
        t_lo, t_hi = float(np.nanmin(tt)), float(np.nanmax(tt))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.15, 1, 1])
        ax_f = self.fig.add_subplot(gs[0])
        ax_ml = self.fig.add_subplot(gs[1], sharex=ax_f)
        ax_ap = self.fig.add_subplot(gs[2], sharex=ax_f)
        self.panels = {"load": ax_f, "ml": ax_ml, "ap": ax_ap}

        wii = fused.wii or {}
        wt = np.asarray(wii.get("trc_time", []), float)
        if len(wt) and np.isfinite(wt).any() and "total_kg" in wii:
            pad = 0.5
            m = np.isfinite(wt) & (wt >= t_lo - pad) & (wt <= t_hi + pad)
            order = np.argsort(wt[m], kind="stable")
            t_load = wt[m][order]
            load = np.asarray(wii["total_kg"], float)[m][order]
            cop_x = np.asarray(wii.get("cop_x_board", np.full(len(wt), np.nan)), float)[m][order]
            cop_y = np.asarray(wii.get("cop_y_board", np.full(len(wt), np.nan)), float)[m][order]
        else:
            t_load, load = tt, np.asarray(fused.total_kg, float)
            cop_x, cop_y = fused.cop_board[:, 0], fused.cop_board[:, 1]

        ax_f.plot(t_load, load, color=LOAD_COLOR, lw=1.0, label="load (Wii)")
        bm = fused.body_mass_kg
        if bm:
            ax_f.axhline(bm, color="#555555", ls=":", lw=1.0, label=f"body weight {bm:.1f} kg")
        self._style(ax_f, "load (kg)", "Vertical load")
        ax_ml.plot(t_load, cop_x * 1000, color=COP_COLOR, lw=0.9, label="COP")
        ax_ap.plot(t_load, cop_y * 1000, color=COP_COLOR, lw=0.9, label="COP")
        com = np.asarray(fused.com_board, float)
        if np.isfinite(com[:, :2]).any():
            ax_ml.plot(tt, com[:, 0] * 1000, color=COM_COLOR, lw=1.3, ls="--", label="COM")
            ax_ap.plot(tt, com[:, 1] * 1000, color=COM_COLOR, lw=1.3, ls="--", label="COM")
        self._style(ax_ml, "ML (mm)", "Medio-lateral (+ = subject's right)")
        self._style(ax_ap, "AP (mm)", "Antero-posterior (+ = front)")
        ax_ap.set_xlabel(".trc time (s)", fontsize=8)
        for ax in (ax_f, ax_ml):
            ax.tick_params(labelbottom=False)

        # events
        evs = fused.events if events is None else events
        for k, e in enumerate(evs or []):
            t = e.get("trc_time")
            if t is None or not np.isfinite(t):
                continue
            for ax in (ax_f, ax_ml, ax_ap):
                ax.axvline(t, color=EVENT_COLOR, lw=1.0, alpha=0.8,
                           label="marked event" if (k == 0 and ax is ax_f) else None)
            ax_f.annotate(str(e.get("label", "")), (t, 1.0), xycoords=("data", "axes fraction"),
                          xytext=(2, -2), textcoords="offset points", fontsize=6.5,
                          color=EVENT_COLOR, va="top", rotation=90)
        first = True
        for e in force_events or []:
            t = e.get("trc_time")
            if t is None or not np.isfinite(t):
                continue
            color = EVENT_STYLE.get(e.get("type"), (DETECTED_COLOR, ""))[0]
            for ax in (ax_f, ax_ml, ax_ap):
                ax.axvline(t, color=color, lw=0.8, ls="--", alpha=0.7,
                           label="detected events" if (first and ax is ax_f) else None)
            first = False

        # analysis window
        if t_from is not None and t_to is not None:
            a, b = sorted((float(t_from), float(t_to)))
            if a > t_lo + 1e-6 or b < t_hi - 1e-6:
                for ax in (ax_f, ax_ml, ax_ap):
                    ax.axvspan(a, b, color=WINDOW_COLOR, alpha=0.10, lw=0,
                               label="analysis window" if ax is ax_f else None)
        ax_f.set_xlim(t_lo, t_hi if t_hi > t_lo else t_lo + 1.0)
        for ax in (ax_f, ax_ml, ax_ap):
            h, _ = ax.get_legend_handles_labels()
            if h:
                ax.legend(loc="upper right", fontsize=6.5, framealpha=0.7)
        self.draw_idle()

    # ------------------------------------------------------------- plot_path
    def plot_path(self, fused, t_from: float | None = None, t_to: float | None = None) -> None:
        """Sway path (AP vs ML, board frame, mm) of the COP (full rate) and the COM inside the
        window, with the COP 95 % confidence ellipse."""
        self.fig.clear()
        self.panels = {}
        if fused is None or not len(fused.trc_time):
            self.draw_idle()
            return
        tt = np.asarray(fused.trc_time, float)
        lo = float(np.nanmin(tt)) if t_from is None else float(t_from)
        hi = float(np.nanmax(tt)) if t_to is None else float(t_to)
        lo, hi = min(lo, hi), max(lo, hi)
        wii = fused.wii or {}
        wt = np.asarray(wii.get("trc_time", []), float)
        if len(wt) and np.isfinite(wt).any() and "cop_x_board" in wii:
            m = np.isfinite(wt) & (wt >= lo - 1e-9) & (wt <= hi + 1e-9)
            order = np.argsort(wt[m], kind="stable")
            cx = np.asarray(wii["cop_x_board"], float)[m][order] * 1000
            cy = np.asarray(wii["cop_y_board"], float)[m][order] * 1000
        else:
            m = (tt >= lo - 1e-9) & (tt <= hi + 1e-9)
            cx, cy = fused.cop_board[m, 0] * 1000, fused.cop_board[m, 1] * 1000
        rows = (tt >= lo - 1e-9) & (tt <= hi + 1e-9)
        mx, my = fused.com_board[rows, 0] * 1000, fused.com_board[rows, 1] * 1000
        ax = self.fig.add_subplot(111)
        self.panels = {"path": ax}
        drawn = False
        if np.isfinite(cx).any():
            ax.plot(cx, cy, color=COP_COLOR, lw=0.8, alpha=0.9, label="COP")
            el = ellipse95(cx, cy)
            if el is not None:
                ax.plot(el[:, 0], el[:, 1], color=COP_COLOR, lw=1.0, ls=":",
                        label="COP 95 % ellipse")
            drawn = True
        if np.isfinite(mx).any():
            ax.plot(mx, my, color=COM_COLOR, lw=1.3, ls="--", label="COM (plumb point)")
            drawn = True
        if not drawn:
            ax.text(0.5, 0.5, "No COP / COM data in the window", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#666666")
        ax.set_aspect("equal", adjustable="datalim")
        self._style(ax, "AP (mm, + = front)",
                    f"Sway path, {lo:.2f}–{hi:.2f} s (board frame, seen from above)")
        ax.set_xlabel("ML (mm, + = subject's right)", fontsize=8)
        if drawn:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
        self.draw_idle()

    # -------------------------------------------------------- plot_alignment
    def plot_alignment(self, force_t: np.ndarray | None, force_kg: np.ndarray | None,
                       trc_t: np.ndarray | None = None, foot_up_m: np.ndarray | None = None,
                       com_up_m: np.ndarray | None = None, *,
                       force_events: list[dict] | None = None,
                       trc_events: list[dict] | None = None, title: str = "",
                       focus: tuple[float, float] | None = None) -> None:
        """Force (kg, left axis) and marker heights (cm relative to their median, right axis)
        on the .trc time axis. ``force_t`` must already be on the .trc time axis (Wii ``t_rel``
        mapped with the alignment to check). ``force_events`` carry ``trc_time`` (mapped the
        same way) and ``type``; ``trc_events`` carry ``t`` (.trc time) and ``type``.

        The time axis shows ``focus`` (e.g. around the detected events) or all the data; a
        view the user zoomed / panned with the toolbar is kept across re-plots (offset
        changes), and the toolbar's "home" returns to the automatic view."""
        keep = None
        old = self.panels.get("force")
        if old is not None and self._auto_xlim is not None:
            cur = tuple(float(v) for v in old.get_xlim())
            if not np.allclose(cur, self._auto_xlim, rtol=0, atol=1e-9):
                keep = cur
        self.fig.clear()
        self.panels = {}
        ax = self.fig.add_subplot(111)
        self.panels = {"force": ax}
        has_force = force_t is not None and force_kg is not None and \
            np.isfinite(np.asarray(force_t, float)).any()
        if has_force:
            ft, fk = np.asarray(force_t, float), np.asarray(force_kg, float)
            order = np.argsort(np.where(np.isfinite(ft), ft, np.inf), kind="stable")
            ax.plot(ft[order], fk[order], color=LOAD_COLOR, lw=0.9, label="Wii load")
        self._style(ax, "load (kg)", title)
        ax.set_xlabel(".trc time (s)", fontsize=8)
        ax2 = None
        if trc_t is not None and (foot_up_m is not None or com_up_m is not None):
            ax2 = ax.twinx()
            self.panels["height"] = ax2
            tt = np.asarray(trc_t, float)
            for sig, color, label, ls in ((foot_up_m, FOOT_COLOR, "lower foot height", "-"),
                                          (com_up_m, COM_COLOR, "COM height", "--")):
                if sig is None:
                    continue
                s = np.asarray(sig, float)
                if np.isfinite(s).any():
                    ax2.plot(tt, (s - np.nanmedian(s)) * 100, color=color, lw=1.1, ls=ls,
                             label=label)
            ax2.set_ylabel("height change (cm)", fontsize=8)
            ax2.tick_params(labelsize=7)
        for e in force_events or []:
            t = e.get("trc_time")
            if t is None or not np.isfinite(t):
                continue
            color, lab = EVENT_STYLE.get(e.get("type"), (DETECTED_COLOR, str(e.get("type"))))
            ax.axvline(t, color=color, lw=1.0, ls="--", alpha=0.85)
            ax.annotate(f"Wii {lab}", (t, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -2), textcoords="offset points", fontsize=6.5, color=color,
                        va="top", rotation=90)
        for e in trc_events or []:
            t = e.get("t")
            if t is None or not np.isfinite(t):
                continue
            color, lab = EVENT_STYLE.get(e.get("type"), (DETECTED_COLOR, str(e.get("type"))))
            ax.axvline(t, color=color, lw=2.2, alpha=0.25)
            ax.annotate(f"markers {lab}", (t, 0.0), xycoords=("data", "axes fraction"),
                        xytext=(2, 2), textcoords="offset points", fontsize=6.5, color=color,
                        va="bottom", rotation=90)
        rng = _finite_range(force_t if has_force else None, trc_t)
        if focus is not None and rng is not None:
            lo, hi = max(focus[0], rng[0]), min(focus[1], rng[1])
            rng = (lo, hi) if hi > lo else rng
        if rng is not None and rng[1] > rng[0]:
            ax.set_xlim(*rng)
        self._auto_xlim = tuple(float(v) for v in ax.get_xlim())
        if self.toolbar is not None:
            try:  # new axes: reset the zoom history, "home" = the automatic view
                self.toolbar.update()
                self.toolbar.push_current()
            except Exception:  # noqa: BLE001
                pass
        if keep is not None:
            ax.set_xlim(*keep)
        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=6.5, framealpha=0.7)
        if not has_force and not (trc_t is not None and len(trc_t)):
            ax.text(0.5, 0.5, "Nothing to show", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#666666")
        self.draw_idle()
