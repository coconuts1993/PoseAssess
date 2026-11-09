"""Headless benchmark of the live 3D skeleton: measures FPS and latency on this PC.

Examples (portable bundle: ``run_live.bat`` passes its arguments here)::

    run_live.bat --project "D:\\Projects\\Trial01"              # cameras + Calib.toml of the project
    run_live.bat --project "D:\\Projects\\Trial01" --size 1280x720 --show
    run_live.bat --sources 0 1 --size 1280x720 --no-3d         # 2D only, no calibration needed
    run_live.bat --sources a.mp4 b.mp4 --calib Calib.toml      # replay videos (loop)

Prints a statistics line every second and a summary at the end; ``--csv`` writes every result
(timings, angles, 3D points).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("poseassess.realtime")


def _size(text: str | None):
    if not text:
        return None, None
    w, h = text.lower().replace("*", "x").split("x")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m poseassess.core.realtime",
                                description="Live multi-webcam 3D skeleton benchmark")
    p.add_argument("--project", help="PoseAssess project: cameras from 3b. Capture settings, "
                                     "calibration from calibration/Calib.toml")
    p.add_argument("--sources", nargs="+", help="camera indices / video files / URLs "
                                                "(override the project's cameras)")
    p.add_argument("--calib", help="Calib.toml (default: the project's)")
    p.add_argument("--no-3d", action="store_true", help="2D only (no calibration needed)")
    p.add_argument("--size", help="request this capture size, e.g. 1280x720")
    p.add_argument("--fps", type=float, help="request this camera frame rate")
    p.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"],
                   help="RTMPose model size (default balanced = RTMPose-m + YOLOX-m)")
    p.add_argument("--device", default="auto", help="auto | cuda | dml | cpu (default auto)")
    p.add_argument("--det-every", type=int, default=3,
                   help="run the person detector every N frames (default 3)")
    p.add_argument("--seconds", type=float, default=30.0, help="run time (default 30 s; 0 = until Ctrl+C / q)")
    p.add_argument("--csv", help="write every result to this CSV")
    p.add_argument("--show", action="store_true", help="show the cameras with the 2D skeleton")
    p.add_argument("--no-smooth", action="store_true", help="disable the One-Euro filter")
    return p


def _open_sources(args):
    """Returns ({cam: CameraStream}, [calib index per cam] or None)."""
    from poseassess.core.capture.camera import CameraStream

    w, h = _size(args.size)
    if args.sources:
        streams = {f"cam{i:02d}": CameraStream(f"cam{i:02d}", s, w, h, args.fps)
                   for i, s in enumerate(args.sources, start=1)}
        return streams
    if not args.project:
        raise SystemExit("give --project or --sources")
    from poseassess.core.capture.settings import load_capture_settings
    from poseassess.core.project import Project

    proj = Project.load(args.project)
    st = load_capture_settings(proj)
    streams = {}
    for cam, cs in sorted(st.cameras.items()):
        if not cs.enabled:
            continue
        streams[cam] = CameraStream(cam, cs.source, w or cs.width, h or cs.height,
                                    args.fps or cs.fps)
    if not streams:
        raise SystemExit("no enabled camera in the project's capture settings")
    return streams


def _calib(args, cams: list[str]):
    if args.no_3d:
        return None
    from .calib import load_calib

    path = args.calib
    if not path:
        if not args.project:
            raise SystemExit("3D needs --calib or --project (or use --no-3d)")
        from poseassess.core.project import Project
        path = Project.load(args.project).calib_toml
    allc = load_calib(path)
    sel = []
    for c in cams:
        i = int(c[3:]) - 1
        if i >= len(allc):
            raise SystemExit(f"{c}: {path} has only {len(allc)} cameras")
        sel.append(allc[i])
    print(f"calibration: {path} -> " + ", ".join(f"{c}={k.name} {k.size[0]}x{k.size[1]}"
                                                for c, k in zip(cams, sel)))
    return sel


def _fmt(s: dict) -> str:
    def f(v, nd=0):
        return "  -" if v != v else f"{v:.{nd}f}"
    return (f"{f(s['fps'], 1)} fps | latency {f(s['latency_ms'])} ms (p95 {f(s['latency_p95_ms'])})"
            f" | 2D {f(s['pose2d_ms'])} ms | 3D {f(s['triangulate_ms'], 1)} ms"
            f" | skew {f(s['skew_ms'])} ms | joints {f(s['joints_3d'])}/26")


def _draw(img, kpts, scores, thr=0.3):
    import cv2
    from .skeleton import BONE_IDX

    col = {"R": (255, 150, 60), "L": (40, 140, 255), "C": (200, 200, 200)}
    for a, b, s in BONE_IDX:
        if scores[a] >= thr and scores[b] >= thr:
            cv2.line(img, tuple(int(v) for v in kpts[a]), tuple(int(v) for v in kpts[b]),
                     col[s], 3, cv2.LINE_AA)
    for (x, y), sc in zip(kpts, scores):
        if sc >= thr:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1, cv2.LINE_AA)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    from poseassess.core.capture.session import open_streams
    from .engine import EngineConfig, RealtimeEngine
    from .pose2d import PoseEstimator, available_devices, resolve_device

    print("onnxruntime devices:", ", ".join(available_devices()))
    device = resolve_device(args.device)
    streams = _open_sources(args)
    cams = list(streams)
    calib = _calib(args, cams)

    print(f"loading RTMPose ({args.mode}) on {device} for {len(cams)} cameras ...")
    ests = [PoseEstimator(args.mode, device, args.det_every) for _ in cams]
    active = getattr(ests[0], "active_device", device)
    if active != device:
        print(f"[WARNING] onnxruntime fell back from {device} to {active} (see README_LIVE.md)")
        device = active

    errs = {c: e for c, e in open_streams(streams).items() if e}
    if errs:
        for s in streams.values():
            s.stop()
        for c, e in errs.items():
            print(f"[ERROR] {c}: {e}")
        return 2
    try:
        t_wait = time.perf_counter()
        while any(s.latest() is None for s in streams.values()):
            if time.perf_counter() - t_wait > 5:
                raise SystemExit("no frames from: " + ", ".join(c for c, s in streams.items()
                                                                if s.latest() is None))
            time.sleep(0.01)
        for c, s in streams.items():
            h, w = s.latest().image.shape[:2]
            print(f"{c}: source {s.source!r}  {w}x{h}  nominal {s.fps:.0f} fps")
        for c, e in zip(cams, ests):
            sz = streams[c].latest().image.shape[1::-1]
            print(f"warm-up {c}: {e.warmup(tuple(sz)):.1f} s")

        eng = RealtimeEngine(streams, ests, calib,
                             EngineConfig(smooth=not args.no_smooth), csv_path=args.csv)
        eng.start()
        t_start = time.perf_counter()
        t_print = t_start + 1.0
        all_lat = []
        last_seq = -1
        while True:
            now = time.perf_counter()
            if args.seconds and now - t_start >= args.seconds:
                break
            if eng.error:
                print("[ERROR]", eng.error)
                break
            r = eng.latest()
            if r is not None and r.seq != last_seq:
                last_seq = r.seq
                if now - t_start > 2.0:  # skip the start-up seconds in the summary
                    all_lat.append((r.t_done, r.latency_ms, r.timings_ms["pose2d"], r.skew_ms))
            if now >= t_print:
                t_print += 1.0
                cam_fps = " ".join(f"{c}:{s.measured_fps:.0f}" for c, s in streams.items())
                line = _fmt(eng.stats.summary()) + f" | cams {cam_fps}"
                if r is not None and r.angles:
                    line += " | knee R/L {:.0f}/{:.0f}".format(r.angles["RKnee"], r.angles["LKnee"])
                print(line, flush=True)
            if args.show and r is not None:
                import cv2
                tiles = []
                for i, c in enumerate(cams):
                    fr = streams[c].latest()
                    img = fr.image.copy()
                    _draw(img, r.kpts2d[i], r.scores[i])
                    th = 360
                    tiles.append(cv2.resize(img, (int(img.shape[1] * th / img.shape[0]), th)))
                view = np.hstack(tiles)
                cv2.putText(view, _fmt(eng.stats.summary()), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("PoseAssess live (q = quit)", view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            else:
                time.sleep(0.005)
        eng.stop()
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams.values():
            s.stop()
        if args.show:
            import cv2
            cv2.destroyAllWindows()

    if len(all_lat) >= 2:
        a = np.array(all_lat)
        span = a[-1, 0] - a[0, 0]
        print("\n=== summary (after the first 2 s) ===")
        print(f"cameras: {len(cams)}   model: {args.mode}   device: {device}   "
              f"detector every {args.det_every} frames   3D: {'yes' if calib else 'no'}")
        print(f"results: {len(a)}   rate: {(len(a) - 1) / span:.1f} fps")
        print(f"latency (frame grabbed -> 3D ready): mean {a[:, 1].mean():.0f} ms, "
              f"median {np.median(a[:, 1]):.0f} ms, p95 {np.percentile(a[:, 1], 95):.0f} ms")
        print(f"2D pose (all cameras in parallel): mean {a[:, 2].mean():.0f} ms")
        print(f"camera skew: mean {a[:, 3].mean():.0f} ms")
        print("(+ camera exposure/USB latency, typically 30-100 ms, and the display)")
        if args.csv:
            print("per-frame results:", args.csv)
    else:
        print("not enough results for a summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
