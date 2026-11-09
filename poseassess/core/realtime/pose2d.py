"""Live 2D keypoints with rtmlib (RTMPose ``Body_with_feet`` -> HALPE_26).

One ``PoseEstimator`` per camera (each keeps its own detector cache when ``det_every`` > 1).
The models are those of the offline pipeline (``%USERPROFILE%\\.cache\\rtmlib``; the portable
bundle ships RTMPose-s/m + YOLOX-tiny/m, so ``lightweight`` and ``balanced`` work offline).

Devices: ``cuda`` needs the ``onnxruntime-gpu`` package (+ CUDA 12 / cuDNN 9 runtime DLLs);
``dml`` needs ``onnxruntime-directml`` (any DirectX 12 GPU on Windows, nothing else to
install); ``cpu`` always works. ``auto`` takes the first available of cuda, dml, cpu.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)

N_KPTS = 26

_PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def available_devices() -> list[str]:
    """Devices onnxruntime can run on here, best first (always ends with ``cpu``)."""
    try:
        import onnxruntime as ort
        prov = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return ["cpu"]
    out = [d for d in ("cuda", "dml") if _PROVIDERS[d] in prov]
    return out + ["cpu"]


def resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    avail = available_devices()
    if device == "auto":
        return avail[0]
    if device not in avail:
        raise RuntimeError(
            f"onnxruntime here cannot use '{device}' (available: {', '.join(avail)}). "
            + ("Install onnxruntime-gpu (and the CUDA 12 / cuDNN 9 runtime) for an NVIDIA GPU, "
               "or onnxruntime-directml for any GPU on Windows." if device != "cpu" else ""))
    return device


def _provider_list(device: str) -> list:
    if device == "cuda":
        return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider"]
    if device == "dml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _rebind_session(tool, device: str) -> None:
    """Re-create the onnxruntime session of an rtmlib model with ``device``'s providers.

    rtmlib only knows ``cpu`` / ``cuda`` / ``mps``: models are built for ``cpu`` and their
    sessions swapped here, which also lets ``dml`` work."""
    import onnxruntime as ort

    path = getattr(tool, "onnx_model", None)
    if tool is None or getattr(tool, "session", None) is None or not path:
        return
    if device == "cuda" and hasattr(ort, "preload_dlls"):
        try:  # CUDA / cuDNN DLLs from the nvidia-* pip packages, if installed (ORT >= 1.21)
            ort.preload_dlls()
        except Exception:  # noqa: BLE001
            log.debug("onnxruntime.preload_dlls failed", exc_info=True)
    tool.session = ort.InferenceSession(path_or_bytes=path, providers=_provider_list(device))
    got = tool.session.get_providers()
    if _PROVIDERS[device] not in got:
        log.warning("%s: %s not used (providers %s)", type(tool).__name__, _PROVIDERS[device], got)


class PoseEstimator:
    """Single-person HALPE_26 keypoints for one camera.

    ``det_every``: run the person detector only every N frames and track the box in between
    (rtmlib ``PoseTracker``); the detector is the slowest stage, so 3-5 roughly doubles the
    rate."""

    def __init__(self, mode: str = "balanced", device: str = "auto", det_every: int = 3):
        from rtmlib import BodyWithFeet, PoseTracker

        self.device = resolve_device(device)
        self.mode = mode
        self.det_every = max(1, int(det_every))
        self._tracker = PoseTracker(BodyWithFeet, det_frequency=self.det_every, tracking=False,
                                    mode=mode, to_openpose=False, backend="onnxruntime",
                                    device="cpu")
        if self.device != "cpu":
            _rebind_session(self._tracker.pose_model, self.device)
            _rebind_session(getattr(self._tracker, "det_model", None), self.device)
        self._prev_center: np.ndarray | None = None

    @property
    def active_device(self) -> str:
        """Device the pose model really runs on (onnxruntime falls back to the CPU when e.g.
        the CUDA DLLs are missing)."""
        sess = getattr(self._tracker.pose_model, "session", None)
        try:
            got = sess.get_providers()
        except Exception:  # noqa: BLE001
            return self.device
        for d in ("cuda", "dml"):
            if _PROVIDERS[d] in got:
                return d
        return "cpu"

    def warmup(self, size=(1280, 720)) -> float:
        """Run once on a blank image (first CUDA/DML run allocates and tunes); returns s."""
        t = time.perf_counter()
        self(np.zeros((size[1], size[0], 3), np.uint8))
        return time.perf_counter() - t

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BGR image -> (26, 2) pixels, (26,) scores (all zero when nobody is seen)."""
        kpts, scores = self._tracker(image)
        kpts = np.asarray(kpts, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        if kpts.ndim != 3 or len(kpts) == 0:
            self._prev_center = None
            return np.full((N_KPTS, 2), np.nan), np.zeros(N_KPTS)
        i = self._pick(kpts, scores)
        k, s = kpts[i, :N_KPTS], scores[i, :N_KPTS]
        self._prev_center = np.nanmean(k, axis=0)
        return k, s

    def _pick(self, kpts: np.ndarray, scores: np.ndarray) -> int:
        """The subject: nearest to the previous one, else the most confident and largest."""
        if len(kpts) == 1:
            return 0
        if self._prev_center is not None:
            d = np.linalg.norm(np.nanmean(kpts, axis=1) - self._prev_center, axis=1)
            return int(np.nanargmin(d))
        ext = np.nanmax(kpts, axis=1) - np.nanmin(kpts, axis=1)
        area = ext[:, 0] * ext[:, 1]
        return int(np.nanargmax(scores.mean(axis=1) * np.sqrt(np.maximum(area, 1.0))))
