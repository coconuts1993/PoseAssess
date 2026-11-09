"""Find, check and set up a LOCAL OpenPose install for the BODY_25B model.

OpenPose is CMU software under an academic / non-profit, non-commercial licence
that forbids redistribution, so PoseAssess never bundles it. Instead this module

* finds the user's own install (project setting, the folder remembered on this
  computer, or a few usual places such as ``C:\\openpose``) and checks that it
  can run BODY_25B (`check_install`, `resolve_openpose_dir`);
* downloads the OFFICIAL Windows GPU release from CMU's GitHub onto the user's
  PC, after the user accepted the licence (`download_openpose`), never into the
  PoseAssess folder (that folder is copied to other PCs);
* installs the BODY_25B files: the official ``pose_deploy.prototxt`` of one of
  the two variants (`download_prototxt`) and the weights the user supplies
  (`install_caffemodel`; there is no official download of them any more);
* checks the NVIDIA GPU / driver against what OpenPose 1.7.0 supports
  (`gpu_info`, `gpu_warnings`).

BODY_25B layout inside the install root (OpenPose looks for these EXACT names)::

    bin/OpenPoseDemo.exe
    bin/cudart64_110.dll                      (GPU build only)
    models/pose/body_25b/pose_deploy.prototxt
    models/pose/body_25b/pose_iter_XXXXXX.caffemodel

Run it with the install root as working directory (or ``--model_folder
<root>/models/``) and ``--model_pose BODY_25B --net_resolution -1x480``.

Qt-free: the GUI runs the slow functions in a worker thread.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import logging
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Official release and BODY_25B model files
# --------------------------------------------------------------------------- #
OFFICIAL_VERSION = "1.7.0"
OFFICIAL_ZIP_NAME = "openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip"
OFFICIAL_ZIP_URL = ("https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/"
                    f"download/v{OFFICIAL_VERSION}/{OFFICIAL_ZIP_NAME}")
OFFICIAL_ZIP_SIZE = 437_912_637
OFFICIAL_ZIP_SHA256 = "82ecf470c81b001d1b34873081c0157821c75a0557d926c7427f688d5cf0b25d"
#: Sum of the file sizes in the official archive (453 entries).
OFFICIAL_UNPACKED_SIZE = 994_329_639
LICENSE_URL = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE"

NET_RESOLUTION = "-1x480"
MODEL_POSE = "BODY_25B"
MODEL_SUBDIR = Path("models", "pose", "body_25b")
CAFFEMODEL_NAME = "pose_iter_XXXXXX.caffemodel"
PROTOTXT_NAME = "pose_deploy.prototxt"
EXE_NAME = "OpenPoseDemo.exe"

#: Oldest NVIDIA driver that runs the CUDA 11 build of OpenPose 1.7.0.
MIN_DRIVER = (456, 38)
#: Compute capabilities OpenPose 1.7.0 runs on: [MIN_COMPUTE_CAP, MAX_COMPUTE_CAP).
MIN_COMPUTE_CAP = (5, 2)
MAX_COMPUTE_CAP = (10, 0)

_TIMEOUT_S = 60
_CHUNK = 1 << 20          # unpacking
_NET_CHUNK = 64 << 10     # downloading: small, so progress and cancel stay responsive
_DOWNLOAD_DIR = "openpose.download"   # holds the .part file next to <dest>/openpose

_TRAIN_RAW = ("https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
              "openpose_train/master/experimental_models")


@dataclass(frozen=True)
class Body25bVariant:
    """One of the two official BODY_25B networks (openpose_train experimental models)."""
    name: str               # openpose_train folder name, e.g. "1_25BBkg"
    title: str              # for a combo box
    prototxt_url: str
    prototxt_sha256: str
    short: str = ""         # for status lines, e.g. "Option 2"


BODY25B_VARIANTS: dict[str, Body25bVariant] = {
    "1_25BBkg": Body25bVariant(
        "1_25BBkg", "Option 2: recommended by CMU (1_25BBkg)",
        f"{_TRAIN_RAW}/1_25BBkg/body_25b/{PROTOTXT_NAME}",
        "6b05f4dc5139b70bb7b48ad90534671edef41f03f387bfd82bf6db1fa9ddb405",
        "Option 2"),
    "1_25BSuperModel11FullVGG": Body25bVariant(
        "1_25BSuperModel11FullVGG",
        "Option 1: maximum accuracy, slower (1_25BSuperModel11FullVGG)",
        f"{_TRAIN_RAW}/1_25BSuperModel11FullVGG/body_25b/{PROTOTXT_NAME}",
        "9fe1dc100e6e88be8d2773a15a6bdeef2001ce6e60d3d8b6f3502f692c07ed24",
        "Option 1"),
}
DEFAULT_VARIANT = "1_25BBkg"


# --------------------------------------------------------------------------- #
# Install status
# --------------------------------------------------------------------------- #
@dataclass
class InstallStatus:
    """Result of `check_install`: can this folder run OpenPose BODY_25B?"""
    root: Path | None
    exe: Path | None
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    variant: str | None = None   # BODY25B_VARIANTS key from the prototxt hash

    def summary(self) -> str:
        """One short line for a status label."""
        if not self.ok:
            first = self.errors[0] if self.errors else "unknown problem"
            return f"OpenPose not ready: {first}"
        v = BODY25B_VARIANTS.get(self.variant or "")
        which = (v.short or v.name) if v else "custom prototxt"
        return f"OpenPose ready: {self.root} ({MODEL_POSE}, {which})"


# --------------------------------------------------------------------------- #
# Locating and checking an install
# --------------------------------------------------------------------------- #
def _is_windows() -> bool:
    return os.name == "nt"


def _is_file(p: Path) -> bool:
    try:
        return p.is_file()
    except OSError:  # e.g. a drive that is not ready
        return False


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except OSError:
        return False


def _file_size(p: Path) -> int:
    """Size of a regular file, 0 when it is missing / not a file / unreadable."""
    try:
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def _clean_path(path: str | Path | None) -> Path | None:
    """Path from user input; strips blanks and the quotes of Explorer's "Copy as path"."""
    if path is None:
        return None
    s = str(path).strip().strip('"').strip()
    return Path(s).expanduser() if s else None


def demo_exe(root: str | Path) -> Path | None:
    """``OpenPoseDemo.exe`` of an install root (in ``bin/`` or the root itself)."""
    root = Path(root)
    for cand in (root / "bin" / EXE_NAME, root / EXE_NAME):
        if _is_file(cand):
            return cand
    return None


def _root_of_exe(exe: Path) -> Path:
    return exe.parent.parent if exe.parent.name.lower() == "bin" else exe.parent


def model_dir(root: str | Path) -> Path:
    """``<root>/models/pose/body_25b``."""
    return Path(root) / MODEL_SUBDIR


def find_root(path: str | Path | None) -> Path | None:
    """Install root (the folder with ``bin/`` and ``models/``) for a user-given path.

    Accepts the root itself, its ``bin`` folder, the exe, or a parent folder that
    contains an ``openpose`` folder (e.g. the folder the release zip was unpacked
    into). None when no ``OpenPoseDemo.exe`` is found there.
    """
    p = _clean_path(path)
    if p is None:
        return None
    if p.name.lower() == EXE_NAME.lower() and _is_file(p):
        return _root_of_exe(p)
    if not _is_dir(p):
        return None
    exe = demo_exe(p)
    if exe is not None:
        return _root_of_exe(exe)
    try:
        children = sorted(c for c in p.iterdir() if c.name.lower() == "openpose")
    except OSError:
        return None
    for child in children:
        exe = demo_exe(child)
        if exe is not None:
            return _root_of_exe(exe)
    return None


def prototxt_variant(path: str | Path) -> str | None:
    """BODY25B_VARIANTS key of a prototxt file (CRLF line endings tolerated), else None."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    by_sha = {v.prototxt_sha256: k for k, v in BODY25B_VARIANTS.items()}
    for blob in (data, data.replace(b"\r\n", b"\n")):
        key = by_sha.get(hashlib.sha256(blob).hexdigest())
        if key:
            return key
    return None


def check_install(path: str | Path | None) -> InstallStatus:
    """Check that `path` (see `find_root`) holds an OpenPose GPU build with BODY_25B."""
    given = _clean_path(path)
    if given is None:
        return InstallStatus(None, None, False, ["no OpenPose folder set"])
    root = find_root(given)
    if root is None:
        return InstallStatus(None, None, False, [f"{EXE_NAME} not found in {given}"])
    exe = demo_exe(root)
    errors: list[str] = []
    warnings: list[str] = []

    try:
        has_cudart = any(exe.parent.glob("cudart64_*.dll"))
    except OSError:
        has_cudart = False
    if not has_cudart:
        errors.append(f"CPU-only build of OpenPose (no cudart64_*.dll next to {EXE_NAME}); "
                      f"{MODEL_POSE} needs the GPU build")

    mdir = model_dir(root)
    variant = None
    prototxt = mdir / PROTOTXT_NAME
    if not _is_file(prototxt):
        errors.append(f"missing {MODEL_SUBDIR / PROTOTXT_NAME}")
    else:
        variant = prototxt_variant(prototxt)
        if variant is None:
            warnings.append(f"{MODEL_SUBDIR / PROTOTXT_NAME} is not one of the two official "
                            f"{MODEL_POSE} prototxt files; it must match the weights")

    caffemodel = mdir / CAFFEMODEL_NAME
    if _is_file(caffemodel):
        if _file_size(caffemodel) == 0:
            errors.append(f"{MODEL_SUBDIR / CAFFEMODEL_NAME} is empty")
    else:
        others = sorted(mdir.glob("pose_iter_*.caffemodel")) if _is_dir(mdir) else []
        if others:
            errors.append(f"{MODEL_POSE} weights must be named {CAFFEMODEL_NAME}: rename "
                          f"{MODEL_SUBDIR / others[0].name} to {CAFFEMODEL_NAME}")
        else:
            errors.append(f"missing {MODEL_POSE} weights {MODEL_SUBDIR / CAFFEMODEL_NAME}")

    if not str(root.absolute()).isascii():
        warnings.append(f"the OpenPose folder path has non-ASCII characters ({root}); "
                        "OpenPose may fail to load its models - move it to e.g. C:\\openpose")

    return InstallStatus(root, exe, not errors, errors, warnings, variant)


# --------------------------------------------------------------------------- #
# Per-computer memory of the install folder
# --------------------------------------------------------------------------- #
def user_settings_path() -> Path:
    """``%APPDATA%\\PoseAssess\\openpose.json`` (Windows) or
    ``~/.config/poseassess/openpose.json``; ``POSEASSESS_CONFIG_DIR`` overrides the folder."""
    env = os.environ.get("POSEASSESS_CONFIG_DIR", "").strip()
    if env:
        base = Path(env)
    elif _is_windows():
        appdata = os.environ.get("APPDATA")
        base = (Path(appdata) if appdata else Path.home() / "AppData" / "Roaming") / "PoseAssess"
    else:
        base = Path.home() / ".config" / "poseassess"
    return base / "openpose.json"


def _read_settings() -> dict:
    try:
        d = json.loads(user_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def remember_dir(path: str | Path) -> None:
    """Remember the OpenPose folder on this computer (for every project)."""
    d = _read_settings()
    d["openpose_dir"] = str(path)
    dest = user_settings_path()
    try:
        _atomic_write(dest, json.dumps(d, indent=2).encode("utf-8"))
    except OSError as e:
        log.warning("could not remember the OpenPose folder in %s: %s", dest, e)


def remembered_dir() -> Path | None:
    """The folder saved by `remember_dir`, or None."""
    v = _read_settings().get("openpose_dir")
    return Path(v) if isinstance(v, str) and v.strip() else None


def bundle_root() -> Path:
    """Folder that holds ``run.bat`` (the parent of the ``poseassess`` package)."""
    return Path(__file__).resolve().parents[2]


def candidate_dirs() -> list[Path]:
    """Usual install places, in search order."""
    out = [bundle_root() / "openpose"]
    if _is_windows():
        out += [Path("C:\\openpose"), Path("D:\\openpose")]
        profile = os.environ.get("USERPROFILE")
        out.append((Path(profile) if profile else Path.home()) / "openpose")
    else:
        out.append(Path.home() / "openpose")
    seen: set[str] = set()
    unique = []
    for p in out:
        key = str(p).lower() if _is_windows() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def resolve_openpose_dir(project_setting: str | Path | None) -> tuple[Path | None, str]:
    """(install root, where it came from) for running OpenPose.

    Looks at the project setting, then the folder remembered on this computer,
    then `candidate_dirs`; the first complete install wins, else the first one
    that at least has ``OpenPoseDemo.exe`` (`check_install` tells what is
    missing), else ``(None, reason)``.
    """
    sources: list[tuple[Path, str]] = []
    setting = _clean_path(project_setting)
    if setting is not None:
        sources.append((setting, "project setting"))
    remembered = remembered_dir()
    if remembered is not None:
        sources.append((remembered, "remembered on this computer"))
    sources += [(c, "") for c in candidate_dirs()]

    partial: tuple[Path | None, str] | None = None
    for path, origin in sources:
        st = check_install(path)
        if st.root is None:
            continue
        found = (st.root, origin or f"found at {st.root}")
        if st.ok:
            return found
        if partial is None:
            partial = found
    if partial is not None:
        return partial
    looked = ", ".join(str(p) for p, _ in sources)
    return None, f"{EXE_NAME} not found (looked in: {looked})"


# --------------------------------------------------------------------------- #
# Downloads and model installation
# --------------------------------------------------------------------------- #
# All failures raise RuntimeError with a short user-facing message, except an
# unknown variant name or bad download parameters (ValueError: a programming error).
Opener = Callable[..., object]


def _unlink_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        part.write_bytes(data)
        os.replace(part, dest)
    finally:
        _unlink_quietly(part)


def _write_file(dest: Path, data: bytes) -> None:
    """`_atomic_write`; RuntimeError for disk problems (read-only folder, ...)."""
    try:
        _atomic_write(dest, data)
    except OSError as e:
        raise RuntimeError(f"cannot write {dest}: {e}") from e


def _fetch(url: str, opener: Opener) -> bytes:
    try:
        with opener(url, timeout=_TIMEOUT_S) as resp:
            return resp.read()
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        raise RuntimeError(f"could not download {url}: {e}") from e


def _install_root(root: str | Path) -> Path:
    """`find_root` of `root`, else `root` itself when it is an existing folder."""
    found = find_root(root)
    if found is not None:
        return found
    p = _clean_path(root)
    if p is None or not _is_dir(p):
        raise RuntimeError(f"{root} is not an OpenPose folder")
    return p


def _variant(name: str | None) -> Body25bVariant:
    key = name or DEFAULT_VARIANT
    if key not in BODY25B_VARIANTS:
        raise ValueError(f"unknown {MODEL_POSE} variant {key!r}; "
                         f"available: {sorted(BODY25B_VARIANTS)}")
    return BODY25B_VARIANTS[key]


def _fetch_prototxt(v: Body25bVariant, opener: Opener) -> bytes:
    """The variant's official prototxt; RuntimeError unless its SHA-256 matches."""
    data = _fetch(v.prototxt_url, opener)
    got = hashlib.sha256(data).hexdigest()
    if got != v.prototxt_sha256:
        raise RuntimeError(f"downloaded {PROTOTXT_NAME} ({v.name}) has an unexpected "
                           f"SHA-256 ({got}); nothing was written")
    return data


def download_prototxt(root: str | Path, variant: str,
                      *, opener: Opener = urllib.request.urlopen) -> Path:
    """Download the variant's official prototxt to ``models/pose/body_25b/pose_deploy.prototxt``.

    The file is written only when its SHA-256 matches; raises RuntimeError otherwise
    and when it cannot be written.
    """
    v = _variant(variant)
    root = _install_root(root)
    data = _fetch_prototxt(v, opener)
    dest = model_dir(root) / PROTOTXT_NAME
    _write_file(dest, data)
    return dest


def _replace_prototxt(prototxt: Path, v: Body25bVariant, opener: Opener) -> None:
    """Install the official prototxt of `v`, keeping a non-official one as ``.bak``.

    Downloads and checks it before touching the disk, so a failed download
    changes nothing. An official prototxt is not backed up (it can be downloaded
    again any time).
    """
    data = _fetch_prototxt(v, opener)
    if _is_file(prototxt) and prototxt_variant(prototxt) is None:
        bak = prototxt.with_name(PROTOTXT_NAME + ".bak")
        try:
            shutil.copyfile(prototxt, bak)
        except OSError as e:
            raise RuntimeError(f"cannot back up {prototxt} to {bak.name}: {e}") from e
    _write_file(prototxt, data)


def _copy_weights(src: Path, dest: Path) -> None:
    """Atomic copy (``.part`` then replace); nothing to do when `src` already is `dest`."""
    part = dest.with_name(dest.name + ".part")
    try:
        if _is_file(dest) and os.path.samefile(src, dest):
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, part)
        os.replace(part, dest)
    except OSError as e:
        raise RuntimeError(f"could not copy {src} to {dest}: {e}") from e
    finally:
        _unlink_quietly(part)


def install_caffemodel(root: str | Path, src: str | Path, *, variant: str | None = None,
                       opener: Opener = urllib.request.urlopen) -> InstallStatus:
    """Install user-supplied BODY_25B weights (copied as ``pose_iter_XXXXXX.caffemodel``).

    The weights must match the prototxt. With `variant` None an installed official
    prototxt is kept, whichever variant it is. Otherwise the official prototxt of
    `variant` (default "1_25BBkg") is downloaded first when it is not the one
    installed; a non-official prototxt is kept as ``pose_deploy.prototxt.bak``.
    The copy is atomic. Returns `check_install` of the root. RuntimeError for
    download / disk problems, ValueError for an unknown variant name.
    """
    wanted = _variant(variant) if variant else None
    root = _install_root(root)
    src = Path(src)
    if _file_size(src) == 0:
        raise RuntimeError(f"{src} is not a (non-empty) file")
    mdir = model_dir(root)
    prototxt = mdir / PROTOTXT_NAME
    installed = prototxt_variant(prototxt)
    if installed is None or (wanted is not None and wanted.name != installed):
        _replace_prototxt(prototxt, wanted or _variant(None), opener)
    _copy_weights(src, mdir / CAFFEMODEL_NAME)
    return check_install(root)


_WIN_BAD_CHARS = frozenset('<>:"|?*')
_WIN_DEVICES = frozenset({"con", "prn", "aux", "nul",
                          *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))})


def _bad_component(part: str) -> bool:
    """True for a path component that is unsafe or special on Windows."""
    return (part == ".."
            or part != part.rstrip(". ")             # Windows drops trailing dots / spaces
            or any(c in _WIN_BAD_CHARS or ord(c) < 32 for c in part)  # ':' = drive / stream
            or part.split(".")[0].rstrip(" ").lower() in _WIN_DEVICES)


def _safe_member(name: str) -> PurePosixPath:
    """Relative path of a zip member, safe to join to the target folder on any OS.

    RuntimeError for absolute / UNC / drive paths, '..', and components Windows
    treats specially: anything with ':' (``C:x`` switches drive, ``x:stream`` is
    an alternate data stream), reserved device names (CON, NUL.txt, ...),
    trailing dots or spaces, and characters not allowed in Windows file names.
    """
    rel = PurePosixPath(name.replace("\\", "/"))
    if (rel.is_absolute() or PureWindowsPath(name).anchor or not rel.parts
            or any(_bad_component(part) for part in rel.parts)):
        raise RuntimeError(f"unsafe path in the OpenPose archive: {name!r}")
    return rel


def _is_inside(folder: str, target: str) -> bool:
    """True when absolute path `target` is `folder` or below it."""
    try:
        common = os.path.commonpath([folder, target])
    except ValueError:  # different drives (Windows)
        return False
    return os.path.normcase(common) == os.path.normcase(folder)


def _no_space_message(folder: Path, need: int, free: int) -> str:
    return (f"not enough free disk space in {folder}: "
            f"{need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")


def _extract_zip(zip_path: Path, dest: Path, cancelled: Callable[[], bool] | None) -> None:
    """Unpack into `dest`, dropping the archive's top ``openpose/`` folder.

    Every member is checked before anything is written: its name (`_safe_member`)
    and that the joined target path stays inside `dest` (no zip-slip).
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = [(info, _safe_member(info.filename)) for info in zf.infolist()]
        tops = {rel.parts[0].lower() for _, rel in members}
        strip = 1 if tops == {"openpose"} else 0
        base = os.path.abspath(dest)
        plan = []
        for info, rel in members:
            parts = rel.parts[strip:]
            if not parts:
                continue
            target = dest.joinpath(*parts)
            if not _is_inside(base, os.path.abspath(target)):
                raise RuntimeError(f"unsafe path in the OpenPose archive: {info.filename!r}")
            plan.append((info, target))
        need = sum(info.file_size for info, _ in plan)
        free = shutil.disk_usage(dest.parent).free
        if need > free:
            raise RuntimeError(_no_space_message(dest.parent, need, free))
        dest.mkdir(exist_ok=True)
        for info, target in plan:
            if cancelled and cancelled():
                raise RuntimeError("OpenPose download cancelled")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout, _CHUNK)


def _read_some(resp) -> bytes:
    """Up to `_NET_CHUNK` bytes as soon as some arrived (``read1``): on a slow link
    ``read(n)`` would block until all n bytes are in, freezing progress and cancel."""
    read1 = getattr(resp, "read1", None)
    return read1(_NET_CHUNK) if read1 is not None else resp.read(_NET_CHUNK)


def _stream_to_file(url: str, part: Path, size: int, opener: Opener,
                    progress: Callable[[int, int], None] | None,
                    cancelled: Callable[[], bool] | None) -> str:
    """Download `url` (exactly `size` bytes) into `part`; returns the SHA-256 of the bytes.

    RuntimeError "cannot write ..." for local disk problems, "could not download
    ..." for network problems.
    """
    try:
        part.parent.mkdir(parents=True, exist_ok=True)
        f = open(part, "wb")
    except OSError as e:
        raise RuntimeError(f"cannot write {part}: {e}") from e
    h = hashlib.sha256()
    done = 0
    with f:
        try:
            resp = opener(url, timeout=_TIMEOUT_S)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            raise RuntimeError(f"could not download {url}: {e}") from e
        with resp:
            while True:
                if cancelled and cancelled():
                    raise RuntimeError("OpenPose download cancelled")
                try:
                    chunk = _read_some(resp)
                except (http.client.HTTPException, OSError) as e:
                    raise RuntimeError(f"could not download {url}: {e}") from e
                if not chunk:
                    break
                done += len(chunk)
                if done > size:
                    raise RuntimeError(f"download is larger than expected ({size} bytes)")
                try:
                    f.write(chunk)
                except OSError as e:
                    raise RuntimeError(f"cannot write {part}: {e}") from e
                h.update(chunk)
                if progress:
                    progress(done, size)
    if done != size:
        raise RuntimeError(f"download incomplete: got {done} of {size} bytes")
    return h.hexdigest()


def _inside_bundle(path: Path) -> bool:
    """True when `path` is the PoseAssess folder or inside it."""
    try:
        p, bundle = path.resolve(), bundle_root()
    except OSError:
        return False
    return p == bundle or bundle in p.parents


def download_openpose(dest_parent: str | Path, *,
                      progress: Callable[[int, int], None] | None = None,
                      cancelled: Callable[[], bool] | None = None,
                      url: str = OFFICIAL_ZIP_URL, size: int = OFFICIAL_ZIP_SIZE,
                      sha256: str = OFFICIAL_ZIP_SHA256,
                      opener: Opener = urllib.request.urlopen) -> Path:
    """Download and unpack the official OpenPose release into ``<dest_parent>/openpose``.

    Only call this after the user accepted the OpenPose licence (`LICENSE_URL`).
    A folder inside PoseAssess is refused: PoseAssess is copied to other PCs, and
    OpenPose must not be passed on. ``C:\\`` is a good `dest_parent` (gives
    ``C:\\openpose``): only folders are created directly in `dest_parent`, which a
    drive root allows ordinary users. The archive is streamed to
    ``<dest_parent>/openpose.download/<zip>.part`` (`progress(done_bytes, total_bytes)`),
    checked against `size` and `sha256` (both required), extracted safely (nothing
    outside the target folder) and deleted; returns the install root
    (``<dest_parent>/openpose``). Raises RuntimeError on failure or when
    `cancelled()` turns true (partial files are removed), ValueError for a missing
    `size` / `sha256`. Refuses to overwrite a non-empty ``<dest_parent>/openpose``.
    """
    sha256 = (sha256 or "").strip().lower()
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError("sha256 must be the SHA-256 (64 hex digits) of the archive")
    if not size or size <= 0:
        raise ValueError("size must be the size of the archive in bytes")
    dest_parent = Path(dest_parent)
    final = dest_parent / "openpose"
    if _inside_bundle(final):
        raise RuntimeError(f"do not put OpenPose inside the PoseAssess folder ({bundle_root()}): "
                           "it would be copied along with PoseAssess, and the OpenPose licence "
                           "does not allow passing it on. Choose another folder, e.g. C:\\")
    try:
        occupied = final.exists() and (not final.is_dir() or any(final.iterdir()))
    except OSError as e:
        raise RuntimeError(f"cannot use {final}: {e}") from e
    if occupied:
        raise RuntimeError(f"{final} already exists; choose another folder or remove it")
    try:
        dest_parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dest_parent).free
    except OSError as e:
        raise RuntimeError(f"cannot write to {dest_parent}: {e}") from e
    # the archive and its unpacked files are on disk together for a while
    need = size + (OFFICIAL_UNPACKED_SIZE if sha256 == OFFICIAL_ZIP_SHA256 else 3 * size)
    if free < need:
        raise RuntimeError(_no_space_message(dest_parent, need, free))

    zip_name = PurePosixPath(urllib.parse.urlsplit(url).path).name or OFFICIAL_ZIP_NAME
    work = dest_parent / _DOWNLOAD_DIR
    part = work / f"{zip_name}.part"
    done = False
    try:
        got = _stream_to_file(url, part, size, opener, progress, cancelled)
        if got != sha256:
            raise RuntimeError(f"downloaded {zip_name} has an unexpected SHA-256 ({got}); "
                               "it was deleted")
        # Unpacked in place (no folder rename afterwards: on Windows a virus scanner
        # still reading the new exe would make that fail).
        try:
            _extract_zip(part, final, cancelled)
        except (zipfile.BadZipFile, OSError) as e:
            raise RuntimeError(f"could not unpack {zip_name}: {e}") from e
        root = find_root(final)
        if root is None:
            raise RuntimeError(f"{zip_name} does not contain {EXE_NAME}")
        done = True
    finally:
        _unlink_quietly(part)
        with contextlib.suppress(OSError):
            work.rmdir()  # only when empty: never deletes anything else
        if not done:
            shutil.rmtree(final, ignore_errors=True)
    return root


# --------------------------------------------------------------------------- #
# GPU / driver check
# --------------------------------------------------------------------------- #
def _nvidia_smi_exes() -> list[str]:
    exes = ["nvidia-smi"]
    if _is_windows():  # drivers before ~R418 install it here, not in System32
        pf = os.environ.get("ProgramFiles", "C:\\Program Files")
        exes.append(str(Path(pf, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")))
    return exes


def _query_nvidia_smi(run: Callable[..., subprocess.CompletedProcess],
                      fields: str) -> tuple[bool, str | None]:
    """(nvidia-smi found, stdout when it succeeded)."""
    kwargs: dict = {"capture_output": True, "text": True, "errors": "replace", "timeout": 15}
    if _is_windows():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    found = False
    for exe in _nvidia_smi_exes():
        try:
            r = run([exe, f"--query-gpu={fields}", "--format=csv,noheader"], **kwargs)
        except (OSError, subprocess.SubprocessError):
            continue
        found = True
        if r.returncode == 0 and (r.stdout or "").strip():
            return True, r.stdout
    return found, None


def _parse_gpus(stdout: str, with_cc: bool) -> list[dict]:
    gpus = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [s.strip() for s in line.rsplit(",", 2 if with_cc else 1)]
        if len(parts) != (3 if with_cc else 2):
            continue
        cc = parts[2] if with_cc else None
        if cc is not None and not cc.replace(".", "").isdigit():
            cc = None  # "[N/A]"
        gpus.append({"name": parts[0], "driver_version": parts[1], "compute_cap": cc})
    return gpus


def gpu_info(*, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict | None:
    """First NVIDIA GPU as {"name", "driver_version", "compute_cap" (str | None), "gpus": [...]}.

    Asks ``nvidia-smi`` (without compute_cap on old drivers that do not know that
    field). None when nvidia-smi is missing or reports no GPU.
    """
    found, out = _query_nvidia_smi(run, "name,driver_version,compute_cap")
    gpus = _parse_gpus(out, True) if out else []
    if not gpus and found:
        _, out = _query_nvidia_smi(run, "name,driver_version")
        gpus = _parse_gpus(out, False) if out else []
    if not gpus:
        return None
    return {**gpus[0], "gpus": gpus}


def _version_tuple(s: str | None) -> tuple[int, ...] | None:
    try:
        return tuple(int(x) for x in str(s).strip().split("."))
    except ValueError:
        return None


def gpu_warnings(info: dict | None) -> list[str]:
    """Reasons why OpenPose 1.7.0 (GPU build) may not run on this PC; empty when fine."""
    if not info or not info.get("gpus", [info]):
        return ["no NVIDIA GPU found (nvidia-smi is missing or reports no GPU): the OpenPose "
                "GPU build needs an NVIDIA graphics card and driver"]
    out = []
    driver = _version_tuple(info.get("driver_version"))
    if driver is not None and driver < MIN_DRIVER:
        need = ".".join(map(str, MIN_DRIVER))
        out.append(f"NVIDIA driver {info.get('driver_version')} is too old for OpenPose "
                   f"{OFFICIAL_VERSION}: update it to {need} or newer")
    for gpu in info.get("gpus", [info]):
        cc = _version_tuple(gpu.get("compute_cap"))
        if cc is None:
            continue
        name, ccs = gpu.get("name", "GPU"), gpu.get("compute_cap")
        if cc >= MAX_COMPUTE_CAP:
            out.append(f"{name} (compute capability {ccs}) is too new for OpenPose "
                       f"{OFFICIAL_VERSION}: RTX 50 series / Blackwell GPUs are not supported")
        elif cc < MIN_COMPUTE_CAP:
            out.append(f"{name} (compute capability {ccs}) is too old for OpenPose "
                       f"{OFFICIAL_VERSION} (needs 5.2 or newer)")
    return out
