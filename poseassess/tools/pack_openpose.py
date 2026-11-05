"""Make the lab's INTERNAL PoseAssess package with OpenPose included (never uploaded anywhere).

Run from the portable PoseAssess folder (``pack_with_openpose.bat`` does this)::

    python -m poseassess.tools.pack_openpose [--src DIR] [--no-zip | --zip-path PATH] [--yes]

1. Finds this PC's OpenPose install (``--src``, else `resolve_openpose_dir`, which finds e.g.
   ``C:\\openpose``) and checks that it runs BODY_25B (`check_install`).
2. Shows the OpenPose license note and asks for confirmation (``--yes`` skips every question).
3. Copies a SLIM install into ``<PoseAssess folder>/openpose``: the whole ``bin`` folder (all its
   DLLs are needed), ``models/pose/body_25b/pose_deploy.prototxt`` and
   ``pose_iter_XXXXXX.caffemodel`` only (no other weights, no face / hand / COCO models), and
   CMU's license / readme / third-party notice files. The copy goes to a temporary sibling folder
   that is renamed at the end, so a failed copy never leaves a half-copied ``openpose`` folder;
   an existing ``openpose`` folder is replaced only after confirmation.
4. Unless ``--no-zip``: zips PoseAssess's own files (``python311``, ``venv``, ``poseassess``,
   ``models``, ``openpose``, ``docs``, the launchers and the readme / requirement lists; see
   `BUNDLE_DIRS`, `BUNDLE_FILE_PATTERNS`) to ``<parent>/<folder>_with_OpenPose.zip`` for the
   lab's shared drive. Anything else in the folder (e.g. ``recordings`` of ``run_wii.bat``, i.e.
   participant data) is left out and listed; ``--include NAME`` adds such an entry. Only the app
   source drops its ``__pycache__`` (as the public build does); the compiled bytecode of
   ``venv`` and ``python311`` is kept. ``--zip-path`` takes a file (``*.zip``) or a folder.

Only the portable PoseAssess folder is packed (it needs ``run.bat``, ``python311\\python.exe``
and ``venv\\pyvenv.cfg``); a development checkout is refused.

OpenPose (CMU) may be used only for noncommercial internal research purposes and must not be
redistributed: the package is a copy for internal noncommercial use at a single site of the
licensee's organization, with CMU's notices kept. Do not upload it.

Exit codes: 0 done, 2 OpenPose not found / not ready, 3 copy or zip failed (also a folder or
destination that cannot be used), 64 wrong command-line options, 130 cancelled.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import os
import shutil
import stat
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from poseassess.core import openpose_setup as ops

EXIT_OK, EXIT_NOT_READY, EXIT_FAILED, EXIT_USAGE, EXIT_CANCELLED = 0, 2, 3, 64, 130

DEST_NAME = "openpose"
NEW_DIR = ".openpose.new"    # temporary sibling of <bundle>/openpose while copying
OLD_DIR = ".openpose.old"    # the replaced <bundle>/openpose until it is deleted
ZIP_SUFFIX = "_with_OpenPose.zip"
SHARED_DRIVE_SENTENCE = ("Put this zip on the lab's shared drive; "
                         "do not upload it (OpenPose license).")

#: Root files kept with the slim copy (case-insensitive): CMU's license / readme / version notes.
NOTICE_PATTERNS = ("license*", "licence*", "copying*", "notice*", "readme*", "*.txt", "*.md")
#: Root folders with third-party code: only their license / version notes are kept.
THIRD_PARTY_DIRS = frozenset({"3rdparty", "3rd_party", "third_party", "thirdparty"})
THIRD_PARTY_NOTICE_PATTERNS = ("license*", "licence*", "copying*", "notice*", "version*")
#: Files that make a folder the portable PoseAssess folder (a development checkout lacks some).
BUNDLE_MARKERS = (Path("run.bat"), Path("python311", "python.exe"), Path("venv", "pyvenv.cfg"))
#: Top-level folders / files of the portable PoseAssess folder that go into the zip
#: (case-insensitive); everything else (recordings, projects, wheels, ...) is left out.
BUNDLE_DIRS = frozenset({"python311", "venv", "poseassess", "models", "docs", DEST_NAME})
BUNDLE_FILE_PATTERNS = ("*.bat", "readme*", "license*", "requirements*.txt", "pip_freeze.txt",
                        "models_list.txt")
#: Directory names never put into the zip (at any depth).
ZIP_EXCLUDE_DIRS = frozenset({".git", ".pytest_cache", ".ruff_cache", ".mypy_cache"})
#: Top-level folders whose ``__pycache__`` is dropped: only the app source, as the public build
#: does. The bytecode pip / Python compiled in venv and python311 is kept, so the first start on
#: each PC (or every start from a read-only share) does not recompile the whole stack.
PYCACHE_FREE_DIRS = frozenset({"poseassess"})
#: Stored without compression (weights and archives hardly compress; deflating them is slow).
STORED_SUFFIXES = frozenset({".caffemodel", ".onnx", ".pth", ".pt", ".zip", ".7z", ".gz",
                             ".bz2", ".xz", ".whl", ".mp4", ".avi", ".mkv", ".png", ".jpg"})
ZIP_LEVEL = 1
_CHUNK = 1 << 20
_PROGRESS_INTERVAL_S = 0.25
_RENAME_ATTEMPTS = 10        # a virus scanner reading a just-copied exe blocks renames briefly
_RENAME_DELAY_S = 0.5

GUI_HINT = "in PoseAssess: 1. Project > Get OpenPose..."

LICENSE_NOTE = f"""\
OpenPose license (CMU): OpenPose may be used only for noncommercial internal research purposes.
The package made here is a copy for internal noncommercial use at a single site of the
licensee's organization (the lab whose members accepted the OpenPose license). CMU's copyright
and license notices are copied with it and must stay with it. Keep it on the lab's shared drive:
do not upload, publish or give it to anyone outside the lab.
License: {ops.LICENSE_URL}"""


class PackError(RuntimeError):
    """A copy / zip problem with a user-facing message (exit code 3)."""


class Cancelled(Exception):
    """The user said no (exit code 130)."""


@dataclass(frozen=True)
class CopyItem:
    """One file of the slim copy."""
    src: Path
    rel: Path      # relative to the install root (and to <bundle>/openpose)
    kind: str      # "program", "model" or "notice"
    size: int


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


def _norm(p: Path) -> str:
    try:
        p = p.resolve()
    except OSError:
        p = p.absolute()
    return os.path.normcase(str(p))


def _same_or_inside(path: Path, folder: Path) -> bool:
    """True when `path` is `folder` or below it (links resolved, case-insensitive on Windows)."""
    a, b = _norm(path), _norm(folder)
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:  # different drives
        return False


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatchcase(low, pat) for pat in patterns)


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _walk_files(folder: Path, keep: Callable[[str], bool] = lambda name: True) -> list[Path]:
    """Files below `folder` (sorted, symlinked folders not followed) whose name passes `keep`."""
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        out += [Path(dirpath, f) for f in sorted(filenames) if keep(f)]
    return out


def _is_link(p: Path) -> bool:
    """Symlink or Windows junction (removed as a link, never by deleting what it points to)."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    junction = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)
    return stat.S_ISLNK(st.st_mode) or (
        junction is not None and getattr(st, "st_reparse_tag", None) == junction)


def _exists(p: Path) -> bool:
    """True for an existing file / folder and for a link (also a broken one)."""
    return p.exists() or _is_link(p)


def _link_target(p: Path) -> str:
    try:
        return os.readlink(p)
    except (OSError, ValueError):
        return "?"


def _remove(p: Path) -> bool:
    """Delete a file / folder / link; True when it is gone."""
    with contextlib.suppress(OSError):
        if _is_link(p):
            try:
                os.unlink(p)
            except OSError:
                os.rmdir(p)  # a junction
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    return not _exists(p)


def _rename(src: Path, dst: Path) -> None:
    """`os.rename`, retried for a few seconds (Windows: files still open by a virus scanner)."""
    for attempt in range(_RENAME_ATTEMPTS):
        try:
            os.rename(src, dst)
            return
        except OSError:
            if attempt == _RENAME_ATTEMPTS - 1:
                raise
            time.sleep(_RENAME_DELAY_S)


class _Progress:
    """One progress line, rewritten in place on a console (only the final state otherwise)."""

    def __init__(self, label: str, total: int):
        self.label, self.total = label, max(total, 1)
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._last = 0.0

    def _line(self, done: int, files: str) -> str:
        pct = min(100, done * 100 // self.total)
        return f"  {self.label}: {files}{done / 1e6:,.0f} / {self.total / 1e6:,.0f} MB ({pct}%)"

    def update(self, done: int, files: str = "") -> None:
        now = time.monotonic()
        if self.tty and now - self._last >= _PROGRESS_INTERVAL_S:
            self._last = now
            print("\r" + self._line(done, files), end="", flush=True)

    def finish(self, done: int, files: str = "") -> None:
        print(("\r" if self.tty else "") + self._line(done, files), flush=True)


# --------------------------------------------------------------------------- #
# source, plan and copy
# --------------------------------------------------------------------------- #
def find_source(dest: Path) -> tuple[Path | None, str]:
    """(install root, where it came from) of this PC's OpenPose (`resolve_openpose_dir`).

    When that is the PoseAssess folder's own copy (`dest`, made by an earlier run), the other
    places (remembered folder, ``C:\\openpose``, ...) are searched for the original install.
    """
    root, origin = ops.resolve_openpose_dir(None)
    if root is None or not _same_or_inside(root, dest):
        return root, origin
    sources: list[tuple[Path, str]] = []
    remembered = ops.remembered_dir()
    if remembered is not None:
        sources.append((remembered, "remembered on this computer"))
    sources += [(c, "") for c in ops.candidate_dirs()]
    partial: tuple[Path | None, str] | None = None
    for path, where in sources:
        st = ops.check_install(path)
        if st.root is None or _same_or_inside(st.root, dest):
            continue
        found = (st.root, where or f"found at {st.root}")
        if st.ok:
            return found
        partial = partial or found
    if partial is not None:
        return partial
    return None, (f"only the PoseAssess folder's own copy ({dest}) was found; give the original "
                  "OpenPose folder with --src DIR")


def plan_slim_copy(root: Path, exe: Path) -> list[CopyItem]:
    """Files of the slim BODY_25B install: the program folder (``bin``), the two model files and
    the root license / readme files plus third-party notices (see the module docstring)."""
    items: dict[str, CopyItem] = {}

    def add(src: Path, kind: str) -> None:
        rel = src.relative_to(root)
        items.setdefault(os.path.normcase(str(rel)), CopyItem(src, rel, kind, _size(src)))

    exe_dir = exe.parent
    if _norm(exe_dir) == _norm(root):  # OpenPoseDemo.exe in the root: its files, not subfolders
        for f in sorted(root.iterdir()):
            if f.is_file():
                add(f, "program")
    else:
        for f in _walk_files(exe_dir):
            add(f, "program")
    mdir = ops.model_dir(root)
    for name in (ops.PROTOTXT_NAME, ops.CAFFEMODEL_NAME):
        add(mdir / name, "model")
    for entry in sorted(root.iterdir()):
        low = entry.name.lower()
        if entry.is_file() and _matches(entry.name, NOTICE_PATTERNS):
            add(entry, "notice")
        elif entry.is_dir() and _norm(entry) != _norm(exe_dir):
            if "licen" in low:  # e.g. licenses/, 3rdparty_licenses/: the whole folder
                for f in _walk_files(entry):
                    add(f, "notice")
            elif low in THIRD_PARTY_DIRS:  # third-party tools: only their notices
                for f in _walk_files(entry, lambda n: _matches(n, THIRD_PARTY_NOTICE_PATTERNS)):
                    add(f, "notice")
    return list(items.values())


def _copy_files(plan: list[CopyItem], target: Path) -> None:
    total = sum(i.size for i in plan)
    progress = _Progress("copying", total)
    done = 0
    for n, item in enumerate(plan, 1):
        out = target / item.rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.src, out)
        done += item.size
        progress.update(done, f"{n}/{len(plan)} files, ")
    progress.finish(done, f"{len(plan)}/{len(plan)} files, ")


def _put_back(old: Path, dest: Path) -> str:
    """Rename the previous copy `old` back to `dest` when `dest` is gone.

    Returns "" when nothing is left to do, else a note that says where the previous copy is.
    """
    if _exists(dest) or not _exists(old):
        return ""
    try:
        _rename(old, dest)
        return ""
    except OSError as e:
        return (f". The previous OpenPose copy is in {old} (putting it back failed: {e}); "
                f"rename it back to {dest.name}, or run this tool again")


def _swap_in(new: Path, dest: Path) -> None:
    """Rename `new` to `dest`, replacing an existing `dest`.

    An existing `dest` is first renamed to ``.openpose.old``. If a rename error or Ctrl+C stops
    the swap before `new` is in place, the previous copy is put back; should even that fail, the
    error names ``.openpose.old``. In that case, and when the process is killed (closed console
    window, power cut), the next run puts it back (`recover_interrupted`).
    """
    old = dest.with_name(OLD_DIR)
    had_old = _exists(dest)
    try:
        if had_old:
            try:
                _rename(dest, old)
            except OSError as e:
                raise PackError(f"cannot replace {dest} (is OpenPose or PoseAssess running from "
                                f"it? close it and try again): {e}") from e
        _rename(new, dest)
    except BaseException as e:
        note = _put_back(old, dest) if had_old else ""
        if isinstance(e, OSError):
            raise PackError(f"cannot rename {new.name} to {dest}: {e}{note}") from e
        if isinstance(e, PackError) and note:
            raise PackError(f"{e}{note}") from e
        raise
    if had_old and not _remove(old):
        _say(f"  Note: could not delete all of the old copy {old}; delete it by hand.")


def recover_interrupted(dest: Path) -> None:
    """Clean up after an interrupted earlier run (Ctrl+C, closed window, power cut).

    A half-made ``.openpose.new`` is deleted. A ``.openpose.old`` is the previous copy that was
    moved away just before the swap: it is put back as `dest` when `dest` is missing, and
    deleted only when `dest` is in place (the swap was complete). Raises PackError when that
    cannot be done.
    """
    new, old = dest.with_name(NEW_DIR), dest.with_name(OLD_DIR)
    if _exists(new) and not _remove(new):
        raise PackError(f"cannot delete the leftover folder {new}; delete it by hand")
    if not _exists(old):
        return
    if _exists(dest):
        if not _remove(old):
            raise PackError(f"cannot delete the leftover folder {old} (an old OpenPose copy); "
                            "delete it by hand")
        return
    try:
        _rename(old, dest)
    except OSError as e:
        raise PackError(f"cannot put the previous OpenPose copy {old} back as {dest}: {e}; "
                        f"rename it to {dest.name} by hand") from e
    _say(f"Put back the previous OpenPose copy that an interrupted run had left in {old.name}: "
         f"{dest}")


def copy_slim(plan: list[CopyItem], dest: Path) -> ops.InstallStatus:
    """Copy `plan` into ``<dest parent>/.openpose.new``, check it, then rename it to `dest`.

    A failed or cancelled copy leaves no ``.openpose.new`` behind, and an existing `dest` is
    replaced only by a complete, checked copy (see `_swap_in`; if even putting the previous
    copy back fails, the error says where it is).
    """
    new = dest.with_name(NEW_DIR)
    total = sum(i.size for i in plan)
    recover_interrupted(dest)
    try:
        free = shutil.disk_usage(dest.parent).free
    except OSError as e:
        raise PackError(f"cannot use {dest.parent}: {e}") from e
    if free < total + (50 << 20):
        raise PackError(f"not enough free disk space in {dest.parent}: {_mb(total)} needed, "
                        f"{_mb(free)} free")
    try:
        new.mkdir()
        _copy_files(plan, new)
        st = ops.check_install(new)
        if not st.ok:
            raise PackError("the copied OpenPose is not complete: " + "; ".join(st.errors))
        _swap_in(new, dest)
    except OSError as e:
        raise PackError(f"copying OpenPose failed: {e}") from e
    finally:
        if _exists(new):
            _remove(new)
    return ops.check_install(dest)


def _report_sizes(plan: list[CopyItem], dest: Path) -> None:
    rows = [("program files (bin)", "program"), (f"{ops.MODEL_POSE} model", "model"),
            ("license / readme files", "notice")]
    _say(f"Copied OpenPose to {dest}:")
    for title, kind in rows:
        sel = [i for i in plan if i.kind == kind]
        _say(f"  {title:<24}{len(sel):>5} files {_mb(sum(i.size for i in sel)):>14}")
    _say(f"  {'total':<24}{len(plan):>5} files {_mb(sum(i.size for i in plan)):>14}")


# --------------------------------------------------------------------------- #
# zip
# --------------------------------------------------------------------------- #
def bundle_entries(bundle: Path, include: Sequence[str] = ()) -> tuple[list[Path], list[Path]]:
    """(zipped, left out): the top-level entries of the PoseAssess folder.

    Zipped are PoseAssess's own folders / files (`BUNDLE_DIRS`, `BUNDLE_FILE_PATTERNS`) and the
    names in `include`; everything else (``recordings``, projects, ...) is left out.
    """
    extra = {n.lower() for n in include}
    keep, skip = [], []
    for entry in sorted(bundle.iterdir(), key=lambda p: p.name.lower()):
        low = entry.name.lower()
        own = ((low in BUNDLE_DIRS and entry.is_dir())
               or (entry.is_file() and _matches(entry.name, BUNDLE_FILE_PATTERNS)))
        (keep if own or low in extra else skip).append(entry)
    return keep, skip


def _zip_entries(entries: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    """(folders, files) for the zip below the top-level `entries`.

    `ZIP_EXCLUDE_DIRS` are skipped everywhere, ``__pycache__`` only in `PYCACHE_FREE_DIRS`.
    """
    dirs, files = [], []
    for entry in entries:
        if not entry.is_dir():
            files.append(entry)
            continue
        dirs.append(entry)
        skip = ZIP_EXCLUDE_DIRS | ({"__pycache__"} if entry.name.lower() in PYCACHE_FREE_DIRS
                                   else set())
        for dirpath, dirnames, filenames in os.walk(entry):
            dirnames[:] = sorted(d for d in dirnames if d.lower() not in skip)
            dirs += [Path(dirpath, d) for d in dirnames]
            files += [Path(dirpath, f) for f in sorted(filenames)]
    return dirs, files


def _set_level(zinfo: zipfile.ZipInfo, level: int) -> None:
    if hasattr(zinfo, "compress_level"):  # Python 3.13+
        zinfo.compress_level = level
    else:
        zinfo._compresslevel = level


def write_zip(bundle: Path, zip_path: Path, entries: Sequence[Path] | None = None) -> int:
    """Zip `bundle` (the folder itself, as the top entry) to `zip_path`; returns the zip size.

    Only the top-level `entries` go in (default: `bundle_entries`, PoseAssess's own files).
    Written as ``<zip>.part`` and renamed when complete; the part file is deleted on failure.
    """
    top = bundle.name or "PoseAssess"
    if entries is None:
        entries = bundle_entries(bundle)[0]
    dirs, files = _zip_entries(entries)
    sizes = [_size(f) for f in files]
    total = sum(sizes)
    part = zip_path.with_name(zip_path.name + ".part")
    progress = _Progress("zipping", total)
    done = 0
    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, compresslevel=ZIP_LEVEL,
                             allowZip64=True, strict_timestamps=False) as zf:
            zf.write(bundle, top)
            for d in dirs:
                zf.write(d, f"{top}/{d.relative_to(bundle).as_posix()}")
            for f, size in zip(files, sizes):
                arc = f"{top}/{f.relative_to(bundle).as_posix()}"
                zinfo = zipfile.ZipInfo.from_file(f, arc, strict_timestamps=False)
                if f.suffix.lower() in STORED_SUFFIXES:
                    zinfo.compress_type = zipfile.ZIP_STORED
                else:
                    zinfo.compress_type = zipfile.ZIP_DEFLATED
                    _set_level(zinfo, ZIP_LEVEL)
                zip64 = size > (1 << 30)
                with open(f, "rb") as fin, zf.open(zinfo, "w", force_zip64=zip64) as out:
                    while chunk := fin.read(_CHUNK):
                        out.write(chunk)
                        done += len(chunk)
                        progress.update(done)
        progress.finish(done)
        os.replace(part, zip_path)
    except (OSError, RuntimeError, ValueError, zipfile.LargeZipFile) as e:
        raise PackError(f"writing {zip_path} failed: {e}") from e
    finally:
        with contextlib.suppress(OSError):
            part.unlink(missing_ok=True)
    return _size(zip_path)


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # usage errors exit with 64: 2 means "OpenPose not ready"
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    ap = _Parser(
        prog="python -m poseassess.tools.pack_openpose",
        description="Make the lab's INTERNAL PoseAssess package with OpenPose (BODY_25B) "
                    "included: copies the needed parts of this PC's OpenPose into the "
                    "PoseAssess folder and zips that folder for the lab's shared drive. "
                    "Never upload the result (OpenPose license).")
    ap.add_argument("--src", metavar="DIR", default=None,
                    help="OpenPose folder to copy (default: the one PoseAssess finds, e.g. "
                         "C:\\openpose)")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--no-zip", action="store_true",
                     help="only copy OpenPose into the PoseAssess folder, do not write a zip")
    out.add_argument("--zip-path", metavar="PATH", default=None,
                     help="zip file (*.zip) to write, or a folder to write "
                          "<folder>_with_OpenPose.zip into (default: next to the PoseAssess "
                          "folder)")
    ap.add_argument("--include", metavar="NAME", action="append", default=[],
                    help="also zip this file / folder of the PoseAssess folder (repeatable); "
                         "by default only PoseAssess's own files go in, not e.g. recordings")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="do not ask (accept the license note, replace an existing openpose "
                         "folder and zip file)")
    return ap


def _confirm(ask: Callable[[str], str], prompt: str, *, default_yes: bool) -> bool:
    try:
        answer = ask(prompt).strip().lower()
    except EOFError:
        return False
    if not answer:
        return default_yes
    return answer in ("y", "yes")


def _zip_target(arg: str | None, bundle: Path) -> Path:
    """The zip file for ``--zip-path``: a ``*.zip`` name is the file; an existing folder, a
    path ending with a separator or a name without ``.zip`` is a folder for the default name."""
    default_name = (bundle.name or "PoseAssess") + ZIP_SUFFIX
    text = (arg or "").strip().strip('"').strip()
    if not text:
        return bundle.parent / default_name
    p = Path(text).expanduser().absolute()
    seps = tuple(s for s in (os.sep, os.altsep) if s)
    if text.endswith(seps) or p.is_dir() or p.suffix.lower() != ".zip":
        return p / default_name
    return p


def _zip_target_problem(zip_path: Path) -> str | None:
    """Why `zip_path` cannot be written (checked before the long copy), else None."""
    if zip_path.is_dir():
        return f"{zip_path} is a folder"
    for parent in zip_path.parents:
        if parent.exists():
            return None if parent.is_dir() else f"{parent} is a file, not a folder"
    return None


def run(args: argparse.Namespace, ask: Callable[[str], str]) -> int:
    bundle = ops.bundle_root()
    dest = bundle / DEST_NAME
    zip_path = None if args.no_zip else _zip_target(args.zip_path, bundle)

    # 0. the PoseAssess folder
    missing = [str(m) for m in BUNDLE_MARKERS if not (bundle / m).is_file()]
    if missing:
        _say(f"{bundle} is not the portable PoseAssess folder (missing: {', '.join(missing)}).")
        _say("Run pack_with_openpose.bat in the PoseAssess_Portable folder.")
        return EXIT_FAILED
    for name in args.include:
        if not _exists(bundle / name):
            _say(f"--include {name}: there is no {name} in {bundle}")
            return EXIT_USAGE
    try:
        recover_interrupted(dest)
    except PackError as e:
        _say(f"[ERROR] {e}")
        return EXIT_FAILED
    if _is_link(dest):
        _say(f"{dest} is a link to {_link_target(dest)}, not a copy of OpenPose. Remove the "
             f'link (e.g. rmdir "{dest}" in a console; the folder it points to is not touched) '
             "and run this again.")
        return EXIT_FAILED

    # 1. source
    if args.src:
        root, origin = ops.find_root(args.src), "given with --src"
        shown = root or args.src
    else:
        root, origin = find_source(dest)
        shown = root
        if root is not None and origin == f"found at {root}":
            origin = "found automatically"
    if shown is None:
        _say(f"OpenPose not found: {origin}")
        _say(f"Install OpenPose BODY_25B first ({GUI_HINT}), or give its folder with --src DIR.")
        return EXIT_NOT_READY
    _say(f"OpenPose source: {shown} ({origin})")
    st = ops.check_install(root if root is not None else args.src)
    if not st.ok:
        _say("OpenPose is not ready for BODY_25B:")
        for e in st.errors:
            _say(f"  - {e}")
        _say(f"Fix it first ({GUI_HINT}), or give another folder with --src.")
        return EXIT_NOT_READY
    for w in st.warnings:
        _say(f"  warning: {w}")
    src = st.root
    assert src is not None and st.exe is not None

    # 2. places
    if _same_or_inside(dest, src):
        _say(f"Cannot copy {src} into {dest}: the destination is the source or inside it.")
        return EXIT_FAILED
    if _same_or_inside(src, dest):
        _say(f"Cannot copy {src}: it is inside the destination {dest} that would be replaced. "
             "Give the original OpenPose folder with --src.")
        return EXIT_FAILED
    if zip_path is not None and _same_or_inside(zip_path, bundle):
        _say(f"Cannot write the zip inside the folder being zipped: {zip_path}")
        return EXIT_FAILED
    problem = _zip_target_problem(zip_path) if zip_path is not None else None
    if problem:
        _say(f"Cannot write the zip {zip_path}: {problem}.")
        return EXIT_FAILED
    zipped = next((e for e in bundle_entries(bundle, args.include)[0]
                   if _same_or_inside(src, e)), None) if zip_path is not None else None
    if zipped is not None:
        _say(f"{src} is inside {zipped}, which goes into the zip, so the zip would contain the "
             "whole OpenPose install. Move it out (e.g. to C:\\openpose) or use --no-zip.")
        return EXIT_FAILED

    # 3. license and questions
    _say()
    _say(LICENSE_NOTE)
    _say()
    if not args.yes and not _confirm(ask, "Press Enter (or y) to continue, n to cancel: ",
                                     default_yes=True):
        _say("Cancelled.")
        return EXIT_CANCELLED
    if _exists(dest) and not args.yes:
        if not _confirm(ask, f"{dest} already exists. Replace it? [y/N]: ", default_yes=False):
            _say("Cancelled; the existing openpose folder was kept.")
            return EXIT_CANCELLED
    if zip_path is not None and _exists(zip_path) and not args.yes:
        if not _confirm(ask, f"{zip_path} already exists. Replace it? [y/N]: ",
                        default_yes=False):
            _say("Cancelled; nothing was changed.")
            return EXIT_CANCELLED

    # 4. slim copy
    try:
        plan = plan_slim_copy(src, st.exe)
        _say(f"Copying {len(plan)} files ({_mb(sum(i.size for i in plan))}) to {dest} ...")
        result = copy_slim(plan, dest)
    except OSError as e:
        _say(f"[ERROR] cannot read {src}: {e}")
        return EXIT_FAILED
    except PackError as e:
        _say()
        _say(f"[ERROR] {e}")
        return EXIT_FAILED
    _report_sizes(plan, dest)
    if not result.ok:
        _say(f"[ERROR] {result.summary()}")
        return EXIT_FAILED
    _say(result.summary())
    for w in result.warnings:
        _say(f"  warning: {w}")

    # 5. zip
    if zip_path is None:
        _say("No zip written (--no-zip).")
        return EXIT_OK
    try:
        keep, skip = bundle_entries(bundle, args.include)
    except OSError as e:
        _say(f"[ERROR] cannot list {bundle}: {e}")
        return EXIT_FAILED
    if skip:
        _say("Left out of the zip (not part of PoseAssess, e.g. recordings or other data; add one "
             "with --include NAME):")
        for p in skip:
            _say(f"  {p.name}{os.sep if p.is_dir() else ''}")
    if zip_path.exists():
        _say(f"Replacing the existing {zip_path}")
    _say(f"Zipping {bundle} ...")
    try:
        size = write_zip(bundle, zip_path, keep)
    except PackError as e:
        _say()
        _say(f"[ERROR] {e}")
        return EXIT_FAILED
    _say()
    _say(f"Done: {zip_path} ({_mb(size)})")
    _say(SHARED_DRIVE_SENTENCE)
    return EXIT_OK


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = input) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in args.include:
        if not name.strip() or name in (".", "..") or any(s in name for s in "/\\:"):
            parser.error(f"--include takes the name of a file / folder directly in the "
                         f"PoseAssess folder, not {name!r}")
    if args.include and args.no_zip:
        parser.error("--include needs a zip (it cannot be used with --no-zip)")
    try:
        return run(args, ask)
    except KeyboardInterrupt:
        _say()
        _say("Cancelled.")
        dest = ops.bundle_root() / DEST_NAME
        old = dest.with_name(OLD_DIR)
        if _exists(old) and not _exists(dest):
            _say(f"The previous OpenPose copy is in {old}; the next run of this tool puts it "
                 f"back (or rename it to {DEST_NAME}).")
        return EXIT_CANCELLED


if __name__ == "__main__":
    sys.exit(main())
