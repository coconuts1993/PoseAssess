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
4. Unless ``--no-zip``: zips the whole PoseAssess folder (without ``__pycache__``) to
   ``<parent>/<folder>_with_OpenPose.zip`` for the lab's shared drive.

OpenPose (CMU) is licensed for non-commercial use only and must not be redistributed: the
package is for internal use at one site of the licensee's organisation. Do not upload it.

Exit codes: 0 done, 2 OpenPose not found / not ready, 3 copy or zip failed (also a destination
that cannot be used), 130 cancelled.
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
from typing import Callable

from poseassess.core import openpose_setup as ops

EXIT_OK, EXIT_NOT_READY, EXIT_FAILED, EXIT_CANCELLED = 0, 2, 3, 130

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
#: Directory names never put into the zip.
ZIP_EXCLUDE_DIRS = frozenset({"__pycache__", ".git", ".pytest_cache", ".ruff_cache",
                              ".mypy_cache", NEW_DIR, OLD_DIR})
#: Stored without compression (weights and archives hardly compress; deflating them is slow).
STORED_SUFFIXES = frozenset({".caffemodel", ".onnx", ".pth", ".pt", ".zip", ".7z", ".gz",
                             ".bz2", ".xz", ".whl", ".mp4", ".avi", ".mkv", ".png", ".jpg"})
ZIP_LEVEL = 1
_CHUNK = 1 << 20
_PROGRESS_INTERVAL_S = 0.25
_RENAME_ATTEMPTS = 10        # a virus scanner reading a just-copied exe blocks renames briefly
_RENAME_DELAY_S = 0.5

LICENSE_NOTE = f"""\
OpenPose license (CMU): OpenPose may be used for non-commercial research and teaching only, and it
must not be passed on. The package made here is for INTERNAL non-commercial use at a single site
of the licensee's organization (the lab whose members accepted the OpenPose license). Keep it on
the lab's shared drive: do not upload, publish or give it to anyone outside the lab.
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
    return not (p.exists() or _is_link(p))


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


def _swap_in(new: Path, dest: Path) -> None:
    """Rename `new` to `dest`, replacing an existing `dest` (put back when the rename fails)."""
    old = dest.with_name(OLD_DIR)
    had_old = dest.exists() or _is_link(dest)
    if had_old:
        try:
            _rename(dest, old)
        except OSError as e:
            raise PackError(f"cannot replace {dest} (is OpenPose or PoseAssess running from it? "
                            f"close it and try again): {e}") from e
    try:
        _rename(new, dest)
    except OSError as e:
        if had_old:
            with contextlib.suppress(OSError):
                _rename(old, dest)
        raise PackError(f"cannot rename {new.name} to {dest}: {e}") from e
    if had_old and not _remove(old):
        _say(f"  Note: could not delete all of the old copy {old}; delete it by hand.")


def copy_slim(plan: list[CopyItem], dest: Path) -> ops.InstallStatus:
    """Copy `plan` into ``<dest parent>/.openpose.new``, check it, then rename it to `dest`.

    Nothing is left behind on failure (PackError) or Ctrl+C; an existing `dest` is kept unless
    the new copy is complete.
    """
    new = dest.with_name(NEW_DIR)
    total = sum(i.size for i in plan)
    for stale in (new, dest.with_name(OLD_DIR)):  # left over by an interrupted earlier run
        if (stale.exists() or _is_link(stale)) and not _remove(stale):
            raise PackError(f"cannot delete the leftover folder {stale}; delete it by hand")
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
        if new.exists() or _is_link(new):
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
def _zip_entries(bundle: Path) -> tuple[list[Path], list[Path]]:
    """(folders, files) of the bundle for the zip, without `ZIP_EXCLUDE_DIRS`."""
    dirs, files = [], []
    for dirpath, dirnames, filenames in os.walk(bundle):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in ZIP_EXCLUDE_DIRS)
        dirs += [Path(dirpath, d) for d in dirnames]
        files += [Path(dirpath, f) for f in sorted(filenames)]
    return dirs, files


def _set_level(zinfo: zipfile.ZipInfo, level: int) -> None:
    if hasattr(zinfo, "compress_level"):  # Python 3.13+
        zinfo.compress_level = level
    else:
        zinfo._compresslevel = level


def write_zip(bundle: Path, zip_path: Path) -> int:
    """Zip `bundle` (the folder itself, as the top entry) to `zip_path`; returns the zip size.

    Written as ``<zip>.part`` and renamed when complete; the part file is deleted on failure.
    """
    top = bundle.name or "PoseAssess"
    dirs, files = _zip_entries(bundle)
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
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
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
                     help="zip file to write (default: <folder>_with_OpenPose.zip next to the "
                          "PoseAssess folder)")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="do not ask (accept the license note, replace an existing openpose "
                         "folder)")
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
    default_name = (bundle.name or "PoseAssess") + ZIP_SUFFIX
    if not arg:
        return bundle.parent / default_name
    p = Path(arg.strip().strip('"')).expanduser().absolute()
    return p / default_name if p.is_dir() else p


def run(args: argparse.Namespace, ask: Callable[[str], str]) -> int:
    bundle = ops.bundle_root()
    dest = bundle / DEST_NAME
    zip_path = None if args.no_zip else _zip_target(args.zip_path, bundle)
    if not (bundle / "run.bat").is_file():
        _say(f"Note: {bundle} has no run.bat; it does not look like the portable PoseAssess "
             "folder.")

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
        _say("Install OpenPose BODY_25B first (PoseAssess, OpenPose settings), or give its "
             "folder with --src DIR.")
        return EXIT_NOT_READY
    _say(f"OpenPose source: {shown} ({origin})")
    st = ops.check_install(root if root is not None else args.src)
    if not st.ok:
        _say("OpenPose is not ready for BODY_25B:")
        for e in st.errors:
            _say(f"  - {e}")
        _say("Fix it first (PoseAssess, OpenPose settings), or give another folder with --src.")
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
    if zip_path is not None and _same_or_inside(src, bundle):
        _say(f"{src} is inside the PoseAssess folder, so the zip would contain the whole "
             "OpenPose install. Move it out (e.g. to C:\\openpose) or use --no-zip.")
        return EXIT_FAILED

    # 3. license and questions
    _say()
    _say(LICENSE_NOTE)
    _say()
    if not args.yes and not _confirm(ask, "Press Enter (or y) to continue, n to cancel: ",
                                     default_yes=True):
        _say("Cancelled.")
        return EXIT_CANCELLED
    if (dest.exists() or _is_link(dest)) and not args.yes:
        if not _confirm(ask, f"{dest} already exists. Replace it? [y/N]: ", default_yes=False):
            _say("Cancelled; the existing openpose folder was kept.")
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
    if zip_path.exists():
        _say(f"Replacing the existing {zip_path}")
    _say(f"Zipping {bundle} ...")
    try:
        size = write_zip(bundle, zip_path)
    except PackError as e:
        _say()
        _say(f"[ERROR] {e}")
        return EXIT_FAILED
    _say()
    _say(f"Done: {zip_path} ({_mb(size)})")
    _say(SHARED_DRIVE_SENTENCE)
    return EXIT_OK


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = input) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args, ask)
    except KeyboardInterrupt:
        _say()
        _say("Cancelled.")
        return EXIT_CANCELLED


if __name__ == "__main__":
    sys.exit(main())
