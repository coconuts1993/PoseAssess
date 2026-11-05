"""Tests for poseassess.tools.pack_openpose (the lab's internal package with OpenPose).

Fake OpenPose installs and a fake portable bundle live under tmp_path; ``bundle_root()`` is
patched to the fake bundle, so nothing is written next to the real checkout. As in
test_openpose_setup, ``POSEASSESS_TEST_PROTOTXT_DIR`` (a folder with the official
``1_25BSuperModel11FullVGG.prototxt``) makes the tests use the official prototxt; otherwise
synthetic bytes with a patched SHA-256 stand in for it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from poseassess.core import openpose_setup as ops
from poseassess.tools import pack_openpose as pk

REPO = Path(__file__).resolve().parents[1]
VARIANT = "1_25BSuperModel11FullVGG"
BUNDLE_NAME = "PoseAssess_Portable"


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Per-test settings folder; no real OpenPose found; no rename retry delays."""
    monkeypatch.setenv("POSEASSESS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [])
    monkeypatch.setattr(pk, "_RENAME_DELAY_S", 0.0)


@pytest.fixture
def prototxt(monkeypatch) -> bytes:
    """Option 1 prototxt bytes matching BODY25B_VARIANTS[VARIANT]."""
    folder = os.environ.get("POSEASSESS_TEST_PROTOTXT_DIR", "")
    path = Path(folder) / f"{VARIANT}.prototxt"
    if folder and path.is_file():
        return path.read_bytes()
    data = f'name: "{VARIANT}"\nlayer {{ name: "image" }}\n'.encode()
    monkeypatch.setitem(ops.BODY25B_VARIANTS, VARIANT, dataclasses.replace(
        ops.BODY25B_VARIANTS[VARIANT], prototxt_sha256=hashlib.sha256(data).hexdigest()))
    return data


def write_files(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def make_bundle(b: Path) -> Path:
    """Minimal portable PoseAssess folder (the files `pk.BUNDLE_MARKERS` asks for)."""
    return write_files(b, {"run.bat": b"@echo off\r\n", "python311/python.exe": b"MZ python",
                           "venv/pyvenv.cfg": b"home = C:\\x\\python311\n"})


#: PoseAssess's own files in the fake bundle (all of them go into the zip).
BUNDLE_OWN = {
    "run.bat": b"@echo off\r\n",
    "pack_with_openpose.bat": b"@echo off\r\n",
    "README_DEPLOY.txt": b"readme",
    "requirements-portable.txt": b"numpy",
    "pip_freeze.txt": b"numpy==1.26.4",
    "python311/python.exe": b"MZ python",
    "python311/Lib/__pycache__/os.cpython-311.pyc": b"pyc",
    "venv/pyvenv.cfg": b"home = C:\\x\\python311\n",
    "venv/Scripts/python.exe": b"MZ venv",
    "venv/Lib/site-packages/pkg/__init__.py": b"",
    "venv/Lib/site-packages/pkg/__pycache__/__init__.cpython-311.pyc": b"pyc",
    "poseassess/__init__.py": b"",
    "models/rtmlib/hub/checkpoints/rtmpose.onnx": b"onnx" * 1000,
    "docs/WII_INTEGRATION.md": b"# Wii",
}
#: Things a lab puts into the PoseAssess folder that must not go into the zip by default.
BUNDLE_OTHER = {
    "poseassess/__pycache__/__init__.cpython-311.pyc": b"pyc",     # dropped like in the CI build
    "poseassess/gui/__pycache__/app.cpython-311.pyc": b"pyc",
    "recordings/20261101_101500_S01/session.json": b'{"subject": "S01"}',  # run_wii.bat takes
    "recordings/20261101_101500_S01/wii.csv": b"t,fz",
    "wheels/hidapi-0.14.0-cp311-cp311-win_amd64.whl": b"PK",
    "subjects.txt": b"S01 Jane Doe",
    "Trial01/videos/cam01.mp4": b"video",
}


@pytest.fixture
def bundle(tmp_path, monkeypatch) -> Path:
    """Fake portable PoseAssess folder with lab data in it; ``bundle_root()`` points at it."""
    b = write_files(tmp_path / "share" / BUNDLE_NAME, {**BUNDLE_OWN, **BUNDLE_OTHER})
    (b / "docs" / "img").mkdir()  # an empty folder of PoseAssess is kept in the zip
    monkeypatch.setattr(ops, "bundle_root", lambda: b)
    return b


def make_openpose(root: Path, prototxt: bytes, *, gpu: bool = True,
                  caffemodel: bool = True) -> Path:
    """Fake OpenPose 1.7.0 GPU release tree (plus the user's extra weights); returns the root."""
    files = {
        "bin/OpenPoseDemo.exe": b"MZ demo",
        "bin/openpose.dll": b"MZ op" * 200,
        "bin/caffe.dll": b"MZ caffe" * 300,
        "bin/cudnn64_8.dll": b"MZ cudnn" * 500,
        "bin/sub/plugin.dll": b"MZ plugin",
        "models/pose/body_25b/pose_deploy.prototxt": prototxt,
        "models/pose/body_25b/pose_deploy.prototxt.bak": b"old prototxt",
        "models/pose/body_25b/pose_iter_440000.caffemodel": b"coco weights" * 100,
        "models/pose/body_25b/pose_iter_584000.caffemodel": b"body25 weights" * 100,
        "models/pose/body_25/pose_deploy.prototxt": b"body_25 prototxt",
        "models/pose/body_25/pose_iter_584000.caffemodel": b"body25 weights" * 100,
        "models/pose/coco/pose_deploy_linevec.prototxt": b"coco prototxt",
        "models/face/pose_deploy.prototxt": b"face prototxt",
        "models/face/pose_iter_116000.caffemodel": b"face weights" * 100,
        "models/face/haarcascade_frontalface_alt.xml": b"<xml/>",
        "models/hand/pose_deploy.prototxt": b"hand prototxt",
        "models/hand/pose_iter_102000.caffemodel": b"hand weights" * 100,
        "models/getBaseModels.bat": b"@echo off\r\n",
        "LICENSE": b"OpenPose ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY",
        "Instructions.txt": b"instructions",
        "Version.txt": b"1.7.0",
        "3rdparty/Versions.txt": b"caffe ...",
        "3rdparty/windows/unzip/LICENSE.txt": b"unzip license",
        "3rdparty/windows/unzip/unzip.exe": b"MZ unzip",
        "3rdparty/windows/wget/COPYING": b"GPL",
        "3rdparty/windows/wget/wget.exe": b"MZ wget",
        "3rdparty/windows/wget/wget.html": b"<html/>",
        "include/openpose/headers.hpp": b"#pragma once",
        "lib/openpose.lib": b"lib",
        "examples/media/video.avi": b"avi" * 100,
        "python/openpose/pyopenpose.pyd": b"MZ pyd",
    }
    if gpu:
        files["bin/cudart64_110.dll"] = b"MZ cudart"
    if caffemodel:
        files["models/pose/body_25b/pose_iter_XXXXXX.caffemodel"] = b"body25b weights" * 1000
    return write_files(root, files)


SLIM = {
    "bin/OpenPoseDemo.exe", "bin/openpose.dll", "bin/caffe.dll", "bin/cudnn64_8.dll",
    "bin/cudart64_110.dll", "bin/sub/plugin.dll",
    "models/pose/body_25b/pose_deploy.prototxt", "models/pose/body_25b/pose_iter_XXXXXX.caffemodel",
    "LICENSE", "Instructions.txt", "Version.txt",
    "3rdparty/Versions.txt", "3rdparty/windows/unzip/LICENSE.txt", "3rdparty/windows/wget/COPYING",
}


def rel_files(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def zip_path_of(bundle: Path) -> Path:
    return bundle.parent / f"{BUNDLE_NAME}_with_OpenPose.zip"


def answers(*replies: str):
    """`ask` that returns `replies` in order (and records the prompts)."""
    it = iter(replies)
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(it)
    ask.prompts = prompts  # type: ignore[attr-defined]
    return ask


def no_questions(prompt: str) -> str:
    raise AssertionError(f"unexpected question: {prompt}")


# --------------------------------------------------------------------------- #
# slim copy
# --------------------------------------------------------------------------- #
def test_slim_copy_contents_and_result_ok(tmp_path, bundle, prototxt, capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    assert pk.main(["--src", str(src), "--yes", "--no-zip"], ask=no_questions) == 0
    dest = bundle / "openpose"
    assert rel_files(dest) == SLIM
    for rel in SLIM:
        assert (dest / rel).read_bytes() == (src / rel).read_bytes()
    st = ops.check_install(dest)
    assert st.ok and st.errors == [] and st.variant == VARIANT
    # the source is untouched, nothing is left over next to the copy
    assert (src / "models/pose/body_25b/pose_iter_440000.caffemodel").is_file()
    assert sorted(p.name for p in bundle.iterdir() if p.name.startswith(".")) == []
    assert not zip_path_of(bundle).exists()
    out = capsys.readouterr().out
    assert f"OpenPose source: {src} (given with --src)" in out
    assert "license" in out.lower() and "single site" in out
    assert "Copied OpenPose to" in out and "total" in out and "MB" in out
    assert "OpenPose ready" in out and "Option 1" in out
    assert "No zip written" in out


def test_plan_excludes_extra_weights_and_keeps_notices(tmp_path, prototxt):
    src = make_openpose(tmp_path / "openpose", prototxt)
    st = ops.check_install(src)
    plan = pk.plan_slim_copy(st.root, st.exe)
    rels = {p.rel.as_posix() for p in plan}
    assert rels == SLIM
    kinds = {p.rel.as_posix(): p.kind for p in plan}
    assert kinds["bin/cudart64_110.dll"] == "program"
    assert kinds["models/pose/body_25b/pose_iter_XXXXXX.caffemodel"] == "model"
    assert kinds["LICENSE"] == kinds["3rdparty/windows/wget/COPYING"] == "notice"
    assert all(p.size == (src / p.rel).stat().st_size for p in plan)


def test_plan_copies_a_licenses_folder_whole(tmp_path, prototxt):
    src = make_openpose(tmp_path / "openpose", prototxt)
    (src / "licenses" / "caffe").mkdir(parents=True)
    (src / "licenses" / "caffe" / "caffe_license").write_bytes(b"BSD")
    (src / "README.md").write_bytes(b"# OpenPose")
    st = ops.check_install(src)
    rels = {p.rel.as_posix() for p in pk.plan_slim_copy(st.root, st.exe)}
    assert {"licenses/caffe/caffe_license", "README.md"} <= rels


# --------------------------------------------------------------------------- #
# zip
# --------------------------------------------------------------------------- #
def test_zip_contains_the_bundle_folder_with_openpose(tmp_path, bundle, prototxt, capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    assert pk.main(["--src", str(src), "--yes"], ask=no_questions) == 0
    zp = zip_path_of(bundle)
    assert zp.is_file()
    assert not zp.with_name(zp.name + ".part").exists()
    with zipfile.ZipFile(zp) as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
        top = f"{BUNDLE_NAME}/"
        assert top in names and all(n.startswith(top) for n in names)
        for rel in SLIM:
            assert f"{top}openpose/{rel}" in names
        for rel in BUNDLE_OWN:  # PoseAssess's own files, with the bytecode of venv / python311
            assert top + rel in names
        assert f"{top}docs/img/" in names
        # left out: lab data (participant recordings, projects, ...) and the app's own caches
        assert {n for n in names if not n.endswith("/")} == (
            {top + rel for rel in BUNDLE_OWN} | {f"{top}openpose/{rel}" for rel in SLIM})
        assert not any(n.startswith(f"{top}recordings") or "S01" in n for n in names)
        assert not any(n.startswith(f"{top}poseassess/") and "__pycache__" in n for n in names)
        assert not any("pose_iter_440000" in n or "pose_iter_584000" in n for n in names)
        assert not any("/models/face/" in n or "/models/hand/" in n for n in names)
        weights = zf.getinfo(f"{top}openpose/models/pose/body_25b/pose_iter_XXXXXX.caffemodel")
        assert weights.compress_type == zipfile.ZIP_STORED
        assert zf.getinfo(f"{top}openpose/bin/caffe.dll").compress_type == zipfile.ZIP_DEFLATED
        assert zf.read(weights) == (src / "models/pose/body_25b/pose_iter_XXXXXX.caffemodel"
                                    ).read_bytes()
    out = capsys.readouterr().out
    left_out = out.split("Left out of the zip")[1].split("Zipping")[0]
    for name in ("recordings", "wheels", "subjects.txt", "Trial01"):
        assert f"  {name}" in left_out
    assert "--include" in left_out and "venv" not in left_out
    assert str(zp) in out
    assert pk.SHARED_DRIVE_SENTENCE in out
    assert pk.SHARED_DRIVE_SENTENCE == ("Put this zip on the lab's shared drive; do not upload it "
                                        "(OpenPose license).")


def test_include_adds_left_out_entries(tmp_path, bundle, prototxt, capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    assert pk.main(["--src", str(src), "--yes", "--include", "recordings",
                    "--include", "subjects.txt"]) == 0
    with zipfile.ZipFile(zip_path_of(bundle)) as zf:
        names = set(zf.namelist())
    top = f"{BUNDLE_NAME}/"
    assert {f"{top}recordings/20261101_101500_S01/session.json", f"{top}subjects.txt"} <= names
    assert not any("wheels" in n or "Trial01" in n for n in names)
    left_out = capsys.readouterr().out.split("Left out of the zip")[1].split("Zipping")[0]
    assert "  recordings" not in left_out and "  wheels" in left_out


def test_include_of_a_missing_entry_is_refused(tmp_path, bundle, prototxt, capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    assert pk.main(["--src", str(src), "--yes", "--include", "recordngs"]) == pk.EXIT_USAGE
    assert not (bundle / "openpose").exists() and not zip_path_of(bundle).exists()
    assert "recordngs" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--bogus"], ["--no-zip", "--zip-path", "x.zip"],
                                  ["--include", "recordings/S01"], ["--include", ".."],
                                  ["--include", "recordings", "--no-zip"]])
def test_usage_errors_exit_64_not_2(bundle, argv):
    with pytest.raises(SystemExit) as e:
        pk.main(argv, ask=no_questions)
    assert e.value.code == pk.EXIT_USAGE == 64 != pk.EXIT_NOT_READY
    assert not (bundle / "openpose").exists()


def test_zip_path_option_file_and_folder(tmp_path, bundle, prototxt):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    name = f"{BUNDLE_NAME}_with_OpenPose.zip"
    target = tmp_path / "out" / "lab.ZIP"
    assert pk.main(["--src", str(src), "--yes", "--zip-path", str(target)]) == 0
    assert target.is_file() and not zip_path_of(bundle).exists()
    folder = tmp_path / "outdir"
    folder.mkdir()
    assert pk.main(["--src", str(src), "--yes", "--zip-path", str(folder)]) == 0
    assert (folder / name).is_file()
    # a new folder: given with a trailing separator, or a name without .zip
    for arg, made in ((str(tmp_path / "share_new") + os.sep, tmp_path / "share_new"),
                      (f'"{tmp_path / "lab_drive" / "PoseAssess"}"',
                       tmp_path / "lab_drive" / "PoseAssess")):
        assert pk.main(["--src", str(src), "--yes", "--zip-path", arg]) == 0
        assert made.is_dir() and zipfile.is_zipfile(made / name)


def test_existing_zip_is_replaced_only_after_yes(tmp_path, bundle, prototxt):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    target = tmp_path / "lab.zip"
    target.write_bytes(b"keep me")
    ask = answers("", "")  # Enter on the replace question means No
    assert pk.main(["--src", str(src), "--zip-path", str(target)], ask=ask) == pk.EXIT_CANCELLED
    assert "Replace" in ask.prompts[1] and str(target) in ask.prompts[1]
    assert target.read_bytes() == b"keep me" and not (bundle / "openpose").exists()

    assert pk.main(["--src", str(src), "--zip-path", str(target)], ask=answers("y", "y")) == 0
    assert zipfile.is_zipfile(target)


def test_zip_path_below_a_file_is_refused_before_copying(tmp_path, bundle, prototxt, capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    (tmp_path / "afile").write_bytes(b"x")
    rc = pk.main(["--src", str(src), "--yes", "--zip-path", str(tmp_path / "afile" / "o.zip")])
    assert rc == pk.EXIT_FAILED
    assert not (bundle / "openpose").exists()
    assert "not a folder" in capsys.readouterr().out


def test_zip_inside_the_bundle_is_refused(tmp_path, bundle, prototxt):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    rc = pk.main(["--src", str(src), "--yes", "--zip-path", str(bundle / "x.zip")])
    assert rc == pk.EXIT_FAILED == 3
    assert not (bundle / "openpose").exists() and not (bundle / "x.zip").exists()


def test_zip_failure_leaves_no_part_file(tmp_path, bundle, prototxt, monkeypatch, capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    real_from_file = zipfile.ZipInfo.from_file
    count = {"n": 0}

    def flaky(*a, **kw):
        count["n"] += 1
        if count["n"] == 5:
            raise OSError("disk full")
        return real_from_file(*a, **kw)
    monkeypatch.setattr(zipfile.ZipInfo, "from_file", staticmethod(flaky))
    assert pk.main(["--src", str(src), "--yes"]) == pk.EXIT_FAILED
    assert list(bundle.parent.glob("*.zip*")) == []
    assert ops.check_install(bundle / "openpose").ok  # the copy itself stays
    assert "disk full" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# source not ready / wrong places
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["no_weights", "cpu", "missing"])
def test_refuses_a_source_that_is_not_ready(tmp_path, bundle, prototxt, kind, capsys):
    src = tmp_path / "C" / "openpose"
    if kind == "no_weights":
        make_openpose(src, prototxt, caffemodel=False)
    elif kind == "cpu":
        make_openpose(src, prototxt, gpu=False)
    assert pk.main(["--src", str(src), "--yes"], ask=no_questions) == pk.EXIT_NOT_READY == 2
    assert not (bundle / "openpose").exists()
    assert not zip_path_of(bundle).exists()
    out = capsys.readouterr().out
    assert "not ready" in out
    assert {"no_weights": "pose_iter_XXXXXX.caffemodel", "cpu": "CPU-only",
            "missing": ops.EXE_NAME}[kind] in out


def test_default_source_is_found(tmp_path, bundle, prototxt, monkeypatch, capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    monkeypatch.setattr(ops, "candidate_dirs",
                        lambda: [ops.bundle_root() / "openpose", tmp_path / "C" / "openpose"])
    assert pk.main(["--yes", "--no-zip"]) == 0
    assert f"OpenPose source: {src} (found automatically)" in capsys.readouterr().out
    assert rel_files(bundle / "openpose") == SLIM


def test_default_source_skips_the_bundles_own_copy(tmp_path, bundle, prototxt, monkeypatch,
                                                   capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    monkeypatch.setattr(ops, "candidate_dirs",
                        lambda: [ops.bundle_root() / "openpose", tmp_path / "C" / "openpose"])
    assert pk.main(["--yes", "--no-zip"]) == 0
    (src / "bin" / "new.dll").write_bytes(b"MZ new")  # the install got an update
    ops.remember_dir(bundle / "openpose")             # e.g. picked in the GUI
    capsys.readouterr()
    assert pk.main(["--yes", "--no-zip"]) == 0
    assert f"OpenPose source: {src} (found automatically)" in capsys.readouterr().out
    assert (bundle / "openpose" / "bin" / "new.dll").is_file()


def test_default_source_only_the_bundles_own_copy(tmp_path, bundle, prototxt, monkeypatch,
                                                  capsys):
    make_openpose(bundle / "openpose", prototxt)
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [ops.bundle_root() / "openpose"])
    assert pk.main(["--yes"]) == pk.EXIT_NOT_READY
    assert "--src" in capsys.readouterr().out
    assert not zip_path_of(bundle).exists()


def test_no_openpose_anywhere(bundle, capsys):
    assert pk.main(["--yes"]) == pk.EXIT_NOT_READY
    out = capsys.readouterr().out
    assert "OpenPose not found" in out and "--src" in out
    assert "1. Project > Get OpenPose..." in out


def test_development_checkout_is_refused(tmp_path, prototxt, monkeypatch, capsys):
    """A clone also has run.bat, but no python311 / venv: nothing is copied or zipped."""
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    repo = write_files(tmp_path / "poseassess-repo", {
        "run.bat": b"@echo off\r\n", "poseassess/__init__.py": b"", "tests/test_x.py": b""})
    monkeypatch.setattr(ops, "bundle_root", lambda: repo)
    assert pk.main(["--src", str(src), "--yes"], ask=no_questions) == pk.EXIT_FAILED
    assert not (repo / "openpose").exists()
    assert list(tmp_path.glob("*.zip")) == []
    out = capsys.readouterr().out
    assert "not the portable PoseAssess folder" in out and "python311" in out


def test_refuses_destination_inside_or_equal_to_source(tmp_path, prototxt, monkeypatch, capsys):
    src = make_openpose(tmp_path / "openpose", prototxt)
    inner = make_bundle(src / BUNDLE_NAME)
    monkeypatch.setattr(ops, "bundle_root", lambda: inner)
    assert pk.main(["--src", str(src), "--yes"]) == pk.EXIT_FAILED
    assert not (inner / "openpose").exists()
    assert "inside" in capsys.readouterr().out

    make_bundle(tmp_path)
    monkeypatch.setattr(ops, "bundle_root", lambda: tmp_path)  # destination = the source
    before = rel_files(src)
    assert pk.main(["--src", str(src / "bin"), "--yes"]) == pk.EXIT_FAILED
    assert rel_files(src) == before


def test_source_inside_the_bundle_is_not_zipped(bundle, prototxt, capsys):
    src = make_openpose(bundle / "models" / "openpose", prototxt)  # models goes into the zip
    assert pk.main(["--src", str(src), "--yes"]) == pk.EXIT_FAILED
    assert not (bundle / "openpose").exists() and not zip_path_of(bundle).exists()
    assert "--no-zip" in capsys.readouterr().out
    assert pk.main(["--src", str(src), "--yes", "--no-zip"]) == 0
    assert rel_files(bundle / "openpose") == SLIM


def test_source_in_a_left_out_folder_of_the_bundle(bundle, prototxt):
    src = make_openpose(bundle / "tools" / "openpose", prototxt)
    assert pk.main(["--src", str(src), "--yes"]) == 0
    with zipfile.ZipFile(zip_path_of(bundle)) as zf:
        assert not any("/tools/" in n for n in zf.namelist())
    assert pk.main(["--src", str(src), "--yes", "--include", "tools"]) == pk.EXIT_FAILED


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
@pytest.mark.parametrize("given", [True, False])
def test_destination_linked_to_the_source_is_explained(tmp_path, bundle, prototxt, monkeypatch,
                                                       capsys, given):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    link = bundle / "openpose"
    try:
        os.symlink(src, link, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create symlinks here")
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [link, src])
    before = rel_files(src)
    argv = ["--src", str(src), "--yes"] if given else ["--yes"]
    assert pk.main(argv, ask=no_questions) == pk.EXIT_FAILED
    assert rel_files(src) == before and link.is_symlink()
    out = capsys.readouterr().out
    assert f"{link} is a link to {src}" in out and "rmdir" in out


# --------------------------------------------------------------------------- #
# questions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply", ["", "y", "Yes"])
def test_license_confirmation_accepts_enter_and_y(tmp_path, bundle, prototxt, reply):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    ask = answers(reply)
    assert pk.main(["--src", str(src), "--no-zip"], ask=ask) == 0
    assert len(ask.prompts) == 1 and "Enter" in ask.prompts[0]
    assert ops.check_install(bundle / "openpose").ok


@pytest.mark.parametrize("reply", ["n", "no", "q"])
def test_license_declined_cancels(tmp_path, bundle, prototxt, reply, capsys):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    assert pk.main(["--src", str(src)], ask=answers(reply)) == pk.EXIT_CANCELLED == 130
    assert not (bundle / "openpose").exists() and not zip_path_of(bundle).exists()
    assert "Cancelled" in capsys.readouterr().out


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_question_without_console_or_ctrl_c_cancels(tmp_path, bundle, prototxt, exc):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)

    def ask(prompt):
        raise exc
    assert pk.main(["--src", str(src)], ask=ask) == pk.EXIT_CANCELLED
    assert not (bundle / "openpose").exists()


def _old_copy(bundle: Path) -> Path:
    old = bundle / "openpose"
    (old / "bin").mkdir(parents=True)
    (old / "bin" / "stale.dll").write_bytes(b"old")
    return old


def test_existing_openpose_replaced_only_after_yes(tmp_path, bundle, prototxt):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    old = _old_copy(bundle)

    ask = answers("", "")  # Enter on the replace question means No
    assert pk.main(["--src", str(src), "--no-zip"], ask=ask) == pk.EXIT_CANCELLED
    assert "Replace" in ask.prompts[1]
    assert rel_files(old) == {"bin/stale.dll"}

    assert pk.main(["--src", str(src), "--no-zip"], ask=answers("y", "n")) == pk.EXIT_CANCELLED
    assert rel_files(old) == {"bin/stale.dll"}

    assert pk.main(["--src", str(src), "--no-zip"], ask=answers("y", "y")) == 0
    assert rel_files(old) == SLIM


def test_existing_openpose_replaced_with_yes_flag(tmp_path, bundle, prototxt):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    old = _old_copy(bundle)
    assert pk.main(["--src", str(src), "--no-zip", "--yes"], ask=no_questions) == 0
    assert rel_files(old) == SLIM
    assert not (bundle / pk.OLD_DIR).exists() and not (bundle / pk.NEW_DIR).exists()


# --------------------------------------------------------------------------- #
# failures in the middle
# --------------------------------------------------------------------------- #
def _fail_copy_after(monkeypatch, n: int, exc: BaseException):
    real = shutil.copy2
    count = {"n": 0}

    def copy2(src, dst, **kw):
        count["n"] += 1
        if count["n"] > n:
            raise exc
        return real(src, dst, **kw)
    monkeypatch.setattr(shutil, "copy2", copy2)


@pytest.mark.parametrize("existing", [False, True])
def test_copy_failure_leaves_no_half_copied_folder(tmp_path, bundle, prototxt, monkeypatch,
                                                   capsys, existing):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    if existing:
        _old_copy(bundle)
    before = sorted(p.name for p in bundle.iterdir())
    _fail_copy_after(monkeypatch, 3, OSError(28, "No space left on device"))
    assert pk.main(["--src", str(src), "--yes"]) == pk.EXIT_FAILED
    assert sorted(p.name for p in bundle.iterdir()) == before
    if existing:
        assert rel_files(bundle / "openpose") == {"bin/stale.dll"}
    else:
        assert not (bundle / "openpose").exists()
    assert not zip_path_of(bundle).exists()
    assert "No space left" in capsys.readouterr().out


def test_ctrl_c_during_copy_leaves_nothing(tmp_path, bundle, prototxt, monkeypatch):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    before = sorted(p.name for p in bundle.iterdir())
    _fail_copy_after(monkeypatch, 2, KeyboardInterrupt())
    assert pk.main(["--src", str(src), "--yes"]) == pk.EXIT_CANCELLED
    assert sorted(p.name for p in bundle.iterdir()) == before


def test_leftovers_of_an_interrupted_run_are_removed(tmp_path, bundle, prototxt):
    """A complete swap whose clean-up was interrupted: .openpose.old is a stale copy."""
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    _old_copy(bundle)
    for name in (pk.NEW_DIR, pk.OLD_DIR):
        (bundle / name / "bin").mkdir(parents=True)
        (bundle / name / "bin" / "half.dll").write_bytes(b"x")
    assert pk.main(["--src", str(src), "--yes"]) == 0
    assert not (bundle / pk.NEW_DIR).exists() and not (bundle / pk.OLD_DIR).exists()
    assert rel_files(bundle / "openpose") == SLIM
    with zipfile.ZipFile(zip_path_of(bundle)) as zf:
        assert not any(".openpose." in n for n in zf.namelist())


def test_stranded_old_copy_is_put_back_not_deleted(tmp_path, bundle, prototxt, capsys):
    """Interrupted between the two renames: the previous copy is only in .openpose.old."""
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    (bundle / pk.OLD_DIR / "bin").mkdir(parents=True)
    (bundle / pk.OLD_DIR / "bin" / "previous.dll").write_bytes(b"old")
    (bundle / pk.NEW_DIR).mkdir()
    ask = answers("", "n")  # license: Enter; replace the (restored) openpose folder: no
    assert pk.main(["--src", str(src), "--no-zip"], ask=ask) == pk.EXIT_CANCELLED
    assert "Replace" in ask.prompts[1]
    assert rel_files(bundle / "openpose") == {"bin/previous.dll"}
    assert not (bundle / pk.OLD_DIR).exists() and not (bundle / pk.NEW_DIR).exists()
    assert "Put back the previous OpenPose copy" in capsys.readouterr().out


def test_rename_is_retried_and_failure_restores_the_old_copy(tmp_path, bundle, prototxt,
                                                             monkeypatch):
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    real = os.rename
    calls = {"n": 0}

    def busy_once(a, b):  # a virus scanner holds the new exe for a moment
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "in use")
        return real(a, b)
    monkeypatch.setattr(os, "rename", busy_once)
    assert pk.main(["--src", str(src), "--yes", "--no-zip"]) == 0
    assert rel_files(bundle / "openpose") == SLIM

    old_files = rel_files(bundle / "openpose")
    (bundle / "openpose" / "bin" / "marker.dll").write_bytes(b"old")

    def new_never_renames(a, b):
        if Path(a).name == pk.NEW_DIR:
            raise PermissionError(13, "in use")
        return real(a, b)
    monkeypatch.setattr(os, "rename", new_never_renames)
    assert pk.main(["--src", str(src), "--yes", "--no-zip"]) == pk.EXIT_FAILED
    assert rel_files(bundle / "openpose") == old_files | {"bin/marker.dll"}
    assert not (bundle / pk.NEW_DIR).exists() and not (bundle / pk.OLD_DIR).exists()


def _installed_with_marker(tmp_path, bundle, prototxt) -> Path:
    src = make_openpose(tmp_path / "C" / "openpose", prototxt)
    assert pk.main(["--src", str(src), "--yes", "--no-zip"]) == 0
    (bundle / "openpose" / "bin" / "marker.dll").write_bytes(b"old")
    return src


def test_ctrl_c_between_the_renames_puts_the_old_copy_back(tmp_path, bundle, prototxt,
                                                           monkeypatch):
    src = _installed_with_marker(tmp_path, bundle, prototxt)
    real = os.rename

    def ctrl_c_on_new(a, b):  # e.g. pressed while the rename of the new copy is retried
        if Path(a).name == pk.NEW_DIR:
            raise KeyboardInterrupt
        return real(a, b)
    monkeypatch.setattr(os, "rename", ctrl_c_on_new)
    assert pk.main(["--src", str(src), "--yes", "--no-zip"]) == pk.EXIT_CANCELLED
    assert (bundle / "openpose" / "bin" / "marker.dll").is_file()
    assert ops.check_install(bundle / "openpose").ok
    assert not (bundle / pk.NEW_DIR).exists() and not (bundle / pk.OLD_DIR).exists()


@pytest.mark.parametrize("exc", [PermissionError(13, "in use"), KeyboardInterrupt()])
def test_failed_put_back_says_where_the_old_copy_is(tmp_path, bundle, prototxt, monkeypatch,
                                                    capsys, exc):
    src = _installed_with_marker(tmp_path, bundle, prototxt)
    real = os.rename
    old = bundle / pk.OLD_DIR

    def stuck(a, b):  # the new copy cannot be renamed, and the old one not back either
        if Path(a).name == pk.NEW_DIR:
            raise exc
        if Path(a).name == pk.OLD_DIR:
            raise PermissionError(13, "in use")
        return real(a, b)
    monkeypatch.setattr(os, "rename", stuck)
    capsys.readouterr()
    rc = pk.main(["--src", str(src), "--yes", "--no-zip"])
    assert rc == (pk.EXIT_CANCELLED if isinstance(exc, KeyboardInterrupt) else pk.EXIT_FAILED)
    assert not (bundle / "openpose").exists() and (old / "bin" / "marker.dll").is_file()
    assert str(old) in capsys.readouterr().out

    # the next run puts it back instead of deleting it as a leftover
    monkeypatch.setattr(os, "rename", real)
    assert pk.main(["--src", str(src), "--no-zip"], ask=answers("", "n")) == pk.EXIT_CANCELLED
    assert (bundle / "openpose" / "bin" / "marker.dll").is_file() and not old.exists()


# --------------------------------------------------------------------------- #
# launcher
# --------------------------------------------------------------------------- #
def test_bat_has_crlf_and_runs_the_module():
    data = (REPO / "pack_with_openpose.bat").read_bytes()
    assert data.isascii()
    assert data.count(b"\n") > 10 and data.count(b"\n") == data.count(b"\r\n")
    assert data.endswith(b"\r\n")
    text = data.decode("ascii")
    assert '"%ROOT%\\venv\\Scripts\\python.exe" -m poseassess.tools.pack_openpose %*' in text
    for line in ('set "PYTHONPATH=%ROOT%"', 'set "PYTHONUTF8=1"', 'set "PYTHONSAFEPATH=1"',
                 '> "%ROOT%\\venv\\pyvenv.cfg" echo home = %PYHOME%',
                 '>> "%ROOT%\\venv\\pyvenv.cfg" echo executable = %PYHOME%\\python.exe'):
        assert line in text
    lines = text.split("\r\n")
    assert "pause" in lines[lines.index('set "RC=%ERRORLEVEL%"'):]


def test_bat_is_copied_into_the_portable_bundle():
    wf = (REPO / ".github" / "workflows" / "build-portable.yml").read_text(encoding="utf-8")
    copy_line = next(ln for ln in wf.splitlines() if "'run_wii.bat'" in ln)
    assert "'pack_with_openpose.bat'" in copy_line


def test_module_help_runs():
    r = subprocess.run([sys.executable, "-m", "poseassess.tools.pack_openpose", "--help"],
                       cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--src" in r.stdout and "--no-zip" in r.stdout and "--yes" in r.stdout
