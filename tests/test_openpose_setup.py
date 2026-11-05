"""Tests for poseassess.core.openpose_setup (local OpenPose BODY_25B install).

Everything runs on fake install trees under tmp_path and fake openers / nvidia-smi
runs; nothing is downloaded. The official prototxt files are used when
``POSEASSESS_TEST_PROTOTXT_DIR`` points at a folder with ``1_25BBkg.prototxt`` and
``1_25BSuperModel11FullVGG.prototxt`` (this also checks the pinned SHA-256s);
otherwise synthetic prototxt bytes with patched hashes stand in for them.
"""
from __future__ import annotations

import builtins
import dataclasses
import hashlib
import io
import os
import subprocess
import types
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from poseassess.core import openpose_setup as ops

VARIANTS = ("1_25BBkg", "1_25BSuperModel11FullVGG")


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def prototxts(monkeypatch) -> dict[str, bytes]:
    """{variant: prototxt bytes}, matching BODY25B_VARIANTS' SHA-256s."""
    folder = os.environ.get("POSEASSESS_TEST_PROTOTXT_DIR", "")
    if folder and all((Path(folder) / f"{v}.prototxt").is_file() for v in VARIANTS):
        return {v: (Path(folder) / f"{v}.prototxt").read_bytes() for v in VARIANTS}
    out = {}
    for v in VARIANTS:
        data = f'name: "{v}"\nlayer {{ name: "image" }}\n'.encode()
        out[v] = data
        monkeypatch.setitem(ops.BODY25B_VARIANTS, v, dataclasses.replace(
            ops.BODY25B_VARIANTS[v], prototxt_sha256=hashlib.sha256(data).hexdigest()))
    return out


@pytest.fixture(autouse=True)
def _config_dir(tmp_path, monkeypatch):
    """Per-test settings folder (never the real %APPDATA%)."""
    monkeypatch.setenv("POSEASSESS_CONFIG_DIR", str(tmp_path / "cfg"))


class FakeResponse(io.BytesIO):
    """urlopen-like response that hands out at most `chunk` bytes per read() / read1()."""

    def __init__(self, data: bytes, chunk: int | None = None):
        super().__init__(data)
        self._chunk = chunk

    def _limit(self, n: int | None) -> int | None:
        if self._chunk is not None and (n is None or n < 0 or n > self._chunk):
            return self._chunk
        return n

    def read(self, n: int = -1) -> bytes:
        return super().read(self._limit(n))

    def read1(self, n: int = -1) -> bytes:
        return super().read1(self._limit(n))


def opener_for(urls: dict[str, bytes], chunk: int | None = None, calls: list | None = None):
    def opener(url, timeout=None):
        if calls is not None:
            calls.append(url)
        if url not in urls:
            raise OSError(f"no fake data for {url}")
        return FakeResponse(urls[url], chunk)
    return opener


def no_network(url, timeout=None):
    raise AssertionError(f"unexpected download of {url}")


def make_install(root: Path, *, gpu: bool = True, prototxt: bytes | None = None,
                 caffemodel: str | None = ops.CAFFEMODEL_NAME, exe_in_bin: bool = True) -> Path:
    """Fake OpenPose install tree; returns the root."""
    bin_dir = root / "bin" if exe_in_bin else root
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / ops.EXE_NAME).write_bytes(b"MZ fake")
    if gpu:
        (bin_dir / "cudart64_110.dll").write_bytes(b"MZ cudart")
    mdir = root / ops.MODEL_SUBDIR
    mdir.mkdir(parents=True, exist_ok=True)
    if prototxt is not None:
        (mdir / ops.PROTOTXT_NAME).write_bytes(prototxt)
    if caffemodel:
        (mdir / caffemodel).write_bytes(b"weights" * 100)
    return root


# --------------------------------------------------------------------------- #
# find_root
# --------------------------------------------------------------------------- #
def test_find_root_accepts_root_bin_exe_and_parent(tmp_path):
    root = make_install(tmp_path / "parent" / "openpose")
    assert ops.find_root(root) == root
    assert ops.find_root(str(root)) == root
    assert ops.find_root(root / "bin") == root
    assert ops.find_root(root / "bin" / ops.EXE_NAME) == root
    assert ops.find_root(tmp_path / "parent") == root
    assert ops.find_root(f'  "{root}" ') == root  # Explorer's "Copy as path"


def test_find_root_exe_in_root_folder(tmp_path):
    root = make_install(tmp_path / "op", exe_in_bin=False)
    assert ops.find_root(root) == root
    assert ops.find_root(root / ops.EXE_NAME) == root


@pytest.mark.parametrize("path", [None, "", "   "])
def test_find_root_empty(path):
    assert ops.find_root(path) is None


def test_find_root_none_without_exe(tmp_path):
    (tmp_path / "openpose" / "bin").mkdir(parents=True)
    assert ops.find_root(tmp_path) is None
    assert ops.find_root(tmp_path / "openpose") is None
    assert ops.find_root(tmp_path / "missing") is None


# --------------------------------------------------------------------------- #
# check_install
# --------------------------------------------------------------------------- #
def test_check_install_gpu_build_ok(tmp_path, prototxts):
    root = make_install(tmp_path / "openpose", prototxt=prototxts["1_25BBkg"])
    st = ops.check_install(root / "bin")
    assert st.ok, st.errors
    assert st.root == root and st.exe == root / "bin" / ops.EXE_NAME
    assert st.errors == [] and st.warnings == []
    assert st.variant == "1_25BBkg"
    assert st.summary() == f"OpenPose ready: {root} (BODY_25B, Option 2)"


def test_check_install_option1_and_crlf_prototxt(tmp_path, prototxts):
    crlf = prototxts["1_25BSuperModel11FullVGG"].replace(b"\n", b"\r\n")
    st = ops.check_install(make_install(tmp_path / "openpose", prototxt=crlf))
    assert st.ok and st.variant == "1_25BSuperModel11FullVGG"
    assert "Option 1" in st.summary()


def test_check_install_cpu_build(tmp_path, prototxts):
    st = ops.check_install(make_install(tmp_path / "openpose", gpu=False,
                                        prototxt=prototxts["1_25BBkg"]))
    assert not st.ok
    assert len(st.errors) == 1 and "CPU" in st.errors[0]
    assert st.summary().startswith("OpenPose not ready: CPU-only build")


def test_check_install_missing_prototxt_and_caffemodel(tmp_path):
    st = ops.check_install(make_install(tmp_path / "openpose", caffemodel=None))
    assert not st.ok and st.variant is None
    assert any(ops.PROTOTXT_NAME in e for e in st.errors)
    assert any("missing BODY_25B weights" in e and ops.CAFFEMODEL_NAME in e for e in st.errors)


def test_check_install_misnamed_caffemodel_gets_rename_hint(tmp_path, prototxts):
    st = ops.check_install(make_install(tmp_path / "openpose", prototxt=prototxts["1_25BBkg"],
                                        caffemodel="pose_iter_464000.caffemodel"))
    assert not st.ok
    assert len(st.errors) == 1
    assert "rename" in st.errors[0]
    assert "pose_iter_464000.caffemodel" in st.errors[0]
    assert ops.CAFFEMODEL_NAME in st.errors[0]


def test_check_install_unknown_prototxt_is_a_warning(tmp_path):
    st = ops.check_install(make_install(tmp_path / "openpose", prototxt=b"name: 'mine'\n"))
    assert st.ok and st.variant is None
    assert len(st.warnings) == 1 and "official" in st.warnings[0]
    assert "custom prototxt" in st.summary()


def test_check_install_non_ascii_path_warning(tmp_path, prototxts):
    root = make_install(tmp_path / "Ünterordner" / "openpose", prototxt=prototxts["1_25BBkg"])
    st = ops.check_install(root)
    assert st.ok
    assert any("non-ASCII" in w for w in st.warnings)


def test_check_install_no_exe(tmp_path):
    st = ops.check_install(tmp_path)
    assert not st.ok and st.root is None and st.exe is None
    assert ops.EXE_NAME in st.errors[0]
    assert not ops.check_install(None).ok
    assert ops.check_install("").summary() == "OpenPose not ready: no OpenPose folder set"


# --------------------------------------------------------------------------- #
# per-computer memory
# --------------------------------------------------------------------------- #
def test_remember_and_remembered(tmp_path):
    path = ops.user_settings_path()
    assert path == tmp_path / "cfg" / "openpose.json"
    assert ops.remembered_dir() is None
    ops.remember_dir(tmp_path / "op")
    assert path.is_file()
    assert ops.remembered_dir() == tmp_path / "op"
    ops.remember_dir(str(tmp_path / "op2"))
    assert ops.remembered_dir() == tmp_path / "op2"
    assert not list(path.parent.glob("*.part"))


def test_remembered_dir_tolerates_bad_file_and_keeps_other_keys(tmp_path):
    import json

    path = ops.user_settings_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert ops.remembered_dir() is None
    path.write_text('{"other": 1}', encoding="utf-8")
    ops.remember_dir(tmp_path / "op")
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "other": 1, "openpose_dir": str(tmp_path / "op")}


def test_user_settings_path_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("POSEASSESS_CONFIG_DIR")
    monkeypatch.setattr(ops, "_is_windows", lambda: True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert ops.user_settings_path() == tmp_path / "Roaming" / "PoseAssess" / "openpose.json"
    monkeypatch.setattr(ops, "_is_windows", lambda: False)
    assert ops.user_settings_path() == Path.home() / ".config" / "poseassess" / "openpose.json"


def test_candidate_dirs(monkeypatch, tmp_path):
    bundle = Path(ops.__file__).resolve().parents[2]
    assert (bundle / "poseassess").is_dir()
    monkeypatch.setattr(ops, "_is_windows", lambda: False)
    assert ops.candidate_dirs() == [bundle / "openpose", Path.home() / "openpose"]
    monkeypatch.setattr(ops, "_is_windows", lambda: True)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "me"))
    dirs = [str(p) for p in ops.candidate_dirs()]
    assert dirs == [str(bundle / "openpose"), "C:\\openpose", "D:\\openpose",
                    str(tmp_path / "me" / "openpose")]


# --------------------------------------------------------------------------- #
# resolve_openpose_dir
# --------------------------------------------------------------------------- #
def test_resolve_order(tmp_path, monkeypatch, prototxts):
    good = prototxts["1_25BBkg"]
    project = make_install(tmp_path / "proj_op", prototxt=good)
    remembered = make_install(tmp_path / "rem_op", prototxt=good)
    cand = make_install(tmp_path / "cand" / "openpose", prototxt=good)
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [tmp_path / "nothing", tmp_path / "cand"])
    ops.remember_dir(remembered)

    assert ops.resolve_openpose_dir(project) == (project, "project setting")
    assert ops.resolve_openpose_dir("") == (remembered, "remembered on this computer")
    assert ops.resolve_openpose_dir(tmp_path / "nothing") == (
        remembered, "remembered on this computer")
    ops.remember_dir(tmp_path / "gone")
    assert ops.resolve_openpose_dir(None) == (cand, f"found at {cand}")


def test_resolve_prefers_complete_install_over_partial(tmp_path, monkeypatch, prototxts):
    partial = make_install(tmp_path / "partial", caffemodel=None)
    complete = make_install(tmp_path / "complete", prototxt=prototxts["1_25BBkg"])
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [complete])
    assert ops.resolve_openpose_dir(partial) == (complete, f"found at {complete}")
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [])
    assert ops.resolve_openpose_dir(partial) == (partial, "project setting")


def test_resolve_nothing_found(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "candidate_dirs", lambda: [tmp_path / "c1"])
    root, reason = ops.resolve_openpose_dir(tmp_path / "x")
    assert root is None
    assert ops.EXE_NAME in reason and str(tmp_path / "c1") in reason


# --------------------------------------------------------------------------- #
# download_prototxt / install_caffemodel
# --------------------------------------------------------------------------- #
def _prototxt_urls(prototxts):
    return {ops.BODY25B_VARIANTS[v].prototxt_url: data for v, data in prototxts.items()}


@pytest.mark.parametrize("variant", VARIANTS)
def test_download_prototxt(tmp_path, prototxts, variant):
    root = make_install(tmp_path / "openpose", caffemodel=None)
    dest = ops.download_prototxt(root, variant, opener=opener_for(_prototxt_urls(prototxts)))
    assert dest == root / ops.MODEL_SUBDIR / ops.PROTOTXT_NAME
    assert dest.read_bytes() == prototxts[variant]
    assert ops.prototxt_variant(dest) == variant


def test_download_prototxt_sha_mismatch_writes_nothing(tmp_path, prototxts):
    root = make_install(tmp_path / "openpose", caffemodel=None)
    url = ops.BODY25B_VARIANTS["1_25BBkg"].prototxt_url
    with pytest.raises(RuntimeError, match="SHA-256"):
        ops.download_prototxt(root, "1_25BBkg", opener=opener_for({url: b"tampered"}))
    assert list((root / ops.MODEL_SUBDIR).iterdir()) == []


def test_download_prototxt_network_error_and_unknown_variant(tmp_path):
    root = make_install(tmp_path / "openpose", caffemodel=None)
    with pytest.raises(RuntimeError, match="could not download"):
        ops.download_prototxt(root, "1_25BBkg", opener=opener_for({}))
    with pytest.raises(ValueError):
        ops.download_prototxt(root, "nope", opener=no_network)


def test_install_caffemodel_downloads_default_prototxt(tmp_path, prototxts):
    root = make_install(tmp_path / "openpose", caffemodel=None)
    src = tmp_path / "dl" / "pose_iter_464000.caffemodel"
    src.parent.mkdir()
    src.write_bytes(b"\x00weights" * 1000)
    calls: list = []
    st = ops.install_caffemodel(root / "bin", src,
                                opener=opener_for(_prototxt_urls(prototxts), calls=calls))
    assert st.ok, st.errors
    assert st.variant == "1_25BBkg"
    assert calls == [ops.BODY25B_VARIANTS["1_25BBkg"].prototxt_url]
    mdir = root / ops.MODEL_SUBDIR
    assert (mdir / ops.CAFFEMODEL_NAME).read_bytes() == src.read_bytes()
    assert sorted(p.name for p in mdir.iterdir()) == [ops.PROTOTXT_NAME, ops.CAFFEMODEL_NAME]
    assert src.exists()  # copied, not moved


def test_install_caffemodel_explicit_variant(tmp_path, prototxts):
    root = make_install(tmp_path / "openpose", caffemodel=None)
    src = tmp_path / "w.caffemodel"
    src.write_bytes(b"w" * 10)
    st = ops.install_caffemodel(root, src, variant="1_25BSuperModel11FullVGG",
                                opener=opener_for(_prototxt_urls(prototxts)))
    assert st.ok and st.variant == "1_25BSuperModel11FullVGG"


def test_install_caffemodel_keeps_official_prototxt_and_replaces_weights(tmp_path, prototxts):
    root = make_install(tmp_path / "openpose", prototxt=prototxts["1_25BSuperModel11FullVGG"],
                        caffemodel="pose_iter_464000.caffemodel")
    assert not ops.check_install(root).ok
    src = root / ops.MODEL_SUBDIR / "pose_iter_464000.caffemodel"
    st = ops.install_caffemodel(root, src, opener=no_network)
    assert st.ok and st.variant == "1_25BSuperModel11FullVGG"
    # installing the already-installed file again (same variant) is a no-op
    st = ops.install_caffemodel(root, root / ops.MODEL_SUBDIR / ops.CAFFEMODEL_NAME,
                                variant="1_25BSuperModel11FullVGG", opener=no_network)
    assert st.ok and st.variant == "1_25BSuperModel11FullVGG"


def test_install_caffemodel_explicit_other_variant_switches_prototxt(tmp_path, prototxts):
    """The weights must match the prototxt: an explicit variant is never ignored."""
    root = make_install(tmp_path / "openpose", prototxt=prototxts["1_25BSuperModel11FullVGG"],
                        caffemodel=None)
    src = tmp_path / "w.caffemodel"
    src.write_bytes(b"w" * 10)
    calls: list = []
    st = ops.install_caffemodel(root, src, variant="1_25BBkg",
                                opener=opener_for(_prototxt_urls(prototxts), calls=calls))
    assert st.ok and st.variant == "1_25BBkg"
    assert calls == [ops.BODY25B_VARIANTS["1_25BBkg"].prototxt_url]
    mdir = root / ops.MODEL_SUBDIR
    assert (mdir / ops.PROTOTXT_NAME).read_bytes() == prototxts["1_25BBkg"]
    # an official prototxt can be downloaded again: no backup of it
    assert sorted(p.name for p in mdir.iterdir()) == [ops.PROTOTXT_NAME, ops.CAFFEMODEL_NAME]


def test_install_caffemodel_failed_download_changes_nothing(tmp_path, prototxts):
    root = make_install(tmp_path / "openpose", prototxt=b"custom", caffemodel=None)
    src = tmp_path / "w.caffemodel"
    src.write_bytes(b"w" * 10)
    url = ops.BODY25B_VARIANTS["1_25BBkg"].prototxt_url
    with pytest.raises(RuntimeError, match="SHA-256"):
        ops.install_caffemodel(root, src, opener=opener_for({url: b"tampered"}))
    mdir = root / ops.MODEL_SUBDIR
    assert sorted(p.name for p in mdir.iterdir()) == [ops.PROTOTXT_NAME]  # no .bak
    assert (mdir / ops.PROTOTXT_NAME).read_bytes() == b"custom"


def test_disk_errors_are_runtime_errors(tmp_path, prototxts):
    """A folder that cannot be written (here: body_25b is a file) gives RuntimeError."""
    root = make_install(tmp_path / "openpose", caffemodel=None)
    mdir = root / ops.MODEL_SUBDIR
    mdir.rmdir()
    mdir.write_bytes(b"not a folder")
    opener = opener_for(_prototxt_urls(prototxts))
    with pytest.raises(RuntimeError, match="cannot write"):
        ops.download_prototxt(root, "1_25BBkg", opener=opener)
    src = tmp_path / "w.caffemodel"
    src.write_bytes(b"w" * 10)
    with pytest.raises(RuntimeError, match="cannot write"):
        ops.install_caffemodel(root, src, opener=opener)
    assert mdir.read_bytes() == b"not a folder"


def test_install_caffemodel_copy_error_is_runtime_error(tmp_path, prototxts):
    root = make_install(tmp_path / "openpose", prototxt=prototxts["1_25BBkg"], caffemodel=None)
    mdir = root / ops.MODEL_SUBDIR
    (mdir / ops.CAFFEMODEL_NAME).mkdir()  # in the way of the copy
    src = tmp_path / "w.caffemodel"
    src.write_bytes(b"w" * 10)
    with pytest.raises(RuntimeError, match="could not copy"):
        ops.install_caffemodel(root, src, opener=no_network)
    assert not list(mdir.glob("*.part"))


def test_install_caffemodel_replaces_unknown_prototxt_with_backup(tmp_path, prototxts):
    root = make_install(tmp_path / "openpose", prototxt=b"custom", caffemodel=None)
    src = tmp_path / "w.caffemodel"
    src.write_bytes(b"w" * 10)
    st = ops.install_caffemodel(root, src, opener=opener_for(_prototxt_urls(prototxts)))
    assert st.ok and st.variant == "1_25BBkg" and st.warnings == []
    assert (root / ops.MODEL_SUBDIR / "pose_deploy.prototxt.bak").read_bytes() == b"custom"


def test_install_caffemodel_sha_mismatch_installs_nothing(tmp_path):
    root = make_install(tmp_path / "openpose", caffemodel=None)
    src = tmp_path / "w.caffemodel"
    src.write_bytes(b"w" * 10)
    url = ops.BODY25B_VARIANTS["1_25BBkg"].prototxt_url
    with pytest.raises(RuntimeError, match="SHA-256"):
        ops.install_caffemodel(root, src, opener=opener_for({url: b"tampered"}))
    assert list((root / ops.MODEL_SUBDIR).iterdir()) == []


def test_install_caffemodel_missing_source_or_root(tmp_path):
    root = make_install(tmp_path / "openpose", caffemodel=None)
    with pytest.raises(RuntimeError, match="not a"):
        ops.install_caffemodel(root, tmp_path / "missing.caffemodel", opener=no_network)
    src = tmp_path / "w.caffemodel"
    src.write_bytes(b"w")
    with pytest.raises(RuntimeError, match="not an OpenPose folder"):
        ops.install_caffemodel(tmp_path / "nowhere", src, opener=no_network)
    assert not (tmp_path / "nowhere").exists()


# --------------------------------------------------------------------------- #
# download_openpose
# --------------------------------------------------------------------------- #
URL = "https://example.invalid/releases/openpose-test.zip"


def make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def release_zip() -> bytes:
    return make_zip({
        "openpose/": b"",
        # incompressible, so the zip is several download chunks long
        "openpose/bin/OpenPoseDemo.exe": b"".join(hashlib.sha256(bytes([i])).digest()
                                                   for i in range(200)),
        "openpose/bin/cudart64_110.dll": b"MZ cudart",
        "openpose/models/getModels.bat": b"echo models",
        "openpose/models/pose/body_25/pose_deploy.prototxt": b"body_25",
    })


def _download(dest_parent, data, chunk=None, **kw):
    kw.setdefault("size", len(data))
    kw.setdefault("sha256", hashlib.sha256(data).hexdigest())
    return ops.download_openpose(dest_parent, url=URL,
                                 opener=opener_for({URL: data}, chunk=chunk), **kw)


def test_download_openpose(tmp_path):
    data = release_zip()
    seen: list[tuple[int, int]] = []
    root = _download(tmp_path / "dl", data, chunk=1000,
                     progress=lambda done, total: seen.append((done, total)))
    assert root == tmp_path / "dl" / "openpose"
    assert ops.find_root(root) == root
    assert (root / "models" / "getModels.bat").read_bytes() == b"echo models"
    st = ops.check_install(root)
    assert st.exe == root / "bin" / ops.EXE_NAME
    assert not any("CPU" in e for e in st.errors)  # GPU build; BODY_25B files still missing
    assert len(seen) > 1 and seen[-1] == (len(data), len(data))
    assert all(a[0] < b[0] for a, b in zip(seen, seen[1:]))
    assert sorted(p.name for p in (tmp_path / "dl").iterdir()) == ["openpose"]


def test_download_openpose_flat_archive(tmp_path):
    (tmp_path / "openpose").mkdir()  # an empty target folder is fine
    data = make_zip({"bin/OpenPoseDemo.exe": b"MZ", "bin/cudart64_110.dll": b"MZ"})
    root = _download(tmp_path, data)
    assert root == tmp_path / "openpose"
    assert (root / "bin" / ops.EXE_NAME).read_bytes() == b"MZ"


def test_download_openpose_cancel_removes_partial_file(tmp_path):
    data = release_zip()
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(RuntimeError, match="cancelled"):
        _download(tmp_path / "dl", data, chunk=500, cancelled=cancelled)
    assert len(calls) == 3
    assert list((tmp_path / "dl").iterdir()) == []


def test_download_openpose_rejects_zip_slip(tmp_path):
    data = make_zip({"openpose/bin/OpenPoseDemo.exe": b"MZ", "../evil.txt": b"pwned"})
    with pytest.raises(RuntimeError, match="unsafe path"):
        _download(tmp_path / "dl", data)
    assert not (tmp_path / "evil.txt").exists()
    assert list((tmp_path / "dl").iterdir()) == []


@pytest.mark.parametrize("name", [
    "/abs/evil.txt", "C:/evil.txt", "openpose\\..\\..\\evil.txt", "",
    # a drive in the middle: on Windows Path("D:/dl").joinpath("C:", ...) switches drive
    "openpose/C:/evil.txt", "openpose/C:evil.txt",
    "openpose/bin/x:stream",                     # NTFS alternate data stream
    "openpose/\\\\?\\C:/x", "\\\\server\\share\\x", "//server/share/x",
    "openpose/bin/CON", "openpose/bin/nul.txt", "openpose/bin/COM1.dll",  # devices
    "openpose/bin/evil. ", "openpose/bin/evil.", "openpose/bin/a|b", "openpose/bin/a\x00b",
])
def test_safe_member_rejects(name):
    with pytest.raises(RuntimeError, match="unsafe path"):
        ops._safe_member(name)


@pytest.mark.parametrize("name", [  # names from the official 1.7.0 archive
    "openpose/", "openpose/bin/en-US/", "openpose/bin/OpenPoseDemo.exe",
    "openpose/models/pose/body_25/pose_deploy.prototxt", "openpose/3rdparty/windows/wget/",
    "openpose/bin/python/openpose/Debug/pyopenpose.pdb",
])
def test_safe_member_accepts_official_names(name):
    assert ops._safe_member(name) == PurePosixPath(name)


def test_extract_zip_checks_the_joined_path_too(tmp_path, monkeypatch):
    """Second line of defence: a target outside the folder is refused even if a
    name got past _safe_member."""
    monkeypatch.setattr(ops, "_safe_member", lambda name: PurePosixPath(name))
    data = make_zip({"openpose/bin/OpenPoseDemo.exe": b"MZ", "openpose/../evil.txt": b"pwned"})
    with pytest.raises(RuntimeError, match="unsafe path"):
        _download(tmp_path / "dl", data)
    assert not (tmp_path / "dl" / "evil.txt").exists()
    assert list((tmp_path / "dl").iterdir()) == []


def test_download_openpose_sha_and_size_mismatch(tmp_path):
    data = release_zip()
    with pytest.raises(RuntimeError, match="SHA-256"):
        _download(tmp_path / "a", data, sha256="0" * 64)
    assert list((tmp_path / "a").iterdir()) == []
    with pytest.raises(RuntimeError, match="incomplete"):
        _download(tmp_path / "b", data, size=len(data) + 10)
    assert list((tmp_path / "b").iterdir()) == []
    with pytest.raises(RuntimeError, match="larger"):
        _download(tmp_path / "c", data, size=len(data) - 10)
    assert list((tmp_path / "c").iterdir()) == []


def test_download_openpose_without_exe(tmp_path):
    data = make_zip({"something/readme.txt": b"hi"})
    with pytest.raises(RuntimeError, match=ops.EXE_NAME):
        _download(tmp_path / "dl", data)
    assert list((tmp_path / "dl").iterdir()) == []


def test_download_openpose_refuses_existing_install(tmp_path):
    existing = make_install(tmp_path / "openpose")
    with pytest.raises(RuntimeError, match="already exists"):
        ops.download_openpose(tmp_path, url=URL, opener=no_network)
    assert (existing / "bin" / ops.EXE_NAME).exists()


def test_download_openpose_network_error(tmp_path):
    with pytest.raises(RuntimeError, match="could not download"):
        ops.download_openpose(tmp_path / "dl", url=URL, size=10, sha256="0" * 64,
                              opener=opener_for({}))
    assert list((tmp_path / "dl").iterdir()) == []


@pytest.mark.parametrize("kw", [{"sha256": ""}, {"sha256": None}, {"sha256": "abc"},
                                {"size": 0}])
def test_download_openpose_requires_size_and_sha256(tmp_path, kw):
    with pytest.raises(ValueError):
        ops.download_openpose(tmp_path / "dl", url=URL, opener=no_network, **kw)
    assert not (tmp_path / "dl").exists()


def test_download_openpose_creates_only_folders_in_dest_parent(tmp_path, monkeypatch):
    """A drive root such as C:\\ lets ordinary users create folders, not files."""
    data = release_zip()
    dest = tmp_path / "dl"
    real_open = builtins.open

    def root_like_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in "wax+") and Path(file).parent == dest:
            raise PermissionError(13, "Permission denied", str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", root_like_open)
    root = _download(dest, data)
    assert ops.find_root(root) == root
    assert sorted(p.name for p in dest.iterdir()) == ["openpose"]


def test_download_openpose_local_write_error_is_not_a_network_error(tmp_path):
    dest = tmp_path / "dl"
    dest.mkdir()
    (dest / ops._DOWNLOAD_DIR).write_bytes(b"a file in the way")
    with pytest.raises(RuntimeError, match="cannot write") as err:
        _download(dest, release_zip())
    assert "could not download" not in str(err.value)
    assert (dest / ops._DOWNLOAD_DIR).read_bytes() == b"a file in the way"  # not deleted


def test_download_openpose_reads_what_has_arrived(tmp_path):
    """read1(): on a slow link read(n) would wait for all n bytes (frozen progress / cancel)."""
    data = release_zip()
    sizes: list[int] = []

    class Resp(FakeResponse):
        def read(self, n=-1):
            raise AssertionError("read(n) blocks until n bytes arrived")

        def read1(self, n=-1):
            sizes.append(n)
            return super().read1(n)

    ops.download_openpose(tmp_path, url=URL, size=len(data),
                          sha256=hashlib.sha256(data).hexdigest(),
                          opener=lambda url, timeout=None: Resp(data))
    assert sizes and max(sizes) <= 64 * 1024


def test_download_openpose_response_without_read1(tmp_path):
    data = release_zip()

    class Resp:
        def __init__(self):
            self._buf = io.BytesIO(data)

        def read(self, n=-1):
            return self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._buf.close()

    root = ops.download_openpose(tmp_path, url=URL, size=len(data),
                                 sha256=hashlib.sha256(data).hexdigest(),
                                 opener=lambda url, timeout=None: Resp())
    assert ops.find_root(root) == root


def test_download_openpose_needs_room_for_zip_and_unpacked_files(tmp_path, monkeypatch):
    gb = 10 ** 9
    monkeypatch.setattr(ops.shutil, "disk_usage",
                        lambda path: types.SimpleNamespace(total=100 * gb, used=99 * gb, free=gb))
    assert ops.OFFICIAL_ZIP_SIZE < gb < ops.OFFICIAL_ZIP_SIZE + ops.OFFICIAL_UNPACKED_SIZE
    with pytest.raises(RuntimeError, match="not enough free disk space") as err:
        ops.download_openpose(tmp_path / "dl", opener=no_network)
    assert "1.4 GB needed" in str(err.value)


@pytest.mark.parametrize("sub", [(), ("sub",)])
def test_download_openpose_refuses_the_poseassess_folder(sub):
    """PoseAssess is copied to other PCs; OpenPose must not travel with it."""
    target = ops.bundle_root().joinpath(*sub)
    with pytest.raises(RuntimeError, match="PoseAssess folder"):
        ops.download_openpose(target, opener=no_network)
    assert not (ops.bundle_root() / "openpose").exists()
    assert not (ops.bundle_root() / ops._DOWNLOAD_DIR).exists()
    assert not (ops.bundle_root() / "sub").exists()


def test_official_release_constants():
    """The pinned values the integrity checks rely on (a typo would ship green otherwise)."""
    assert ops.OFFICIAL_ZIP_URL == (
        "https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/"
        "openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip")
    assert ops.OFFICIAL_ZIP_NAME == ops.OFFICIAL_ZIP_URL.rsplit("/", 1)[1]
    assert ops.OFFICIAL_ZIP_SIZE == 437912637
    assert ops.OFFICIAL_ZIP_SHA256 == (
        "82ecf470c81b001d1b34873081c0157821c75a0557d926c7427f688d5cf0b25d")
    assert ops.MODEL_SUBDIR == Path("models", "pose", "body_25b")
    assert ops.CAFFEMODEL_NAME == "pose_iter_XXXXXX.caffemodel"
    assert ops.PROTOTXT_NAME == "pose_deploy.prototxt"
    assert (ops.MODEL_POSE, ops.NET_RESOLUTION) == ("BODY_25B", "-1x480")
    raw = ("https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose_train/"
           "master/experimental_models")
    expected = {
        "1_25BBkg": "6b05f4dc5139b70bb7b48ad90534671edef41f03f387bfd82bf6db1fa9ddb405",
        "1_25BSuperModel11FullVGG":
            "9fe1dc100e6e88be8d2773a15a6bdeef2001ce6e60d3d8b6f3502f692c07ed24",
    }
    assert set(ops.BODY25B_VARIANTS) == set(expected)
    for key, v in ops.BODY25B_VARIANTS.items():
        assert v.name == key
        assert v.prototxt_url == f"{raw}/{key}/body_25b/pose_deploy.prototxt"
        assert v.prototxt_sha256 == expected[key]


# --------------------------------------------------------------------------- #
# GPU check
# --------------------------------------------------------------------------- #
def fake_run(*outputs):
    """subprocess.run stand-in: returns the given (returncode, stdout) in turn;
    an exception instance is raised instead."""
    queue = list(outputs)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        out = queue.pop(0)
        if isinstance(out, BaseException):
            raise out
        code, stdout = out
        return subprocess.CompletedProcess(cmd, code, stdout, "")
    run.calls = calls
    return run


def test_gpu_info_rtx4090_ok(monkeypatch):
    monkeypatch.setattr(ops, "_is_windows", lambda: False)
    run = fake_run((0, "NVIDIA GeForce RTX 4090, 566.36, 8.9\n"))
    info = ops.gpu_info(run=run)
    assert info["name"] == "NVIDIA GeForce RTX 4090"
    assert info["driver_version"] == "566.36" and info["compute_cap"] == "8.9"
    assert len(info["gpus"]) == 1
    assert run.calls[0][0] == "nvidia-smi"
    assert "--query-gpu=name,driver_version,compute_cap" in run.calls[0]
    assert ops.gpu_warnings(info) == []


def test_gpu_rtx5090_not_supported():
    info = ops.gpu_info(run=fake_run((0, "NVIDIA GeForce RTX 5090, 576.02, 12.0\n")))
    warns = ops.gpu_warnings(info)
    assert len(warns) == 1
    assert "RTX 5090" in warns[0] and "12.0" in warns[0] and "RTX 50" in warns[0]


def test_gpu_old_driver_without_compute_cap_field(monkeypatch):
    monkeypatch.setattr(ops, "_is_windows", lambda: False)
    run = fake_run((6, 'Field "compute_cap" is not a valid field to query.\n'),
                   (0, "GeForce GTX 1080, 451.48\n"))
    info = ops.gpu_info(run=run)
    assert "--query-gpu=name,driver_version" in run.calls[1]
    assert info == {"name": "GeForce GTX 1080", "driver_version": "451.48", "compute_cap": None,
                    "gpus": [{"name": "GeForce GTX 1080", "driver_version": "451.48",
                              "compute_cap": None}]}
    warns = ops.gpu_warnings(info)
    assert len(warns) == 1 and "451.48" in warns[0] and "456.38" in warns[0]


def test_gpu_old_compute_capability():
    warns = ops.gpu_warnings(ops.gpu_info(run=fake_run((0, "GeForce GTX 750 Ti, 472.12, 5.0\n"))))
    assert len(warns) == 1 and "too old" in warns[0] and "5.0" in warns[0]


def test_gpu_missing_nvidia_smi(monkeypatch):
    monkeypatch.setattr(ops, "_is_windows", lambda: True)
    run = fake_run(FileNotFoundError(), FileNotFoundError())
    assert ops.gpu_info(run=run) is None
    assert len(run.calls) == 2 and run.calls[1][0].endswith("nvidia-smi.exe")
    warns = ops.gpu_warnings(None)
    assert len(warns) == 1 and "no NVIDIA GPU" in warns[0]


def test_gpu_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(ops, "_is_windows", lambda: False)
    run = fake_run((9, "NVIDIA-SMI has failed\n"), (9, "NVIDIA-SMI has failed\n"))
    assert ops.gpu_info(run=run) is None
    assert len(run.calls) == 2


def test_gpu_two_gpus_one_unsupported(monkeypatch):
    monkeypatch.setattr(ops, "_is_windows", lambda: False)
    info = ops.gpu_info(run=fake_run((0, "NVIDIA GeForce RTX 3060, 576.02, 8.6\n"
                                         "NVIDIA GeForce RTX 5070, 576.02, 12.0\n")))
    assert info["name"] == "NVIDIA GeForce RTX 3060" and len(info["gpus"]) == 2
    warns = ops.gpu_warnings(info)
    assert len(warns) == 1 and "RTX 5070" in warns[0]
