"""Tests for poseassess.core.openpose_setup (local OpenPose BODY_25B install).

Everything runs on fake install trees under tmp_path and fake openers / nvidia-smi
runs; nothing is downloaded. The official prototxt files are used when
``POSEASSESS_TEST_PROTOTXT_DIR`` points at a folder with ``1_25BBkg.prototxt`` and
``1_25BSuperModel11FullVGG.prototxt`` (this also checks the pinned SHA-256s);
otherwise synthetic prototxt bytes with patched hashes stand in for them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import subprocess
import zipfile
from pathlib import Path

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
    """urlopen-like response that hands out at most `chunk` bytes per read()."""

    def __init__(self, data: bytes, chunk: int | None = None):
        super().__init__(data)
        self._chunk = chunk

    def read(self, n: int = -1) -> bytes:
        if self._chunk is not None and (n is None or n < 0 or n > self._chunk):
            n = self._chunk
        return super().read(n)


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
    # installing the already-installed file again is a no-op
    st = ops.install_caffemodel(root, root / ops.MODEL_SUBDIR / ops.CAFFEMODEL_NAME,
                                variant="1_25BBkg", opener=no_network)
    assert st.ok and st.variant == "1_25BSuperModel11FullVGG"


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


@pytest.mark.parametrize("name", ["/abs/evil.txt", "C:/evil.txt", "openpose\\..\\..\\evil.txt"])
def test_safe_member_rejects(name):
    with pytest.raises(RuntimeError, match="unsafe path"):
        ops._safe_member(name)


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
        ops.download_openpose(tmp_path / "dl", url=URL, size=10, sha256="", opener=opener_for({}))
    assert list((tmp_path / "dl").iterdir()) == []


def test_official_release_constants():
    assert ops.OFFICIAL_ZIP_URL.endswith("/v1.7.0/" + ops.OFFICIAL_ZIP_NAME)
    assert ops.OFFICIAL_ZIP_URL.startswith(
        "https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/")
    assert len(ops.OFFICIAL_ZIP_SHA256) == 64
    assert ops.MODEL_SUBDIR == Path("models", "pose", "body_25b")
    assert set(ops.BODY25B_VARIANTS) == set(VARIANTS)
    for key, v in ops.BODY25B_VARIANTS.items():
        assert v.name == key and f"/{key}/body_25b/pose_deploy.prototxt" in v.prototxt_url


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
