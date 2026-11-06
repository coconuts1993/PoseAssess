"""Static checks of scripts/get_poseassess_with_openpose.{bat,ps1} (download -> unzip -> pack).

They only run on Windows (Windows PowerShell 5.1), so here: plain ASCII (PowerShell 5.1 reads
a .ps1 without BOM in the ANSI code page), CRLF launcher, and names that match the CI build
(artifact / zip / folder names) and the pack tool (zip suffix, options).
"""
from __future__ import annotations

import re
from pathlib import Path

from poseassess.tools import pack_openpose as pk

REPO = Path(__file__).resolve().parents[1]
PS1 = REPO / "scripts" / "get_poseassess_with_openpose.ps1"
BAT = REPO / "scripts" / "get_poseassess_with_openpose.bat"
WORKFLOW = REPO / ".github" / "workflows" / "build-portable.yml"


def _ps_string(text: str, name: str) -> str:
    m = re.search(rf"^\s*\[?[a-z]*\]?\${name}\s*=\s*'([^']*)'", text, re.M | re.I)
    assert m, name
    return m.group(1)


def test_scripts_are_ascii():
    for f in (PS1, BAT):
        assert f.read_bytes().isascii(), f


def test_launcher_has_crlf_and_runs_the_ps1():
    data = BAT.read_bytes()
    assert data.count(b"\n") > 10 and data.count(b"\n") == data.count(b"\r\n")
    assert data.endswith(b"\r\n")
    text = data.decode("ascii")
    assert 'set "PS1=%~dp0get_poseassess_with_openpose.ps1"' in text
    assert 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*' in text
    lines = text.split("\r\n")
    assert "pause" in lines[lines.index('set "RC=%ERRORLEVEL%"'):]


def test_names_match_the_ci_build_and_the_pack_tool():
    text = PS1.read_text(encoding="ascii")
    wf = WORKFLOW.read_text(encoding="utf-8")
    # artifact zip "PoseAssess_Portable_win64.zip" holding the folder "PoseAssess_Portable"
    assert "name: PoseAssess_Portable_win64" in wf
    assert '"$env:GITHUB_WORKSPACE\\PoseAssess_Portable_win64.zip" PoseAssess_Portable' in wf
    assert _ps_string(text, "ZipPattern") == "PoseAssess_Portable_win64*.zip"
    assert _ps_string(text, "DefaultName") == "PoseAssess_Portable"
    assert _ps_string(text, "OutSuffix") == pk.ZIP_SUFFIX
    url = _ps_string(text, "ArtifactUrl")
    assert re.fullmatch(r"https://github\.com/coconuts1993/PoseAssess/actions/runs/\d+/artifacts/\d+",
                        url), url
    # the bundle's launcher and the pack tool options it uses
    assert "'pack_with_openpose.bat'" in wf
    assert "& $packBat --src $OpenPoseDir --zip-path $resultZip" in text
    assert "& $packBat --src $OpenPoseDir --zip-path $resultZip --yes" in text
    opts = {a for act in pk.build_parser()._actions for a in act.option_strings}
    assert {"--src", "--zip-path", "--yes"} <= opts


def test_openpose_check_matches_the_files_the_backend_needs():
    from poseassess.core import openpose_setup as ops

    text = PS1.read_text(encoding="ascii")
    assert "bin\\OpenPoseDemo.exe" in text
    assert f"'{ops.PROTOTXT_NAME}'" in text and f"'{ops.CAFFEMODEL_NAME}'" in text
    assert "models\\pose\\body_25b" in text


def test_no_join_path_and_no_powershell7_only_syntax():
    """Join-Path fails in PowerShell 5.1 for a drive that does not exist (D:\openpose on a PC
    without D:); ternary / null-coalescing operators do not exist in 5.1."""
    code = [ln.split("#", 1)[0] for ln in PS1.read_text(encoding="ascii").splitlines()
            if not ln.lstrip().startswith("#")]
    body = "\n".join(code)
    assert "Join-Path" not in body
    for op in ("??", "?.", "&&", "||"):
        assert op not in body, op
    assert not re.search(r"\s\?\s.+\s:\s", body)  # ternary


def test_launcher_has_no_block_around_the_folder_echo():
    """%~dp0 inside ( ) breaks on a folder name with ) or &."""
    text = BAT.read_bytes().decode("ascii")
    assert 'if exist "%PS1%" goto run' in text and ":run" in text
    assert "(" not in text.split(":run")[0].split("setlocal", 1)[1]
