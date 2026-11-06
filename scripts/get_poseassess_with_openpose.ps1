<#
.SYNOPSIS
  PoseAssess: download -> unzip -> the lab's INTERNAL package with OpenPose BODY_25B.

.DESCRIPTION
  Start it with get_poseassess_with_openpose.bat (double-click; keep the .bat and this
  .ps1 in the same folder). It
    1. checks the OpenPose BODY_25B folder on this PC (e.g. C:\openpose);
    2. finds the downloaded PoseAssess_Portable_win64*.zip in your Downloads folder, or
       opens the GitHub download page (log in to GitHub) and waits for the download;
    3. unzips it (both zip levels) to <Dest>\PoseAssess_Portable
       (default D:\ when there is a local D: drive, else C:\PoseAssess);
    4. runs PoseAssess_Portable\pack_with_openpose.bat --src <OpenPose folder>: it copies
       the BODY_25B parts of OpenPose into PoseAssess_Portable\openpose and writes
       <Dest>\PoseAssess_Portable_with_OpenPose.zip for the lab's shared drive.
  OpenPose license (CMU): internal non-commercial use at your site only.
  NEVER upload or publish the _with_OpenPose zip.
  Nothing is deleted except what this script itself made: its temporary zip and unzip
  folder (_poseassess_download_*.zip, _pa_unzip_*; also leftovers of an interrupted
  earlier run) and empty folders it made for a place you then declined. The new copy is
  unzipped next to PoseAssess_Portable and renamed only when complete; an existing
  PoseAssess_Portable is never written into, only renamed (to PoseAssess_Portable_old_<date>,
  after you chose [R] and the new copy is complete).

.EXAMPLE
  get_poseassess_with_openpose.bat
  get_poseassess_with_openpose.bat -Dest E:\ -OpenPose D:\tools\openpose
  get_poseassess_with_openpose.bat -Zip C:\Users\me\Desktop\PoseAssess_Portable_win64.zip
  get_poseassess_with_openpose.bat -OutZip S:\Lab\PoseAssess
  get_poseassess_with_openpose.bat -NoPack        (only download + unzip)
  get_poseassess_with_openpose.bat -Yes           (unattended, see below)
  -Yes: no questions. Takes the default answers, uses an existing PoseAssess_Portable folder
  as it is ([R] if it is an older build without pack_with_openpose.bat), stops where a
  typed answer is needed, and passes --yes to pack_with_openpose.bat: with -Yes YOU accept
  the OpenPose license note it shows, and an existing openpose folder in
  PoseAssess_Portable and an existing zip of the same name are replaced.

  This file must stay plain ASCII (Windows PowerShell 5.1 reads it in the ANSI code page).
#>
[CmdletBinding()]
param(
    [string]$Zip = '',
    [string]$Dest = '',
    [string]$OpenPose = '',
    [string]$OutZip = '',
    [switch]$NoPack,
    [switch]$Yes,
    # The build of 2026-11-06 (GitHub keeps it for 30 days, until 2026-12-06).
    [string]$ArtifactUrl = 'https://github.com/coconuts1993/PoseAssess/actions/runs/42172708068/artifacts/13482078730',
    [string]$BuildsUrl = 'https://github.com/coconuts1993/PoseAssess/actions/workflows/build-portable.yml',
    [int]$WaitMinutes = 120
)

$ErrorActionPreference = 'Stop'
$ZipPattern = 'PoseAssess_Portable_win64*.zip'
$DefaultName = 'PoseAssess_Portable'
$OutSuffix = '_with_OpenPose.zip'
$StagePrefix = '_pa_unzip_'
$TempZipPrefix = '_poseassess_download_'
$LockName = '_poseassess_get.lock'
$NeedFreeGB = 8
$MaxPath = 259

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

# ------------------------------------------------------------------ helpers
function Say([string]$Text = '') { Write-Host $Text }

function Step([string]$Text) {
    Write-Host ''
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

function Stop-Script([string]$Text) {
    Write-Host ''
    Write-Host "[ERROR] $Text" -ForegroundColor Red
    exit 1
}

function Read-Answer([string]$Prompt) {
    if ($Yes) { Stop-Script "An answer is needed here (run without -Yes): $Prompt" }
    $a = [string](Read-Host $Prompt)
    return $a.Trim().Trim('"').Trim()
}

function Confirm-YesNo([string]$Question, [bool]$DefaultYes) {
    if ($DefaultYes) { $hint = '[Y/n]' } else { $hint = '[y/N]' }
    if ($Yes) {
        Say "$Question $hint (-Yes: $(if ($DefaultYes) { 'yes' } else { 'no' }))"
        return $DefaultYes
    }
    while ($true) {
        $a = (Read-Answer "$Question $hint").ToLower()
        if ($a -eq '') { return $DefaultYes }
        if ($a -eq 'y' -or $a -eq 'yes') { return $true }
        if ($a -eq 'n' -or $a -eq 'no') { return $false }
    }
}

# [IO.Path]::Combine, not Join-Path: in PowerShell 5.1 Join-Path fails for a drive that
# does not exist (e.g. D:\openpose on a PC without D:).
function PathJoin([string]$A, [string]$B) { return [System.IO.Path]::Combine($A, $B) }

function Test-Ascii([string]$Text) { return ($Text -notmatch '[^\x20-\x7E]') }

# Why a path cannot be used ('' = fine). cmd.exe (the .bat files) breaks on & ^ % ! < > | ".
function Get-PathProblem([string]$Path, [bool]$IsInstallFolder) {
    if (-not (Test-Ascii $Path)) { return 'the path has non-ASCII characters' }
    if ($Path -match '[&^%!<>|"]') { return 'the path has one of the characters & ^ % ! < > | "' }
    if ($IsInstallFolder -and $Path -match '[ ()]') { return 'the path has spaces or brackets' }
    return ''
}

function Format-Size([long]$Bytes) {
    if ($Bytes -ge 1GB) { return ('{0:N2} GB' -f ($Bytes / 1GB)) }
    return ('{0:N0} MB' -f ($Bytes / 1MB))
}

function Format-Elapsed([TimeSpan]$Span) {
    return ('{0}:{1:00}' -f [int][Math]::Floor($Span.TotalMinutes), $Span.Seconds)
}

function Get-FreeBytes([string]$Path) {
    try {
        $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path))
        $drive = New-Object System.IO.DriveInfo $root
        if ($drive.IsReady) { return [long]$drive.AvailableFreeSpace }
    } catch { }
    return [long]-1
}

# The folders of $Dir that do not exist yet (outermost first).
function Get-MissingDirs([string]$Dir) {
    $missing = @()
    $d = $Dir
    while ($d -and -not (Test-Path -LiteralPath $d)) {
        $missing = @($d) + $missing
        $d = [System.IO.Path]::GetDirectoryName($d)
    }
    return $missing
}

# Removes folders this script made, innermost first, only while they are empty.
function Remove-OwnEmptyDirs($Dirs) {
    $list = @($Dirs)
    [array]::Reverse($list)
    foreach ($d in $list) {
        try { [System.IO.Directory]::Delete($d, $false) } catch { }
    }
}

# Rename with a few retries (a virus scanner may hold a just-written file for a moment).
function Rename-WithRetry([string]$Path, [string]$NewName) {
    for ($i = 1; $true; $i++) {
        try {
            Rename-Item -LiteralPath $Path -NewName $NewName
            return
        } catch {
            if ($i -ge 10) { throw }
            Start-Sleep -Seconds 1
        }
    }
}

function Get-DownloadsDir {
    try {
        $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders'
        $p = (Get-ItemProperty -Path $key -ErrorAction Stop).'{374DE290-123F-4565-9164-39C4925E467B}'
        if ($p) {
            $p = [Environment]::ExpandEnvironmentVariables($p)
            if (Test-Path -LiteralPath $p -PathType Container) { return $p }
        }
    } catch { }
    return (PathJoin $env:USERPROFILE 'Downloads')
}

function Find-PackageZips([string]$Dir) {
    if (-not (Test-Path -LiteralPath $Dir -PathType Container)) { return @() }
    return @(Get-ChildItem -LiteralPath $Dir -Filter $ZipPattern -ErrorAction SilentlyContinue |
        Where-Object { -not $_.PSIsContainer -and $_.Name -like $ZipPattern -and $_.Length -gt 0 } |
        Sort-Object LastWriteTime -Descending)
}

# A typed / dragged / -Zip path: the zip file itself, or a folder holding one ('' = none).
function Resolve-ZipInput([string]$Path) {
    $p = $Path.Trim().Trim('"')
    if (-not $p) { return '' }
    if (Test-Path -LiteralPath $p -PathType Leaf) { return [System.IO.Path]::GetFullPath($p) }
    if (Test-Path -LiteralPath $p -PathType Container) {
        $found = @(Find-PackageZips $p)
        if ($found.Count -gt 0) { return $found[0].FullName }
    }
    return ''
}

# The OpenPose install root for a given path, like PoseAssess's find_root: the root, its
# bin folder, OpenPoseDemo.exe, or a folder holding an "openpose" folder.
function Resolve-OpenPoseRoot([string]$Path) {
    $p = $Path.Trim().Trim('"').TrimEnd('\')
    if (-not $p) { return '' }
    if ($p -match '^[A-Za-z]:$') { $p += '\' }
    if ((Test-Path -LiteralPath $p -PathType Leaf) -and ([System.IO.Path]::GetFileName($p) -eq 'OpenPoseDemo.exe')) {
        $p = [System.IO.Path]::GetDirectoryName($p)
    }
    if (-not (Test-Path -LiteralPath $p -PathType Container)) { return $p }
    if (Test-Path -LiteralPath (PathJoin $p 'bin\OpenPoseDemo.exe')) { return $p }
    if ((Test-Path -LiteralPath (PathJoin $p 'OpenPoseDemo.exe')) -and ([System.IO.Path]::GetFileName($p) -eq 'bin')) {
        return [System.IO.Path]::GetDirectoryName($p)
    }
    $sub = PathJoin $p 'openpose'
    if (Test-Path -LiteralPath (PathJoin $sub 'bin\OpenPoseDemo.exe')) { return $sub }
    return $p
}

# What is wrong with an OpenPose root for BODY_25B (empty = fine); the same checks as
# PoseAssess's check_install, which the pack step runs again.
function Get-OpenPoseProblems([string]$Root) {
    if (-not $Root -or -not (Test-Path -LiteralPath $Root -PathType Container)) { return @('folder not found') }
    $problems = @()
    $bin = PathJoin $Root 'bin'
    if (-not (Test-Path -LiteralPath (PathJoin $bin 'OpenPoseDemo.exe') -PathType Leaf)) {
        $problems += 'bin\OpenPoseDemo.exe is missing'
    } elseif (@(Get-ChildItem -LiteralPath $bin -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue).Count -eq 0) {
        $problems += 'CPU-only OpenPose (no bin\cudart64_*.dll): BODY_25B needs the GPU build'
    }
    $mdir = PathJoin $Root 'models\pose\body_25b'
    if (-not (Test-Path -LiteralPath (PathJoin $mdir 'pose_deploy.prototxt') -PathType Leaf)) {
        $problems += 'models\pose\body_25b\pose_deploy.prototxt is missing'
    }
    $weights = PathJoin $mdir 'pose_iter_XXXXXX.caffemodel'
    if (Test-Path -LiteralPath $weights -PathType Leaf) {
        if ((Get-Item -LiteralPath $weights).Length -eq 0) {
            $problems += 'models\pose\body_25b\pose_iter_XXXXXX.caffemodel is empty'
        }
    } else {
        $others = @(Get-ChildItem -LiteralPath $mdir -Filter 'pose_iter_*.caffemodel' -ErrorAction SilentlyContinue)
        if ($others.Count -gt 0) {
            $problems += "the BODY_25B weights must be named pose_iter_XXXXXX.caffemodel: rename $($others[0].Name)"
        } else {
            $problems += 'models\pose\body_25b\pose_iter_XXXXXX.caffemodel (the BODY_25B weights) is missing'
        }
    }
    return $problems
}

# The OpenPose folder PoseAssess remembers on this PC (1. Project), or ''.
function Get-RememberedOpenPose {
    try {
        $f = PathJoin (PathJoin $env:APPDATA 'PoseAssess') 'openpose.json'
        if (Test-Path -LiteralPath $f -PathType Leaf) {
            $v = (Get-Content -LiteralPath $f -Raw -Encoding UTF8 | ConvertFrom-Json).openpose_dir
            if ($v) { return [string]$v }
        }
    } catch { }
    return ''
}

# 'ok', 'broken' (not a zip / cut off) or 'busy' (cannot be opened now, e.g. still being
# written or scanned by the antivirus).
function Get-ZipState([string]$Path) {
    try {
        $z = [System.IO.Compression.ZipFile]::OpenRead($Path)
        $z.Dispose()
        return 'ok'
    } catch {
        # .NET wraps the cause: InvalidDataException(inner: IOException ...) for a cut-off
        # zip, so look at the whole chain. Busy = sharing / lock violation or access denied.
        $busy = $false
        $ex = $_.Exception
        while ($null -ne $ex) {
            if ($ex -is [System.IO.InvalidDataException]) { return 'broken' }
            if ($ex -is [System.UnauthorizedAccessException]) { $busy = $true }
            if ($ex -is [System.IO.IOException] -and
                ($ex.HResult -eq -2147024864 -or $ex.HResult -eq -2147024863)) { $busy = $true }
            $ex = $ex.InnerException
        }
        if ($busy) { return 'busy' }
        return 'broken'
    }
}

# What a zip is: the PoseAssess package itself (a top folder with run.bat; HasPack: it has
# pack_with_openpose.bat, i.e. it is not an older build), the GitHub download around it
# (one .zip inside), or something else.
function Get-ZipKind([string]$Path) {
    $z = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $innerZips = @()
        $runTops = @()
        $packTops = @()
        $rootRunBat = $false
        $rootPack = $false
        foreach ($e in $z.Entries) {
            $n = $e.FullName.Replace('\', '/')
            if ($n -match '^([^/]+)/run\.bat$') { $runTops += $Matches[1] }
            elseif ($n -match '^([^/]+)/pack_with_openpose\.bat$') { $packTops += $Matches[1] }
            elseif ($n -eq 'run.bat') { $rootRunBat = $true }
            elseif ($n -eq 'pack_with_openpose.bat') { $rootPack = $true }
            elseif ($n -notmatch '/' -and $n.ToLower().EndsWith('.zip')) { $innerZips += $n }
        }
        if ($runTops.Count -gt 0) {
            return @{ Kind = 'package'; Top = $runTops[0]; HasPack = ($packTops -contains $runTops[0]) }
        }
        if ($rootRunBat) { return @{ Kind = 'package'; Top = ''; HasPack = $rootPack } }
        if ($innerZips.Count -eq 1) { return @{ Kind = 'outer'; Inner = $innerZips[0] } }
        return @{ Kind = 'other' }
    } finally {
        $z.Dispose()
    }
}

# Length of the longest entry name under $Prefix (without the prefix).
function Get-LongestName([string]$ZipPath, [string]$Prefix) {
    $z = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $max = 0
        foreach ($e in $z.Entries) {
            $n = $e.FullName.Replace('\', '/')
            if ($Prefix) {
                if (-not $n.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
                $n = $n.Substring($Prefix.Length)
            }
            if ($n.Length -gt $max) { $max = $n.Length }
        }
        return $max
    } finally {
        $z.Dispose()
    }
}

# Unzips the entries under $Prefix (e.g. 'PoseAssess_Portable/') into $Target, with progress.
function Expand-Package([string]$ZipPath, [string]$Prefix, [string]$Target) {
    $targetFull = [System.IO.Path]::GetFullPath($Target).TrimEnd('\') + '\'
    [void][System.IO.Directory]::CreateDirectory($targetFull)
    $z = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $total = $z.Entries.Count
        $i = 0
        $nextReport = 0
        $skipped = 0
        foreach ($e in $z.Entries) {
            $i++
            $name = $e.FullName.Replace('\', '/')
            if ($Prefix) {
                if (-not $name.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $skipped++
                    continue
                }
                $name = $name.Substring($Prefix.Length)
            }
            if ($name -eq '') { continue }
            $out = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($targetFull, $name.Replace('/', '\')))
            if (-not $out.StartsWith($targetFull, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Unsafe path in the zip: $($e.FullName)"
            }
            if ($name.EndsWith('/')) {
                [void][System.IO.Directory]::CreateDirectory($out)
            } else {
                [void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($out))
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $out, $true)
            }
            $pct = [int][Math]::Floor(100.0 * $i / $total)
            if ($pct -ge $nextReport) {
                Say ('  {0,3}%   ({1} of {2} files)' -f $pct, $i, $total)
                $nextReport = $pct + 10
            }
        }
        if ($skipped -gt 0) { Say "  ($skipped entries outside $Prefix were left out)" }
    } finally {
        $z.Dispose()
    }
}

function Get-FileKey($File) { return ('{0}|{1}' -f $File.Length, $File.LastWriteTimeUtc.Ticks) }

# Opens the GitHub download page and waits until a NEW (or changed) PoseAssess zip is
# complete in $Dir.
function Wait-ForDownload([string]$Dir) {
    $before = @{}
    foreach ($f in (Find-PackageZips $Dir)) { $before[$f.FullName] = (Get-FileKey $f) }
    $start = Get-Date
    Say 'Opening the download page in your browser:'
    Say "  $ArtifactUrl"
    Say 'Log in to GitHub if it asks (an account with access to the PoseAssess repository).'
    Say 'The download (about 850 MB) then starts by itself.'
    Say "If the browser asks where to save it, save it in:  $Dir"
    Start-Process $ArtifactUrl
    Say ''
    Say "Waiting for the download (up to $WaitMinutes minutes; Ctrl+C stops this script)..."
    $deadline = $start.AddMinutes($WaitMinutes)
    $lastLen = [long]-1
    $stableSince = $null
    $lastReport = $start
    $sawProgress = $false
    $hinted = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        $new = @(Find-PackageZips $Dir | Where-Object {
            (-not $before.ContainsKey($_.FullName)) -or ($before[$_.FullName] -ne (Get-FileKey $_)) })
        $done = $null
        if ($new.Count -gt 0) {
            $sawProgress = $true
            $f = $new[0]
            $f.Refresh()
            $partial = (Test-Path -LiteralPath ($f.FullName + '.part')) -or
                       (Test-Path -LiteralPath ($f.FullName + '.crdownload'))
            if (-not $partial -and $f.Length -gt 0) {
                if ($f.Length -eq $lastLen) {
                    if ($null -eq $stableSince) { $stableSince = Get-Date }
                    if (((Get-Date) - $stableSince).TotalSeconds -ge 6) {
                        $state = Get-ZipState $f.FullName
                        if ($state -eq 'ok') {
                            $done = $f
                        } elseif ($state -eq 'broken' -and ((Get-Date) - $stableSince).TotalSeconds -ge 60) {
                            Stop-Script ("$($f.FullName) is not a readable zip (download broken?). " +
                                         'Delete it and run this script again.')
                        }
                    }
                } else {
                    $lastLen = $f.Length
                    $stableSince = $null
                }
            }
        }
        if ($null -ne $done) {
            Say "Downloaded: $($done.FullName)  ($(Format-Size $done.Length))"
            return $done.FullName
        }
        if (((Get-Date) - $lastReport).TotalSeconds -ge 20) {
            $lastReport = Get-Date
            $msg = '  still waiting (' + (Format-Elapsed ((Get-Date) - $start)) + ')'
            $busy = @(Get-ChildItem -LiteralPath $Dir -ErrorAction SilentlyContinue |
                Where-Object { -not $_.PSIsContainer -and $_.LastWriteTime -ge $start -and
                               ($_.Extension -eq '.crdownload' -or $_.Extension -eq '.part' -or
                                $_.Extension -eq '.partial') } |
                Sort-Object LastWriteTime -Descending)
            if ($busy.Count -gt 0) {
                $sawProgress = $true
                $msg += ' - ' + (Format-Size $busy[0].Length) + ' so far'
            }
            Say $msg
            if (-not $sawProgress -and -not $hinted -and ((Get-Date) - $start).TotalSeconds -ge 90) {
                $hinted = $true
                Say ''
                Say '  Nothing is downloading yet. In the browser:'
                Say '  - "Not Found" / "404": log in to GitHub with an account that can see the'
                Say '    PoseAssess repository, then open the link above again.'
                Say '  - "expired" or "Gone": GitHub keeps each build 30 days. Open'
                Say "      $BuildsUrl"
                Say '    click the newest run with a green tick and download'
                Say '    PoseAssess_Portable_win64 under "Artifacts" at the bottom of that page.'
                Say "  This script keeps waiting for the zip in $Dir."
                Say '  (Saved it somewhere else? Close this window and run the script again: it'
                Say '   then lets you type or drag in the file.)'
                Say ''
            }
        }
    }
    Stop-Script ("No new $ZipPattern appeared in $Dir within $WaitMinutes minutes. " +
                 'When it is downloaded, run this script again (or use -Zip <file>).')
}

# ------------------------------------------------------------------ main
Say '======================================================================'
Say ' PoseAssess: download -> unzip -> internal package with OpenPose'
Say ' OpenPose license: internal non-commercial use at your site only.'
Say ' NEVER upload or publish the _with_OpenPose zip.'
Say '======================================================================'

# --- 1. OpenPose on this PC (checked first, before the long download) -----------------
$OpenPoseDir = ''
if (-not $NoPack) {
    Step '1/4  OpenPose on this PC'
    if ($OpenPose) {
        $OpenPoseDir = Resolve-OpenPoseRoot $OpenPose
    } else {
        $firstBad = ''
        foreach ($c in @('C:\openpose', 'D:\openpose', (PathJoin $env:USERPROFILE 'openpose'), (Get-RememberedOpenPose))) {
            if (-not $c) { continue }
            $r = Resolve-OpenPoseRoot $c
            if (-not (Test-Path -LiteralPath $r -PathType Container)) { continue }
            # not the openpose copy inside a PoseAssess folder (an earlier internal package)
            if (Test-Path -LiteralPath (PathJoin ([System.IO.Path]::GetDirectoryName($r)) 'run.bat')) { continue }
            if (@(Get-OpenPoseProblems $r).Count -eq 0) { $OpenPoseDir = $r; break }
            if (-not $firstBad) { $firstBad = $r }
        }
        if (-not $OpenPoseDir) { $OpenPoseDir = $firstBad }
    }
    while ($true) {
        if (-not $OpenPoseDir) {
            Say '  OpenPose not found (looked in C:\openpose, D:\openpose, your user folder and the'
            Say '  folder PoseAssess remembers).'
        } else {
            $problems = @(Get-OpenPoseProblems $OpenPoseDir)
            if ($problems.Count -eq 0) {
                $pp = Get-PathProblem $OpenPoseDir $false
                if (-not $pp) { break }
                $problems = @("$pp - move OpenPose to e.g. C:\openpose")
            }
            Say "  Cannot use OpenPose in $OpenPoseDir :"
            foreach ($p in $problems) { Say "    - $p" }
        }
        Say '  (PoseAssess > 1. Project > "Get OpenPose..." can download and set up OpenPose.)'
        $typed = Read-Answer 'OpenPose folder (e.g. C:\openpose), or press Enter to quit'
        if (-not $typed) { Stop-Script 'No OpenPose folder - nothing done.' }
        $OpenPoseDir = Resolve-OpenPoseRoot $typed
    }
    Say "  OpenPose: $OpenPoseDir  (GPU build, BODY_25B files present)"
}

# --- 2. Where PoseAssess goes --------------------------------------------------------
Step '2/4  Where to put PoseAssess'
if ($Dest) {
    $DestRoot = $Dest.Trim().Trim('"')
    if ($DestRoot -match '^[A-Za-z]:$') { $DestRoot += '\' }
} else {
    $DestRoot = 'C:\PoseAssess'
    try {
        $d = New-Object System.IO.DriveInfo 'D'
        if ($d.IsReady -and $d.DriveType -eq [System.IO.DriveType]::Fixed -and
            $d.AvailableFreeSpace -gt ($NeedFreeGB * 1GB)) { $DestRoot = 'D:\' }
    } catch { }
}
while ($true) {
    $problem = ''
    $madeDirs = @()
    try { $DestRoot = [System.IO.Path]::GetFullPath($DestRoot) } catch { $problem = 'not a valid folder path' }
    if (-not $problem) { $problem = Get-PathProblem $DestRoot $true }
    if (-not $problem) {
        $madeDirs = @(Get-MissingDirs $DestRoot)
        try {
            [void][System.IO.Directory]::CreateDirectory($DestRoot)
            $probe = PathJoin $DestRoot ('_poseassess_write_test_' + [guid]::NewGuid().ToString('N') + '.tmp')
            [System.IO.File]::WriteAllText($probe, 'x')
            Remove-Item -LiteralPath $probe -Force
        } catch {
            $problem = 'cannot write there (' + $_.Exception.Message + ')'
        }
    }
    if (-not $problem) {
        $Final = PathJoin $DestRoot $DefaultName
        Say "  PoseAssess folder:  $Final"
        if (-not $NoPack) { Say "  Lab package (zip):  $(PathJoin $DestRoot ($DefaultName + $OutSuffix))" }
        $free = Get-FreeBytes $DestRoot
        if ($free -ge 0) { Say "  Free space there:   $(Format-Size $free)  (about $NeedFreeGB GB needed)" }
        if ($free -ge 0 -and $free -lt ($NeedFreeGB * 1GB)) {
            Write-Host '  WARNING: probably not enough free space.' -ForegroundColor Yellow
        }
        if (Confirm-YesNo 'Use this place?' $true) { break }
    } else {
        Say "  Cannot use $DestRoot : $problem"
    }
    Remove-OwnEmptyDirs $madeDirs
    $DestRoot = Read-Answer 'Folder to put PoseAssess in (e.g. D:\ or E:\Tools), or press Enter to quit'
    if (-not $DestRoot) { Stop-Script 'Nothing done.' }
    if ($DestRoot -match '^[A-Za-z]:$') { $DestRoot += '\' }
}
$Final = PathJoin $DestRoot $DefaultName

# One copy of this script per place at a time (the lock file goes away when it ends).
$lockPath = PathJoin $DestRoot $LockName
try {
    $Lock = New-Object -TypeName System.IO.FileStream -ArgumentList @(
        $lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None, 4096, [System.IO.FileOptions]::DeleteOnClose)
} catch {
    Stop-Script "Another copy of this script is working in $DestRoot. Let it finish (or close it) and run this again."
}

# Leftovers of an interrupted earlier run (this script's own temporary names only).
foreach ($l in @(Get-ChildItem -LiteralPath $DestRoot -Force -ErrorAction SilentlyContinue | Where-Object {
        ($_.PSIsContainer -and $_.Name -like ($StagePrefix + '*')) -or
        (-not $_.PSIsContainer -and $_.Name -like ($TempZipPrefix + '*.zip')) })) {
    Say "  Removing a leftover of an interrupted earlier run: $($l.FullName)"
    Remove-Item -LiteralPath $l.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

# Where the lab package goes (checked now, not after the long unzip).
$resultZip = PathJoin $DestRoot ($DefaultName + $OutSuffix)
if ($OutZip -and -not $NoPack) {
    $o = $OutZip.Trim().Trim('"').TrimEnd('\')
    $problem = ''
    try {
        if ($o -match '^[A-Za-z]:$') { $o += '\' }
        if ($o.ToLower().EndsWith('.zip')) { $resultZip = [System.IO.Path]::GetFullPath($o) }
        else { $resultZip = PathJoin ([System.IO.Path]::GetFullPath($o)) ($DefaultName + $OutSuffix) }
    } catch {
        $problem = 'not a valid path'
    }
    if (-not $problem) { $problem = Get-PathProblem $resultZip $false }
    if (-not $problem) {
        $outDir = [System.IO.Path]::GetDirectoryName($resultZip)
        if (-not (Test-Path -LiteralPath $outDir -PathType Container)) { $problem = "the folder $outDir does not exist" }
    }
    if ($problem) { Stop-Script "Cannot use -OutZip $OutZip : $problem." }
    Say "  Lab package goes to: $resultZip  (-OutZip)"
}

# --- 3. Download + unzip (unless the folder is there already) ------------------------
Step '3/4  Download and unzip PoseAssess'
$skipUnzip = $false
$renameOld = $false
if (Test-Path -LiteralPath $Final) {
    $oldBuild = (-not $NoPack) -and -not (Test-Path -LiteralPath (PathJoin $Final 'pack_with_openpose.bat'))
    if ($oldBuild) {
        Say "  $Final already exists, but it is an older PoseAssess build"
        Say '  (no pack_with_openpose.bat), so it cannot make the OpenPose package.'
        $choices = 'R / Q'
    } else {
        Say "  $Final already exists."
        Say '    [U] use it as it is (no download, no unzip)'
        $choices = 'U / R / Q'
    }
    Say '    [R] unzip a fresh copy; once it is complete, the existing folder is renamed to'
    Say "        ${DefaultName}_old_<date> (nothing in it is changed or deleted)"
    Say '    [Q] quit'
    if ($Yes) {
        if ($oldBuild) { Say '  -Yes: [R]'; $renameOld = $true } else { Say '  -Yes: [U]'; $skipUnzip = $true }
    }
    while (-not $skipUnzip -and -not $renameOld) {
        $a = (Read-Answer $choices).ToUpper()
        if ($a -eq 'U' -and -not $oldBuild) { $skipUnzip = $true }
        elseif ($a -eq 'R') { $renameOld = $true }
        elseif ($a -eq 'Q') { Stop-Script 'Nothing changed.' }
    }
}

if (-not $skipUnzip) {
    # 3a. the downloaded zip
    $zipFile = ''
    if ($Zip) {
        $zipFile = Resolve-ZipInput $Zip
        if (-not $zipFile) { Stop-Script "-Zip $Zip : no such file (and no $ZipPattern in such a folder)." }
    } else {
        $downloads = Get-DownloadsDir
        $found = @(Find-PackageZips $downloads)
        if ($found.Count -gt 0) {
            $f = $found[0]
            Say "  Found in Downloads: $($f.Name)  ($(Format-Size $f.Length), $($f.LastWriteTime.ToString('yyyy-MM-dd HH:mm')))"
            if (Confirm-YesNo '  Use this file?' $true) { $zipFile = $f.FullName }
        }
        if (-not $zipFile) {
            Say ''
            Say 'If you already downloaded PoseAssess_Portable_win64.zip to another folder, type its'
            Say 'full path (or drag the file into this window) and press Enter.'
            while (-not $zipFile) {
                if ($Yes) { $typed = '' } else { $typed = Read-Answer 'Otherwise just press Enter to download it now' }
                if (-not $typed) {
                    $zipFile = Wait-ForDownload $downloads
                } else {
                    $zipFile = Resolve-ZipInput $typed
                    if (-not $zipFile) { Say "  Not found: $typed  (give the .zip file, or a folder that holds $ZipPattern)" }
                }
            }
        }
    }
    $state = Get-ZipState $zipFile
    if ($state -eq 'broken') {
        Stop-Script "$zipFile is not a readable zip (download incomplete or broken?). Delete it and download again."
    }
    if ($state -eq 'busy') {
        Stop-Script ("$zipFile cannot be opened right now (still downloading, or being scanned?). " +
                     'Wait a minute and run this script again.')
    }

    # 3b-3c. Take the package zip out of the GitHub download if needed and check it; unzip
    # into a new folder of our own (_pa_unzip_<time>) next to PoseAssess_Portable; only
    # when that is complete, rename an old folder ([R]) and the new one into place. One
    # try/finally: the temporary zip and an unfinished unzip folder are always removed.
    $tempInner = ''
    $stage = ''
    $stageMade = $false
    $renamedOldTo = ''
    $ok = $false
    $fail = ''
    try {
        $kind = Get-ZipKind $zipFile
        $packageZip = $zipFile
        if ($kind.Kind -eq 'outer') {
            $tempInner = PathJoin $DestRoot ($TempZipPrefix + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.zip')
            Say "  Taking $($kind.Inner) out of the GitHub download..."
            $z = [System.IO.Compression.ZipFile]::OpenRead($zipFile)
            try {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($z.GetEntry($kind.Inner), $tempInner, $true)
            } finally {
                $z.Dispose()
            }
            $packageZip = $tempInner
            $kind = Get-ZipKind $packageZip
        }
        if ($kind.Kind -ne 'package') {
            $fail = "$zipFile is not a PoseAssess package (no run.bat inside)."
        } elseif (-not $NoPack -and -not $kind.HasPack) {
            $fail = ("$zipFile is an older PoseAssess build (no pack_with_openpose.bat inside). " +
                     "Download the current build: run this script again and answer n to 'Use this " +
                     "file?' (or delete that zip), or get it from $BuildsUrl")
        }
        $prefix = ''
        if (-not $fail -and $kind.Top) { $prefix = $kind.Top + '/' }
        if (-not $fail) {
            $stage = PathJoin $DestRoot ($StagePrefix + (Get-Date -Format 'yyyyMMdd_HHmmss'))
            $deepest = [Math]::Max($Final.Length, $stage.Length) + 1 + (Get-LongestName $packageZip $prefix)
            Say "  Longest path after unzipping: $deepest characters (Windows limit: $MaxPath)"
            if ($deepest -gt $MaxPath) {
                $fail = ("$DestRoot is too deep for Windows path length limits: choose a shorter " +
                         'folder, e.g. D:\ (-Dest D:\).')
            } elseif (Test-Path -LiteralPath $stage) {
                $fail = "$stage already exists - run this script again."
            }
        }
        if (-not $fail) {
            [void][System.IO.Directory]::CreateDirectory($stage)
            $stageMade = $true
            Say "  Unzipping to $stage  (a few minutes)..."
            Expand-Package $packageZip $prefix $stage
            if ($renameOld -and (Test-Path -LiteralPath $Final)) {
                $old = $DefaultName + '_old_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
                try {
                    Rename-WithRetry $Final $old
                    $renamedOldTo = PathJoin $DestRoot $old
                    Say "  Renamed the previous folder to $renamedOldTo"
                } catch {
                    $fail = ("Cannot rename $Final (is PoseAssess or a file in it still open? Close " +
                             "it and run this script again): $($_.Exception.Message)")
                }
            }
            if (-not $fail -and (Test-Path -LiteralPath $Final)) {
                $fail = ("$Final appeared while this script was running; nothing in it was changed. " +
                         'Rename or remove it and run this script again.')
            }
            if (-not $fail) {
                Rename-WithRetry $stage $DefaultName
                $ok = $true
            }
        }
    } catch {
        $fail = "Unzipping failed: $($_.Exception.Message)"
    } finally {
        if ($tempInner -and (Test-Path -LiteralPath $tempInner)) {
            Remove-Item -LiteralPath $tempInner -Force -ErrorAction SilentlyContinue
        }
        if ($stageMade -and -not $ok -and (Test-Path -LiteralPath $stage)) {
            Say "  Removing the unfinished $stage ..."
            Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
            if (Test-Path -LiteralPath $stage) {
                Say "  Could not remove all of $stage - the next run of this script removes it."
            }
        }
        if ($renamedOldTo -and -not $ok -and -not (Test-Path -LiteralPath $Final)) {
            try {
                Rename-WithRetry $renamedOldTo $DefaultName
                Say "  Your previous folder is back at $Final"
            } catch {
                Say "  Your previous folder is at $renamedOldTo"
            }
        }
    }
    if ($fail) { Stop-Script $fail }
    Say "  Done: $Final"
}

foreach ($need in @('run.bat', 'python311\python.exe', 'venv\Scripts\python.exe')) {
    if (-not (Test-Path -LiteralPath (PathJoin $Final $need))) {
        Stop-Script "$Final is incomplete: $need is missing. Run again and choose [R] for a fresh copy."
    }
}

if ($NoPack) {
    Step 'Finished (-NoPack: no OpenPose package made)'
    Say "  Start PoseAssess with:  $(PathJoin $Final 'run.bat')"
    exit 0
}

# --- 4. Pack with OpenPose ------------------------------------------------------------
Step '4/4  Make the internal package with OpenPose'
$packBat = PathJoin $Final 'pack_with_openpose.bat'
if (-not (Test-Path -LiteralPath $packBat)) {
    Stop-Script "$packBat is missing: this is an older PoseAssess build. Run again and choose [R]."
}
Say "  Running $packBat"
if ($Yes) {
    Say '  -Yes: the OpenPose license note is accepted for you (--yes).'
    & $packBat --src $OpenPoseDir --zip-path $resultZip --yes
} else {
    Say '  It shows the OpenPose license note and asks before copying: answer it below.'
    Say '  At the end it says "Press any key to continue": press a key to come back here.'
    Say ''
    & $packBat --src $OpenPoseDir --zip-path $resultZip
}
$rc = $LASTEXITCODE

Step 'Result'
if ($rc -ne 0) {
    if ($rc -eq 130) { Stop-Script 'Packing was cancelled.' }
    Stop-Script "Packing failed (exit code $rc). Send a screenshot of the messages above."
}
if (Test-Path -LiteralPath $resultZip -PathType Leaf) {
    $size = (Get-Item -LiteralPath $resultZip).Length
    Say "  Internal package:  $resultZip  ($(Format-Size $size))"
} else {
    Say "  (No zip at $resultZip - see the messages above for where it was written.)"
}
Say "  PoseAssess with OpenPose on this PC:  $Final"
Say ''
Say 'Next:'
Say '  1. Test here: double-click run.bat in that folder > 1. Project > choose'
Say '     "OpenPose - BODY_25B (GPU)" (it should say ready) > Save > 4. Run, a short trial.'
Say '  2. Copy the _with_OpenPose zip to the lab shared drive. NEVER upload it anywhere.'
Say '  3. Other lab PCs (NVIDIA GPU): unzip it to a local disk (e.g. D:\) and run run.bat.'
Say ''
if (Confirm-YesNo 'Start PoseAssess (run.bat) now to test?' $false) {
    Start-Process -FilePath (PathJoin $Final 'run.bat') -WorkingDirectory $Final
}
exit 0
