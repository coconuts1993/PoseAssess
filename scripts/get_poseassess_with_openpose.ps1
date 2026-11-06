<#
.SYNOPSIS
  PoseAssess: download -> unzip -> the lab's INTERNAL package with OpenPose BODY_25B.

.DESCRIPTION
  Start it with get_poseassess_with_openpose.bat (double-click; keep the .bat and this
  .ps1 in the same folder). It
    1. finds the downloaded PoseAssess_Portable_win64*.zip in your Downloads folder, or
       opens the GitHub download page (log in to GitHub) and waits for the download;
    2. unzips it (both zip levels) to <Dest>\PoseAssess_Portable
       (default D:\ when there is a local D: drive, else C:\PoseAssess);
    3. runs PoseAssess_Portable\pack_with_openpose.bat --src <OpenPose folder>: it copies
       the BODY_25B parts of OpenPose (e.g. C:\openpose) into PoseAssess_Portable\openpose
       and writes <Dest>\PoseAssess_Portable_with_OpenPose.zip for the lab's shared drive.
  OpenPose license (CMU): internal non-commercial use at your site only.
  NEVER upload or publish the _with_OpenPose zip.
  Nothing is ever deleted except what this script itself made (its temporary zip, or an
  unzip of its own that failed). An existing PoseAssess_Portable is only renamed, and only
  after a new download has been checked.

.EXAMPLE
  get_poseassess_with_openpose.bat
  get_poseassess_with_openpose.bat -Dest E:\ -OpenPose D:\tools\openpose
  get_poseassess_with_openpose.bat -Zip C:\Users\me\Desktop\PoseAssess_Portable_win64.zip
  get_poseassess_with_openpose.bat -OutZip S:\Lab\PoseAssess
  get_poseassess_with_openpose.bat -NoPack        (only download + unzip)
  get_poseassess_with_openpose.bat -Yes           (unattended, see below)
  -Yes: no questions. Takes the default answers, uses an existing PoseAssess_Portable folder
  as it is (renames it if an earlier unzip into it did not finish), stops where a typed
  answer is needed, and passes --yes to pack_with_openpose.bat: with -Yes YOU accept the
  OpenPose license note it shows, and an existing openpose folder in PoseAssess_Portable and
  an existing zip of the same name are replaced.

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
    # The build of 2026-11-05 (GitHub keeps it for 30 days, until 2026-12-05).
    [string]$ArtifactUrl = 'https://github.com/coconuts1993/PoseAssess/actions/runs/42029741982/artifacts/13415702734',
    [string]$BuildsUrl = 'https://github.com/coconuts1993/PoseAssess/actions/workflows/build-portable.yml',
    [int]$WaitMinutes = 120
)

$ErrorActionPreference = 'Stop'
$ZipPattern = 'PoseAssess_Portable_win64*.zip'
$DefaultName = 'PoseAssess_Portable'
$OutSuffix = '_with_OpenPose.zip'
$Marker = '.poseassess_unzip_incomplete'
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

function Get-FreeBytes([string]$Path) {
    try {
        $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path))
        $drive = New-Object System.IO.DriveInfo $root
        if ($drive.IsReady) { return [long]$drive.AvailableFreeSpace }
    } catch { }
    return [long]-1
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

function Test-OpenPoseDir([string]$Dir) {
    if (-not $Dir) { return $false }
    if (-not (Test-Path -LiteralPath $Dir -PathType Container)) { return $false }
    $body25b = PathJoin $Dir 'models\pose\body_25b'
    return ((Test-Path -LiteralPath (PathJoin $Dir 'bin\OpenPoseDemo.exe')) -and
            (Test-Path -LiteralPath (PathJoin $body25b 'pose_deploy.prototxt')) -and
            (Test-Path -LiteralPath (PathJoin $body25b 'pose_iter_XXXXXX.caffemodel')))
}

function Show-OpenPoseProblem([string]$Dir) {
    Say "  Not a usable OpenPose folder: $Dir"
    Say '  It needs:  bin\OpenPoseDemo.exe'
    Say '             models\pose\body_25b\pose_deploy.prototxt'
    Say '             models\pose\body_25b\pose_iter_XXXXXX.caffemodel'
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

# What a zip is: the PoseAssess package itself (a top folder with run.bat), the GitHub
# download around it (one .zip inside), or something else.
function Get-ZipKind([string]$Path) {
    $z = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $innerZips = @()
        $rootRunBat = $false
        foreach ($e in $z.Entries) {
            $n = $e.FullName.Replace('\', '/')
            if ($n -match '^([^/]+)/run\.bat$') {
                return @{ Kind = 'package'; Top = $Matches[1] }
            }
            if ($n -eq 'run.bat') { $rootRunBat = $true }
            if ($n -notmatch '/' -and $n.ToLower().EndsWith('.zip')) { $innerZips += $n }
        }
        if ($rootRunBat) { return @{ Kind = 'package'; Top = '' } }
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
            $msg = '  still waiting ({0:mm\:ss})' -f ((Get-Date) - $start)
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
        $OpenPoseDir = $OpenPose.Trim().Trim('"').TrimEnd('\')
    } else {
        foreach ($c in @('C:\openpose', 'D:\openpose', (PathJoin $env:USERPROFILE 'openpose'))) {
            if (Test-OpenPoseDir $c) { $OpenPoseDir = $c; break }
        }
    }
    while ($true) {
        $problem = ''
        if (-not $OpenPoseDir) {
            Say '  OpenPose not found in C:\openpose, D:\openpose or your user folder.'
        } elseif (-not (Test-OpenPoseDir $OpenPoseDir)) {
            Show-OpenPoseProblem $OpenPoseDir
        } else {
            $problem = Get-PathProblem $OpenPoseDir $false
            if (-not $problem) { break }
            Say "  Cannot use $OpenPoseDir : $problem. Move OpenPose to e.g. C:\openpose."
        }
        $OpenPoseDir = (Read-Answer 'OpenPose folder (e.g. C:\openpose), or press Enter to quit').TrimEnd('\')
        if (-not $OpenPoseDir) { Stop-Script 'No OpenPose folder - nothing done.' }
    }
    Say "  OpenPose: $OpenPoseDir  (BODY_25B files present)"
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
    try { $DestRoot = [System.IO.Path]::GetFullPath($DestRoot) } catch { $problem = 'not a valid folder path' }
    if (-not $problem) { $problem = Get-PathProblem $DestRoot $true }
    if (-not $problem) {
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
    $DestRoot = Read-Answer 'Folder to put PoseAssess in (e.g. D:\ or E:\Tools), or press Enter to quit'
    if (-not $DestRoot) { Stop-Script 'Nothing done.' }
    if ($DestRoot -match '^[A-Za-z]:$') { $DestRoot += '\' }
}
$Final = PathJoin $DestRoot $DefaultName

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
    $incomplete = Test-Path -LiteralPath (PathJoin $Final $Marker)
    if ($incomplete) {
        Say "  $Final exists, but an earlier unzip into it did not finish."
        $choices = 'R / Q'
    } else {
        Say "  $Final already exists."
        Say '    [U] use it as it is (no download, no unzip)'
        $choices = 'U / R / Q'
    }
    Say "    [R] rename it to ${DefaultName}_old_<date> (once the new download is checked) and"
    Say '        unzip a fresh copy'
    Say '    [Q] quit'
    if ($Yes) {
        if ($incomplete) { Say '  -Yes: [R]'; $renameOld = $true } else { Say '  -Yes: [U]'; $skipUnzip = $true }
    }
    while (-not $skipUnzip -and -not $renameOld) {
        $a = (Read-Answer $choices).ToUpper()
        if ($a -eq 'U' -and -not $incomplete) { $skipUnzip = $true }
        elseif ($a -eq 'R') { $renameOld = $true }
        elseif ($a -eq 'Q') { Stop-Script 'Nothing changed.' }
    }
}

if (-not $skipUnzip) {
    # 3a. the downloaded zip
    $zipFile = ''
    if ($Zip) {
        $zipFile = $Zip.Trim().Trim('"')
        if (-not (Test-Path -LiteralPath $zipFile -PathType Leaf)) { Stop-Script "No such file: $zipFile" }
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
            if ($Yes) { $typed = '' } else { $typed = Read-Answer 'Otherwise just press Enter to download it now' }
            if ($typed) {
                if (-not (Test-Path -LiteralPath $typed -PathType Leaf)) { Stop-Script "No such file: $typed" }
                $zipFile = $typed
            } else {
                $zipFile = Wait-ForDownload $downloads
            }
        }
    }
    $zipFile = [System.IO.Path]::GetFullPath($zipFile)
    $state = Get-ZipState $zipFile
    if ($state -eq 'broken') {
        Stop-Script "$zipFile is not a readable zip (download incomplete or broken?). Delete it and download again."
    }
    if ($state -eq 'busy') {
        Stop-Script ("$zipFile cannot be opened right now (still downloading, or being scanned?). " +
                     'Wait a minute and run this script again.')
    }

    # 3b-3c. take the package zip out of the GitHub download if needed, check it, then
    # (only now) rename an old folder and unzip. One try/finally: the temporary zip and a
    # failed unzip of our own are always removed.
    $tempInner = ''
    $created = $false
    $ok = $false
    $fail = ''
    try {
        $kind = Get-ZipKind $zipFile
        $packageZip = $zipFile
        if ($kind.Kind -eq 'outer') {
            $tempInner = PathJoin $DestRoot ('_poseassess_download_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.zip')
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
        if ($kind.Kind -ne 'package') { $fail = "$zipFile is not a PoseAssess package (no run.bat inside)." }

        $prefix = ''
        if (-not $fail -and $kind.Top) { $prefix = $kind.Top + '/' }
        if (-not $fail) {
            $longest = $Final.Length + 1 + (Get-LongestName $packageZip $prefix)
            Say "  Longest path after unzipping: $longest characters (Windows limit: $MaxPath)"
            if ($longest -gt $MaxPath) {
                $fail = ("$Final is too deep for Windows path length limits: choose a shorter " +
                         'folder, e.g. D:\ (-Dest D:\).')
            }
        }
        if (-not $fail -and $renameOld) {
            $old = $DefaultName + '_old_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
            try {
                Rename-Item -LiteralPath $Final -NewName $old
                Say "  Renamed the old folder to $(PathJoin $DestRoot $old)"
            } catch {
                $fail = ("Cannot rename $Final (is PoseAssess or a file in it still open? Close it " +
                         "and run this script again): $($_.Exception.Message)")
            }
        }
        if (-not $fail) {
            [void][System.IO.Directory]::CreateDirectory($Final)
            $created = $true
            [System.IO.File]::WriteAllText((PathJoin $Final $Marker), 'Unzip by get_poseassess_with_openpose.ps1 not finished.')
            Say "  Unzipping to $Final  (a few minutes)..."
            Expand-Package $packageZip $prefix $Final
            Remove-Item -LiteralPath (PathJoin $Final $Marker) -Force
            $ok = $true
        }
    } catch {
        $fail = "Unzipping failed: $($_.Exception.Message)"
    } finally {
        if ($tempInner -and (Test-Path -LiteralPath $tempInner)) {
            Remove-Item -LiteralPath $tempInner -Force -ErrorAction SilentlyContinue
        }
        if ($created -and -not $ok -and (Test-Path -LiteralPath $Final)) {
            Say "  Removing the incomplete $Final ..."
            Remove-Item -LiteralPath $Final -Recurse -Force -ErrorAction SilentlyContinue
            if (Test-Path -LiteralPath $Final) {
                Say "  Could not remove all of $Final - delete it by hand (it is marked incomplete)."
            }
        }
    }
    if ($fail) { Stop-Script $fail }
    Say "  Done: $Final"
}

if (Test-Path -LiteralPath (PathJoin $Final $Marker)) {
    Stop-Script "An earlier unzip into $Final did not finish. Run again and choose [R]."
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
    Stop-Script "$packBat is missing: this is an older PoseAssess build. Run again and choose [R] to unzip the current download."
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
