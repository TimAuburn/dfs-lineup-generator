#Requires -Version 5.1
<#
  Create the two Desktop shortcuts for the local DFS Lineup Generator.

  Desktop comes from [Environment]::GetFolderPath('Desktop'), which follows
  OneDrive Known Folder Move. Nothing in this script hard-codes a user path.

  Re-running overwrites only these shortcuts:
    DFS Lineup Generator.lnk
    Stop DFS Lineup Generator.lnk
#>
$ErrorActionPreference = 'Stop'

if ($env:DFS_SHORTCUT_STA -ne '1' -and [System.Threading.Thread]::CurrentThread.GetApartmentState() -ne 'STA') {
    $env:DFS_SHORTCUT_STA = '1'
    $exe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $exe)) { $exe = 'powershell.exe' }
    & $exe -NoProfile -ExecutionPolicy Bypass -STA -File $PSCommandPath
    exit $LASTEXITCODE
}

$launcherDir = $PSScriptRoot
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $launcherDir '..')).Path
$launchBat = Join-Path $launcherDir 'launch_dfs.bat'
$stopBat = Join-Path $launcherDir 'stop_dfs.bat'

if (-not (Test-Path -LiteralPath $launchBat)) {
    throw "Launcher not found: $launchBat"
}
if (-not (Test-Path -LiteralPath $stopBat)) {
    throw "Stop script not found: $stopBat"
}

$desktop = [Environment]::GetFolderPath('Desktop')
if ([string]::IsNullOrWhiteSpace($desktop)) {
    throw 'Windows did not return a Desktop folder.'
}
if (-not (Test-Path -LiteralPath $desktop)) {
    throw "Desktop folder does not exist: $desktop"
}

$shell = New-Object -ComObject WScript.Shell

function Set-DfsShortcut {
    param(
        [string]$Name,
        [string]$Target,
        [string]$Description,
        [int]$WindowStyle
    )
    $path = Join-Path $desktop $Name
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = $Target
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.WindowStyle = $WindowStyle
    $shortcut.Description = $Description
    $shortcut.Arguments = ''
    $icon = Join-Path $launcherDir 'dfs.ico'
    if (Test-Path -LiteralPath $icon) {
        $shortcut.IconLocation = ($icon + ',0')
    }
    $shortcut.Save()

    $saved = $shell.CreateShortcut($path)
    $savedTarget = ([System.IO.Path]::GetFullPath($saved.TargetPath)).TrimEnd('\')
    $expectedTarget = ([System.IO.Path]::GetFullPath($Target)).TrimEnd('\')
    if ($savedTarget -ine $expectedTarget) {
        throw "Shortcut target was not saved. Expected $expectedTarget but the shortcut has $($saved.TargetPath)"
    }
    $savedWork = ([string]$saved.WorkingDirectory).TrimEnd('\')
    $expectedWork = ($projectRoot).TrimEnd('\')
    if ($savedWork -ine $expectedWork) {
        throw "Shortcut working directory was not saved. Expected $expectedWork but the shortcut has $($saved.WorkingDirectory)"
    }
    if ([int]$saved.WindowStyle -ne $WindowStyle) {
        throw "Shortcut window style was not saved. Expected $WindowStyle but the shortcut has $($saved.WindowStyle)"
    }
    return $path
}

$launchLink = Set-DfsShortcut -Name 'DFS Lineup Generator.lnk' -Target $launchBat -Description 'Start the local DFS Lineup Generator' -WindowStyle 7
$stopLink = Set-DfsShortcut -Name 'Stop DFS Lineup Generator.lnk' -Target $stopBat -Description 'Stop the local DFS Lineup Generator server on port 8501' -WindowStyle 1

Write-Output ("Desktop: " + $desktop)
Write-Output ("Created: " + $launchLink)
Write-Output ("Created: " + $stopLink)
Write-Output ("Project: " + $projectRoot)
Write-Output 'Run minimized is set on DFS Lineup Generator. The stop shortcut opens a normal window so its result stays visible.'
