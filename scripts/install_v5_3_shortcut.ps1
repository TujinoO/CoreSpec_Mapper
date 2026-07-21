param(
    [string]$Pythonw = "",
    [string]$Workspace = "",
    [string]$Desktop = [Environment]::GetFolderPath("Desktop")
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Workspace)) {
    $Workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}
if ([string]::IsNullOrWhiteSpace($Pythonw)) {
    $command = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "pythonw.exe was not found. Pass -Pythonw with the desktop environment path."
    }
    $Pythonw = $command.Source
}
$pythonwPath = (Resolve-Path -LiteralPath $Pythonw).Path
$workspacePath = (Resolve-Path -LiteralPath $Workspace).Path
$iconPath = Join-Path $workspacePath "src\corespec_mapper\resources\corespec_logo.ico"
$shortcutPath = Join-Path $Desktop "CoreSpec Mapper V5.3.lnk"
$oldShortcutPath = Join-Path $Desktop "CoreSpec Mapper V5.2.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonwPath
$shortcut.Arguments = "-m corespec_mapper.desktop_v5"
$shortcut.WorkingDirectory = $workspacePath
$shortcut.IconLocation = "$iconPath,0"
$shortcut.Description = "CoreSpec Mapper V5.3"
$shortcut.WindowStyle = 1
$shortcut.Save()

if (Test-Path -LiteralPath $oldShortcutPath) {
    Remove-Item -LiteralPath $oldShortcutPath
}

Write-Output $shortcutPath
