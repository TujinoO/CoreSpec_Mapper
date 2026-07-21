param(
    [string]$Python = "",
    [string]$Iscc = "",
    [string]$BuildTag = "r7"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $root "tools\packenv\Scripts\python.exe"
}
if ([string]::IsNullOrWhiteSpace($Iscc)) {
    $Iscc = Join-Path $root "tools\InnoSetup\ISCC.exe"
}
$staging = Join-Path $root "release\staging_$BuildTag"
$work = Join-Path $root "build\pyinstaller_v5_3_stable_$BuildTag"
$spec = Join-Path $root "packaging\corespec_mapper_v5_3.spec"
$iss = Join-Path $root "packaging\CoreSpec_Mapper_V5.3.0_Stable.iss"

if (Test-Path -LiteralPath $staging) {
    throw "Staging directory already exists. Per project safety policy, remove it manually before rebuilding: $staging"
}
if (Test-Path -LiteralPath $work) {
    throw "PyInstaller work directory already exists. Remove it manually before rebuilding: $work"
}
if (-not (Test-Path -LiteralPath $Iscc)) {
    throw "Inno Setup compiler not found: $Iscc"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Packaging Python not found: $Python"
}

& $Python -m PyInstaller --noconfirm --distpath $staging --workpath $work $spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

& $Iscc "/DAppSource=$staging\CoreSpecMapperV5_3" $iss
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE"
}

$installer = Join-Path $root "release\CoreSpec_Mapper_V5.3.0_Stable_Setup.exe"
$hash = Get-FileHash -LiteralPath $installer -Algorithm SHA256
"$($hash.Hash)  $([System.IO.Path]::GetFileName($installer))" | Set-Content -LiteralPath (Join-Path $root "release\CoreSpec_Mapper_V5.3.0_Stable_Setup.sha256") -Encoding ascii
$hash
