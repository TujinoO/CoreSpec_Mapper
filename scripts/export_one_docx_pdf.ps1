param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Destination
)

$ErrorActionPreference = "Stop"
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$destinationPath = if ([System.IO.Path]::IsPathRooted($Destination)) {
    [System.IO.Path]::GetFullPath($Destination)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Destination))
}
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destinationPath)) | Out-Null
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($sourcePath, $false, $true, $false)
    Write-Output "OPENED"
    $document.ExportAsFixedFormat($destinationPath, 17)
    Write-Output $destinationPath
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
