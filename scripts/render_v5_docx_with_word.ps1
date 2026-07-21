param(
    [string]$DocsDirectory = "docs",
    [string]$OutputDirectory = "validation_runs\docx_render\pdfs"
)

$ErrorActionPreference = "Stop"
$docsRoot = (Resolve-Path -LiteralPath $DocsDirectory).Path
$outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
}
[System.IO.Directory]::CreateDirectory($outputRoot) | Out-Null
$documents = Get-ChildItem -LiteralPath $docsRoot -File |
    Where-Object { $_.Name -like "CoreSpec_Mapper_V5_2_*_2026-07-21.docx" } |
    Sort-Object Name

if ($documents.Count -ne 6) {
    throw "Expected six V5.2 DOCX files, found $($documents.Count)."
}

$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    foreach ($source in $documents) {
        $document = $null
        try {
            $document = $word.Documents.Open($source.FullName, $false, $true)
            foreach ($field in $document.Fields) {
                [void]$field.Update()
            }
            foreach ($toc in $document.TablesOfContents) {
                [void]$toc.Update()
            }
            $pdf = Join-Path $outputRoot ($source.BaseName + ".pdf")
            $document.ExportAsFixedFormat($pdf, 17)
            Write-Output $pdf
        }
        finally {
            if ($null -ne $document) {
                $document.Close(0)
                [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
            }
        }
    }
}
finally {
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
