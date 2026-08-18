[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Yolo11Seed42,
    [Parameter(Mandatory = $true)][string]$Yolo11Seed43,
    [Parameter(Mandatory = $true)][string]$YoloXSeed42,
    [Parameter(Mandatory = $true)][string]$YoloXSeed43,
    [Parameter(Mandatory = $true)][string]$Yolo11Seed44,
    [Parameter(Mandatory = $true)][string]$YoloXSeed44,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$CompareExecutable = Join-Path $ProjectRoot ".venv-collect\Scripts\mcu-compare-runs.exe"
$Attestation = Join-Path $ProjectRoot "configs\experiments\mixed_commit_rpi_v2_attestation.json"
$Runs = @(
    $Yolo11Seed42,
    $Yolo11Seed43,
    $YoloXSeed42,
    $YoloXSeed43,
    $Yolo11Seed44,
    $YoloXSeed44
) | ForEach-Object { [System.IO.Path]::GetFullPath($_) }

foreach ($Run in $Runs) {
    if (-not (Test-Path -LiteralPath (Join-Path $Run "run_manifest.json") -PathType Leaf)) {
        throw "Run manifest is missing: $Run"
    }
}
foreach ($Required in @($CompareExecutable, $Attestation)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required comparison input is missing: $Required"
    }
}

$Arguments = @("--runs") + $Runs + @(
    "--output-dir",
    [System.IO.Path]::GetFullPath($OutputDirectory),
    "--provenance-attestation",
    $Attestation
)
Push-Location $ProjectRoot
try {
    & $CompareExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Formal mixed-run comparison failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
