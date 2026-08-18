[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Yolo11Seed42,
    [Parameter(Mandatory = $true)][string]$Yolo11Seed43,
    [Parameter(Mandatory = $true)][string]$YoloXSeed42,
    [Parameter(Mandatory = $true)][string]$YoloXSeed43,
    [Parameter(Mandatory = $true)][string]$Yolo11Seed44,
    [Parameter(Mandatory = $true)][string]$YoloXSeed44,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$CompareExecutableOverride
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ($CompareExecutableOverride -and $env:MCU_TEST_COMPARE_EXECUTABLE_OVERRIDE -ne "1") {
    throw "CompareExecutableOverride is test-only"
}
$CompareExecutable = if ($CompareExecutableOverride) {
    [System.IO.Path]::GetFullPath($CompareExecutableOverride)
} else {
    Join-Path $ProjectRoot ".venv-collect\Scripts\mcu-compare-runs.exe"
}
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
    $Attestation,
    "--formal"
)
Push-Location $ProjectRoot
try {
    & $CompareExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Formal mixed-run comparison failed with exit code $LASTEXITCODE"
    }
    $CompatibilityPath = Join-Path ([System.IO.Path]::GetFullPath($OutputDirectory)) "protocol_compatibility.json"
    if (-not (Test-Path -LiteralPath $CompatibilityPath -PathType Leaf)) {
        throw "Comparator did not create protocol_compatibility.json"
    }
    $Compatibility = Get-Content -LiteralPath $CompatibilityPath -Raw | ConvertFrom-Json
    if (
        $Compatibility.release_ready -ne $true -or
        $Compatibility.comparable -ne $true -or
        @($Compatibility.release_blockers).Count -ne 0 -or
        @($Compatibility.critical_mismatches).Count -ne 0
    ) {
        throw "Formal comparison is BLOCKED; inspect protocol_compatibility.json"
    }
    $FormalValidationPath = Join-Path ([System.IO.Path]::GetFullPath($OutputDirectory)) "formal_validation.json"
    if (-not (Test-Path -LiteralPath $FormalValidationPath -PathType Leaf)) {
        throw "Comparator did not finalize formal_validation.json"
    }
    $FormalValidation = Get-Content -LiteralPath $FormalValidationPath -Raw | ConvertFrom-Json
    if ($FormalValidation.status -ne "PASS" -or $FormalValidation.run_count -ne 6) {
        throw "Formal comparison validation is not an exact six-run PASS"
    }
}
finally {
    Pop-Location
}
