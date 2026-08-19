[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Yolo11Seed42,
    [Parameter(Mandatory = $true)][string]$Yolo11Seed43,
    [Parameter(Mandatory = $true)][string]$YoloXSeed42,
    [Parameter(Mandatory = $true)][string]$YoloXSeed43,
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
$Policy = Join-Path $ProjectRoot "configs\experiments\rpi_bootstrap_paired_2seed_release_v1.yaml"
$Attestation = Join-Path $ProjectRoot "configs\experiments\mixed_commit_rpi_paired_2seed_v1_attestation.json"
$ExpectedPolicyId = "rpi_bootstrap_paired_2seed_release_v1"
$ExpectedPolicySha = "1865539e9b3569dd4942d9d17495a3644e059df70259b555bd7985e7bdf76f27"
$ExpectedBaseSha = "02facd21ef061fc6530c064d4397ab82e36af3e0601cb502d46f7a6ec34f46f5"
$RunSpecs = @(
    [PSCustomObject]@{ Path = [System.IO.Path]::GetFullPath($Yolo11Seed42); Model = "yolo11m"; Seed = 42 },
    [PSCustomObject]@{ Path = [System.IO.Path]::GetFullPath($Yolo11Seed43); Model = "yolo11m"; Seed = 43 },
    [PSCustomObject]@{ Path = [System.IO.Path]::GetFullPath($YoloXSeed42); Model = "yoloxs"; Seed = 42 },
    [PSCustomObject]@{ Path = [System.IO.Path]::GetFullPath($YoloXSeed43); Model = "yoloxs"; Seed = 43 }
)
$Runs = @($RunSpecs | ForEach-Object { $_.Path })

foreach ($Spec in $RunSpecs) {
    $ManifestPath = Join-Path $Spec.Path "run_manifest.json"
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Run manifest is missing: $($Spec.Path)"
    }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $NormalizedModel = ([string]$Manifest.model).ToLowerInvariant() -replace "[^a-z0-9]", ""
    $SeedHasIntegerType = ($Manifest.protocol.seed -is [int]) -or ($Manifest.protocol.seed -is [long])
    if (
        $NormalizedModel -cne $Spec.Model -or
        -not $SeedHasIntegerType -or
        $Manifest.protocol.seed -ne $Spec.Seed
    ) {
        throw (
            "Run parameter does not match its exact model/seed slot: " +
            "path=$($Spec.Path), expected=$($Spec.Model)/$($Spec.Seed), " +
            "actual=$NormalizedModel/$($Manifest.protocol.seed)"
        )
    }
}
foreach ($Required in @($CompareExecutable, $Policy, $Attestation)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required comparison input is missing: $Required"
    }
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
$Arguments = @("--runs") + $Runs + @(
    "--output-dir",
    $OutputPath,
    "--provenance-attestation",
    $Attestation,
    "--formal-release-policy",
    $Policy,
    "--formal"
)
Push-Location $ProjectRoot
try {
    & $CompareExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Formal paired 2-seed comparison failed with exit code $LASTEXITCODE"
    }
    $CompatibilityPath = Join-Path $OutputPath "protocol_compatibility.json"
    $FormalValidationPath = Join-Path $OutputPath "formal_validation.json"
    foreach ($RequiredOutput in @($CompatibilityPath, $FormalValidationPath)) {
        if (-not (Test-Path -LiteralPath $RequiredOutput -PathType Leaf)) {
            throw "Comparator did not create required formal output: $RequiredOutput"
        }
    }
    $Compatibility = Get-Content -LiteralPath $CompatibilityPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $CompatibilityRunCountIsInteger = (
        ($Compatibility.run_count -is [int]) -or ($Compatibility.run_count -is [long])
    )
    if (
        $Compatibility.release_ready -isnot [bool] -or
        $Compatibility.release_ready -ne $true -or
        $Compatibility.comparable -isnot [bool] -or
        $Compatibility.comparable -ne $true -or
        @($Compatibility.release_blockers).Count -ne 0 -or
        @($Compatibility.critical_mismatches).Count -ne 0 -or
        -not $CompatibilityRunCountIsInteger -or
        $Compatibility.run_count -ne 4 -or
        $Compatibility.formal_release_policy.policy_id -cne $ExpectedPolicyId -or
        $Compatibility.formal_release_policy.policy_sha256 -cne $ExpectedPolicySha -or
        $Compatibility.formal_release_policy.base_protocol_sha256 -cne $ExpectedBaseSha -or
        $Compatibility.formal_release_policy.evidence_tier -cne "paired_2seed_descriptive"
    ) {
        throw "Formal paired 2-seed comparison is BLOCKED; inspect protocol_compatibility.json"
    }
    $FormalValidation = Get-Content -LiteralPath $FormalValidationPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $FormalRunCountIsInteger = (
        ($FormalValidation.run_count -is [int]) -or ($FormalValidation.run_count -is [long])
    )
    $PairedNIsInteger = (
        ($FormalValidation.paired_n -is [int]) -or ($FormalValidation.paired_n -is [long])
    )
    $DfIsInteger = (
        ($FormalValidation.degrees_of_freedom -is [int]) -or
        ($FormalValidation.degrees_of_freedom -is [long])
    )
    if (
        $FormalValidation.status -isnot [string] -or
        $FormalValidation.status -cne "PASS" -or
        -not $FormalRunCountIsInteger -or
        $FormalValidation.run_count -ne 4 -or
        $FormalValidation.policy_id -cne $ExpectedPolicyId -or
        $FormalValidation.policy_sha256 -cne $ExpectedPolicySha -or
        $FormalValidation.base_protocol_sha256 -cne $ExpectedBaseSha -or
        $FormalValidation.evidence_tier -cne "paired_2seed_descriptive" -or
        -not $PairedNIsInteger -or
        $FormalValidation.paired_n -ne 2 -or
        -not $DfIsInteger -or
        $FormalValidation.degrees_of_freedom -ne 1 -or
        $FormalValidation.interpretation -cne "descriptive_only"
    ) {
        throw "Formal validation is not an exact paired 2-seed descriptive PASS"
    }
}
finally {
    Pop-Location
}
