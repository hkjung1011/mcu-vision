[CmdletBinding()]
param(
    [int[]]$Seeds = @(42, 43, 44),
    [ValidateRange(1, 100000)]
    [int]$Epochs = 100,
    [ValidateRange(1, 100000)]
    [int]$Batch = 8,
    [ValidateRange(32, 16384)]
    [int]$ImageSize = 640,
    [ValidateRange(0, 64)]
    [int]$Workers = 0,
    [string]$ProtocolConfig = "configs/experiments/baseline_v1.yaml",
    [string]$YoloData = "data/processed/micropcb_rpi_phash_v2/dataset.yaml",
    [string]$CocoRoot = "data/processed/micropcb_rpi_phash_v2_coco",
    [string]$DatasetEvidence = "data/evidence/micropcb_rpi_phash_v2/dataset_evidence.json",
    [string]$CampaignId,
    [switch]$Smoke,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$Yolo11Python = Join-Path $ProjectRoot ".venv-yolo11\Scripts\python.exe"
$YoloXPython = Join-Path $ProjectRoot ".venv-yolox\Scripts\python.exe"
$CompareExecutable = Join-Path $ProjectRoot ".venv-collect\Scripts\mcu-compare-runs.exe"
$Yolo11Wrapper = Join-Path $PSScriptRoot "train_yolo11_logged.py"
$YoloXWrapper = Join-Path $PSScriptRoot "train_yolox_logged.py"
$Yolo11Pretrained = Join-Path $ProjectRoot "weights\pretrained\yolo11m.pt"
$YoloXPretrained = Join-Path $ProjectRoot "weights\pretrained\yolox_s.pth"
$YoloXConfig = Join-Path $ProjectRoot "configs\yolox_s_micropcb.py"
$YoloXSourceRoot = Join-Path $ProjectRoot ".deps\YOLOX"
$ImplementationInputs = @(
    $PSCommandPath,
    $Yolo11Wrapper,
    $YoloXWrapper,
    $YoloXConfig,
    (Join-Path $ProjectRoot "src\mcu_data\common.py"),
    (Join-Path $ProjectRoot "src\mcu_data\dataset_evidence.py"),
    (Join-Path $ProjectRoot "src\mcu_data\methodology.py"),
    (Join-Path $ProjectRoot "src\mcu_data\publishing.py"),
    (Join-Path $ProjectRoot "src\mcu_data\reporting.py"),
    (Join-Path $ProjectRoot "src\mcu_data\runlog.py"),
    (Join-Path $ProjectRoot "src\mcu_data\yolox_metrics.py"),
    $Yolo11Pretrained,
    $YoloXPretrained
)
$RequiredEvidenceFields = @(
    "canonical_dataset_manifest_sha256",
    "class_map_sha256",
    "train_image_list_sha256",
    "val_image_list_sha256",
    "canonical_train_records_sha256",
    "canonical_val_records_sha256"
)

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Path))
}

function Assert-LeafPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label file is missing: $Path"
    }
}

function Assert-ContainerPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label directory is missing: $Path"
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $Algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return ([System.BitConverter]::ToString($Algorithm.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $Algorithm.Dispose()
    }
}

function Get-ImplementationRecords {
    $Records = [ordered]@{}
    $RootPrefix = $ProjectRoot.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
    foreach ($Path in $ImplementationInputs) {
        $Resolved = [System.IO.Path]::GetFullPath($Path)
        if (-not $Resolved.StartsWith($RootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Implementation input is outside the project root: $Resolved"
        }
        $Relative = $Resolved.Substring($RootPrefix.Length).Replace("\", "/")
        $Records[$Relative] = Get-Sha256 $Resolved
    }
    return $Records
}

function Get-ImplementationSnapshotSha256 {
    $Records = Get-ImplementationRecords
    $Lines = @($Records.GetEnumerator() | ForEach-Object { "$($_.Key):$($_.Value)" })
    return Get-TextSha256 ($Lines -join "`n")
}

function Invoke-GitText {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$Repository = $ProjectRoot
    )
    $Output = @(& git -C $Repository @Arguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git -C $Repository $($Arguments -join ' ')`n$($Output -join "`n")"
    }
    return (($Output | ForEach-Object { [string]$_ }) -join "`n").Trim()
}

function Get-YoloXSourceState {
    $Head = Invoke-GitText -Repository $YoloXSourceRoot -Arguments @("rev-parse", "HEAD")
    $Status = Invoke-GitText -Repository $YoloXSourceRoot -Arguments @("status", "--porcelain=v1", "--untracked-files=all")
    return [PSCustomObject]@{
        head = $Head
        clean = [string]::IsNullOrWhiteSpace($Status)
        status = $Status
    }
}

function Assert-YoloXSourceFrozen {
    param([Parameter(Mandatory = $true)][string]$ExpectedHead)
    $State = Get-YoloXSourceState
    if ($State.head -ne $ExpectedHead) {
        throw "YOLOX source HEAD changed after campaign preflight: expected=$ExpectedHead actual=$($State.head)"
    }
    if (-not $State.clean) {
        throw "YOLOX editable source must stay clean during the campaign:`n$($State.status)"
    }
    return $State
}

function Get-RepositoryState {
    $Head = Invoke-GitText @("rev-parse", "HEAD")
    $Upstream = Invoke-GitText @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    $UpstreamHead = Invoke-GitText @("rev-parse", "@{u}")
    $Status = Invoke-GitText @("status", "--porcelain=v1", "--untracked-files=all")
    return [PSCustomObject]@{
        head = $Head
        upstream = $Upstream
        upstream_head = $UpstreamHead
        clean = [string]::IsNullOrWhiteSpace($Status)
        status = $Status
    }
}

function Assert-RepositoryFrozen {
    param([Parameter(Mandatory = $true)][string]$ExpectedHead)
    $State = Get-RepositoryState
    if ($State.head -ne $ExpectedHead) {
        throw "Git HEAD changed after campaign preflight: expected=$ExpectedHead actual=$($State.head)"
    }
    if (-not $State.clean) {
        throw "Repository must stay clean during the campaign. Commit/push or revert these paths:`n$($State.status)"
    }
    if ($State.upstream_head -ne $State.head) {
        throw "Git HEAD is not pushed to upstream $($State.upstream): local=$($State.head) upstream=$($State.upstream_head)"
    }
    return $State
}

function ConvertTo-SafeStem {
    param([Parameter(Mandatory = $true)][string]$Value)
    $Stem = $Value -replace "[^A-Za-z0-9._-]+", "_"
    $Stem = $Stem.Trim("_", ".", "-")
    if ([string]::IsNullOrWhiteSpace($Stem)) {
        throw "Could not derive a safe campaign name from: $Value"
    }
    return $Stem
}

function Format-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][object[]]$Arguments
    )
    $Parts = @($Executable) + @($Arguments)
    return (($Parts | ForEach-Object {
                $Text = [string]$_
                if ($Text -match "[\s']") {
                    return "'" + $Text.Replace("'", "''") + "'"
                }
                return $Text
            }) -join " ")
}

function New-ComparisonArguments {
    param(
        [Parameter(Mandatory = $true)][string[]]$RunDirectories,
        [Parameter(Mandatory = $true)][string]$OutputDirectory,
        [switch]$Formal
    )
    $Arguments = @("--runs") + @($RunDirectories) + @("--output-dir", $OutputDirectory)
    if ($Formal) {
        $Arguments += "--formal"
    }
    return $Arguments
}

function Assert-ComparisonResult {
    param(
        [Parameter(Mandatory = $true)][string]$ComparisonDirectory,
        [switch]$Smoke
    )
    $CompatibilityPath = Join-Path $ComparisonDirectory "protocol_compatibility.json"
    if (-not (Test-Path -LiteralPath $CompatibilityPath -PathType Leaf)) {
        throw "Comparison compatibility file is missing: $CompatibilityPath"
    }
    try {
        $Compatibility = Get-Content -LiteralPath $CompatibilityPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        throw "Comparison compatibility is not valid JSON: $CompatibilityPath`n$($_.Exception.Message)"
    }
    $ComparableIsPass = (
        ($Compatibility.comparable -is [bool]) -and
        ($Compatibility.comparable -eq $true)
    )
    $CriticalMismatchesAreEmpty = (
        ($Compatibility.critical_mismatches -is [System.Array]) -and
        (@($Compatibility.critical_mismatches).Count -eq 0)
    )
    if (-not $ComparableIsPass -or -not $CriticalMismatchesAreEmpty) {
        $Mismatches = @($Compatibility.critical_mismatches | ForEach-Object { $_.field }) -join ", "
        throw "Comparison completed but comparable is not an unblocked PASS. Critical mismatches: $Mismatches"
    }

    $FormalValidationPath = Join-Path $ComparisonDirectory "formal_validation.json"
    if ($Smoke) {
        if (Test-Path -LiteralPath $FormalValidationPath -PathType Leaf) {
            throw "Smoke comparison must not produce formal_validation.json: $FormalValidationPath"
        }
        return [PSCustomObject]@{
            Compatibility = $Compatibility
            FormalValidation = $null
        }
    }

    $ReleaseReadyIsPass = (
        ($Compatibility.release_ready -is [bool]) -and
        ($Compatibility.release_ready -eq $true)
    )
    $ReleaseBlockersAreEmpty = (
        ($Compatibility.release_blockers -is [System.Array]) -and
        (@($Compatibility.release_blockers).Count -eq 0)
    )
    $CompatibilityRunCountIsSix = (
        (($Compatibility.run_count -is [int]) -or ($Compatibility.run_count -is [long])) -and
        ($Compatibility.run_count -eq 6)
    )
    $CompatibilityRunCountType = "<missing>"
    if ($null -ne $Compatibility.run_count) {
        $CompatibilityRunCountType = $Compatibility.run_count.GetType().FullName
    }
    if (-not $ReleaseReadyIsPass -or -not $ReleaseBlockersAreEmpty -or -not $CompatibilityRunCountIsSix) {
        $Blockers = @($Compatibility.release_blockers | ForEach-Object { $_.field }) -join ", "
        throw (
            "Full comparison release gate is not an unblocked PASS for an exact six-run comparison. " +
            "release_ready_pass=$ReleaseReadyIsPass, release_blockers_empty=$ReleaseBlockersAreEmpty, " +
            "run_count_six=$CompatibilityRunCountIsSix, " +
            "run_count_type=$CompatibilityRunCountType. Blockers: $Blockers"
        )
    }
    if (-not (Test-Path -LiteralPath $FormalValidationPath -PathType Leaf)) {
        throw "Full comparison did not produce formal_validation.json: $FormalValidationPath"
    }
    try {
        $FormalValidation = Get-Content -LiteralPath $FormalValidationPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        throw "Formal validation is not valid JSON: $FormalValidationPath`n$($_.Exception.Message)"
    }
    $FormalStatusIsPass = (
        ($FormalValidation.status -is [string]) -and
        ($FormalValidation.status -ceq "PASS")
    )
    $FormalRunCountIsSix = (
        (($FormalValidation.run_count -is [int]) -or ($FormalValidation.run_count -is [long])) -and
        ($FormalValidation.run_count -eq 6)
    )
    if (-not $FormalStatusIsPass -or -not $FormalRunCountIsSix) {
        throw "Formal validation is not an exact six-run PASS: status=$($FormalValidation.status), run_count=$($FormalValidation.run_count)"
    }
    return [PSCustomObject]@{
        Compatibility = $Compatibility
        FormalValidation = $FormalValidation
    }
}

function Read-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    try {
        return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
    }
    catch {
        throw "$Label is not valid JSON: $Path`n$($_.Exception.Message)"
    }
}

function Get-ProtocolSummary {
    param([Parameter(Mandatory = $true)][string]$Path)
    $ProtocolId = $null
    $Common = [ordered]@{}
    $Dataset = [ordered]@{}
    $Evidence = [ordered]@{}
    $TopSection = ""
    $DatasetSubsection = ""
    foreach ($Line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        if ($Line -match "^protocol_id:\s*(\S.*?)\s*$") {
            $ProtocolId = $Matches[1].Trim("'", '"')
            continue
        }
        if ($Line -match "^([A-Za-z0-9_]+):(?:\s.*)?$") {
            $TopSection = $Matches[1]
            $DatasetSubsection = ""
            continue
        }
        if ($TopSection -eq "common" -and $Line -match "^  ([A-Za-z0-9_]+):\s*(.*?)\s*$") {
            $Name = $Matches[1]
            $Value = $Matches[2].Trim("'", '"')
            $Common[$Name] = $Value
            continue
        }
        if ($TopSection -eq "dataset" -and $Line -match "^  ([A-Za-z0-9_]+):\s*(.*?)\s*$") {
            $Name = $Matches[1]
            $Value = $Matches[2].Trim("'", '"')
            if ([string]::IsNullOrWhiteSpace($Value)) {
                $DatasetSubsection = $Name
            }
            else {
                $Dataset[$Name] = $Value
                $DatasetSubsection = ""
            }
            continue
        }
        if ($TopSection -eq "dataset" -and $DatasetSubsection -eq "evidence" -and $Line -match "^    ([A-Za-z0-9_]+):\s*(.*?)\s*$") {
            $Evidence[$Matches[1]] = $Matches[2].Trim("'", '"')
        }
    }
    $Dataset["evidence"] = [PSCustomObject]$Evidence
    if ($Common.Contains("seeds")) {
        $Common["seeds"] = @(($Common["seeds"].Trim("[", "]") -split ",") | ForEach-Object { [int]$_.Trim() })
    }
    return [PSCustomObject]@{
        protocol_id = $ProtocolId
        common = [PSCustomObject]$Common
        dataset = [PSCustomObject]$Dataset
    }
}

function Test-SamePath {
    param(
        [Parameter(Mandatory = $true)][string]$Left,
        [Parameter(Mandatory = $true)][string]$Right
    )
    return [string]::Equals(
        [System.IO.Path]::GetFullPath($Left).TrimEnd("\", "/"),
        [System.IO.Path]::GetFullPath($Right).TrimEnd("\", "/"),
        [System.StringComparison]::OrdinalIgnoreCase
    )
}

function Test-RunComplete {
    param(
        [Parameter(Mandatory = $true)][string]$RunDirectory,
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$ExpectedModel,
        [Parameter(Mandatory = $true)][int]$ExpectedSeed,
        [Parameter(Mandatory = $true)][int]$ExpectedEpochs,
        [Parameter(Mandatory = $true)][string]$ExpectedProtocolSha256,
        [Parameter(Mandatory = $true)][string]$ExpectedPretrainedSha256,
        [Parameter(Mandatory = $true)][object]$ExpectedEvidence
    )
    $Reasons = [System.Collections.Generic.List[string]]::new()
    if (-not (Test-Path -LiteralPath $RunDirectory -PathType Container)) {
        return [PSCustomObject]@{
            Exists = $false
            Complete = $false
            Reasons = @("run directory does not exist")
        }
    }

    $ManifestPath = Join-Path $RunDirectory "run_manifest.json"
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        $Reasons.Add("run_manifest.json is missing")
        return [PSCustomObject]@{ Exists = $true; Complete = $false; Reasons = @($Reasons) }
    }
    try {
        $Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        $Reasons.Add("run_manifest.json is invalid JSON: $($_.Exception.Message)")
        return [PSCustomObject]@{ Exists = $true; Complete = $false; Reasons = @($Reasons) }
    }

    if ($Manifest.status -ne "complete") {
        $Reasons.Add("manifest status is '$($Manifest.status)', expected 'complete'")
    }
    if ($Manifest.run_id -ne $RunId) {
        $Reasons.Add("manifest run_id is '$($Manifest.run_id)', expected '$RunId'")
    }
    $NormalizedActualModel = ([string]$Manifest.model).ToLowerInvariant() -replace "[^a-z0-9]", ""
    $NormalizedExpectedModel = $ExpectedModel.ToLowerInvariant() -replace "[^a-z0-9]", ""
    if ($NormalizedActualModel -ne $NormalizedExpectedModel) {
        $Reasons.Add("manifest model is '$($Manifest.model)', expected '$ExpectedModel'")
    }
    if ([int]$Manifest.protocol.seed -ne $ExpectedSeed) {
        $Reasons.Add("manifest seed is '$($Manifest.protocol.seed)', expected '$ExpectedSeed'")
    }
    if ([int]$Manifest.protocol.epochs -ne $ExpectedEpochs) {
        $Reasons.Add("manifest epochs is '$($Manifest.protocol.epochs)', expected '$ExpectedEpochs'")
    }
    if ([int]$Manifest.protocol.batch -ne $Batch) {
        $Reasons.Add("manifest batch is '$($Manifest.protocol.batch)', expected '$Batch'")
    }
    if ([int]$Manifest.protocol.imgsz -ne $ImageSize) {
        $Reasons.Add("manifest imgsz is '$($Manifest.protocol.imgsz)', expected '$ImageSize'")
    }
    $ExpectedManifestWorkers = $Workers
    if ($Smoke) {
        $ExpectedManifestWorkers = 0
    }
    if ([int]$Manifest.protocol.workers -ne $ExpectedManifestWorkers) {
        $Reasons.Add("manifest workers is '$($Manifest.protocol.workers)', expected '$ExpectedManifestWorkers'")
    }
    if ($Smoke) {
        if ($Manifest.stage -ne "smoke_not_comparable") {
            $Reasons.Add("manifest stage is '$($Manifest.stage)', expected 'smoke_not_comparable'")
        }
    }
    elseif ([string]::IsNullOrWhiteSpace([string]$Manifest.stage) -or $Manifest.stage -eq "smoke_not_comparable") {
        $Reasons.Add("manifest stage '$($Manifest.stage)' is not a full-training stage")
    }
    if ($Manifest.protocol_config.sha256 -ne $ExpectedProtocolSha256) {
        $Reasons.Add("protocol SHA-256 differs from this campaign")
    }
    if ($Manifest.pretrained_checkpoint.sha256 -ne $ExpectedPretrainedSha256) {
        $Reasons.Add("pretrained checkpoint SHA-256 differs from this campaign")
    }
    foreach ($Field in $RequiredEvidenceFields) {
        $Actual = $Manifest.dataset.$Field
        $Expected = $ExpectedEvidence.$Field
        if ($Actual -ne $Expected) {
            $Reasons.Add("dataset evidence '$Field' is '$Actual', expected '$Expected'")
        }
    }

    $RequiredFiles = @(
        "terminal.log",
        "epoch_metrics.csv",
        "final_metrics.json",
        "predictions.coco.json",
        "latency.json",
        "gpu_summary.json",
        "best_weights_summary.csv"
    )
    foreach ($RelativePath in $RequiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $RunDirectory $RelativePath) -PathType Leaf)) {
            $Reasons.Add("required artifact is missing: $RelativePath")
        }
    }

    $EpochMetricsPath = Join-Path $RunDirectory "epoch_metrics.csv"
    if (Test-Path -LiteralPath $EpochMetricsPath -PathType Leaf) {
        try {
            $EpochRows = @(Import-Csv -LiteralPath $EpochMetricsPath)
            if ($EpochRows.Count -ne $ExpectedEpochs) {
                $Reasons.Add("epoch_metrics.csv has $($EpochRows.Count) rows, expected $ExpectedEpochs")
            }
            elseif ([int]$EpochRows[-1].epoch -ne $ExpectedEpochs) {
                $Reasons.Add("last epoch row is '$($EpochRows[-1].epoch)', expected '$ExpectedEpochs'")
            }
        }
        catch {
            $Reasons.Add("epoch_metrics.csv could not be parsed: $($_.Exception.Message)")
        }
    }

    $CheckpointPath = [string]$Manifest.best_checkpoint.path
    if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
        $Reasons.Add("best_checkpoint.path is missing")
    }
    elseif (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
        $Reasons.Add("checkpoint file is missing: $CheckpointPath")
    }
    else {
        $ActualCheckpointSha256 = Get-Sha256 $CheckpointPath
        if ($ActualCheckpointSha256 -ne $Manifest.best_checkpoint.sha256) {
            $Reasons.Add("checkpoint SHA-256 does not match run_manifest.json")
        }
    }

    $FinalMetricsPath = Join-Path $RunDirectory "final_metrics.json"
    if (Test-Path -LiteralPath $FinalMetricsPath -PathType Leaf) {
        try {
            $FinalMetrics = Get-Content -LiteralPath $FinalMetricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($Metric in @("ap50_95", "ap50", "ap75", "ar100", "precision", "recall", "f1", "tp", "fp", "fn")) {
                if ($null -eq $FinalMetrics.metrics.$Metric) {
                    $Reasons.Add("final metric is missing: $Metric")
                }
            }
        }
        catch {
            $Reasons.Add("final_metrics.json is invalid: $($_.Exception.Message)")
        }
    }

    $LatencyPath = Join-Path $RunDirectory "latency.json"
    if (Test-Path -LiteralPath $LatencyPath -PathType Leaf) {
        try {
            $Latency = Get-Content -LiteralPath $LatencyPath -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($Metric in @("e2e_p50_ms", "e2e_p95_ms", "sustained_fps")) {
                if ($null -eq $Latency.$Metric) {
                    $Reasons.Add("latency metric is missing: $Metric")
                }
            }
        }
        catch {
            $Reasons.Add("latency.json is invalid: $($_.Exception.Message)")
        }
    }

    return [PSCustomObject]@{
        Exists = $true
        Complete = ($Reasons.Count -eq 0)
        Reasons = @($Reasons)
    }
}

foreach ($Executable in @($Yolo11Python, $YoloXPython, $CompareExecutable)) {
    Assert-LeafPath $Executable "Required executable"
}
foreach ($SourceInput in $ImplementationInputs) {
    Assert-LeafPath $SourceInput "Training implementation input"
}

$ProtocolPath = Resolve-ProjectPath $ProtocolConfig
$YoloDataPath = Resolve-ProjectPath $YoloData
$CocoRootPath = Resolve-ProjectPath $CocoRoot
$DatasetEvidencePath = Resolve-ProjectPath $DatasetEvidence
Assert-LeafPath $ProtocolPath "Protocol config"
Assert-LeafPath $YoloDataPath "YOLO dataset YAML"
Assert-ContainerPath $CocoRootPath "COCO root"
Assert-LeafPath $DatasetEvidencePath "Dataset evidence"
Assert-LeafPath (Join-Path $CocoRootPath "annotations\instances_train2017.json") "COCO train annotation"
Assert-LeafPath (Join-Path $CocoRootPath "annotations\instances_val2017.json") "COCO validation annotation"
Assert-ContainerPath (Join-Path $CocoRootPath "train2017") "COCO train images"
Assert-ContainerPath (Join-Path $CocoRootPath "val2017") "COCO validation images"

if ($Seeds.Count -eq 0) {
    throw "At least one seed is required."
}
$DuplicateSeeds = @($Seeds | Group-Object | Where-Object Count -gt 1)
if ($DuplicateSeeds.Count -gt 0) {
    throw "Duplicate seeds are not allowed: $((@($DuplicateSeeds.Name) -join ', '))"
}
$NormalizedSeeds = @($Seeds | Sort-Object)
$ExpectedEpochs = $Epochs
if ($Smoke) {
    $ExpectedEpochs = 1
}

$Evidence = Read-JsonFile $DatasetEvidencePath "Dataset evidence"
if ($Evidence.status -ne "PASS") {
    throw "Dataset evidence status is '$($Evidence.status)', expected 'PASS': $DatasetEvidencePath"
}
foreach ($Field in $RequiredEvidenceFields) {
    $Value = [string]$Evidence.$Field
    if ($Value -notmatch "^[0-9a-fA-F]{64}$") {
        throw "Dataset evidence field '$Field' is missing or is not a SHA-256 value."
    }
}

$Protocol = Get-ProtocolSummary $ProtocolPath
if ([string]::IsNullOrWhiteSpace([string]$Protocol.protocol_id)) {
    throw "protocol_id is missing from protocol config: $ProtocolPath"
}
$ProtocolSeeds = @($Protocol.common.seeds | ForEach-Object { [int]$_ } | Sort-Object)
if (-not $Smoke) {
    if ([int]$Protocol.common.epochs -ne $Epochs) {
        throw "Epochs=$Epochs differs from protocol common.epochs=$($Protocol.common.epochs). Create/use a matching protocol config."
    }
    if ([int]$Protocol.common.batch_size -ne $Batch) {
        throw "Batch=$Batch differs from protocol common.batch_size=$($Protocol.common.batch_size). Create/use a matching protocol config."
    }
    if ([int]$Protocol.common.image_size -ne $ImageSize) {
        throw "ImageSize=$ImageSize differs from protocol common.image_size=$($Protocol.common.image_size). Create/use a matching protocol config."
    }
    if ($null -ne $Protocol.common.workers -and [int]$Protocol.common.workers -ne $Workers) {
        throw "Workers=$Workers differs from protocol common.workers=$($Protocol.common.workers). Create/use a matching protocol config."
    }
    if (($NormalizedSeeds -join ",") -ne ($ProtocolSeeds -join ",")) {
        throw "Seeds=$($NormalizedSeeds -join ',') differs from protocol common.seeds=$($ProtocolSeeds -join ','). Create/use a matching protocol config."
    }
}

foreach ($DatasetBinding in @(
        [PSCustomObject]@{ Name = "dataset.yolo_dataset"; Configured = $Protocol.dataset.yolo_dataset; Actual = $YoloDataPath },
        [PSCustomObject]@{ Name = "dataset.coco_root"; Configured = $Protocol.dataset.coco_root; Actual = $CocoRootPath },
        [PSCustomObject]@{ Name = "dataset.equivalence_evidence"; Configured = $Protocol.dataset.equivalence_evidence; Actual = $DatasetEvidencePath }
    )) {
    if (-not [string]::IsNullOrWhiteSpace([string]$DatasetBinding.Configured)) {
        $ConfiguredPath = Resolve-ProjectPath ([string]$DatasetBinding.Configured)
        if (-not (Test-SamePath $ConfiguredPath $DatasetBinding.Actual)) {
            throw "$($DatasetBinding.Name) resolves to '$ConfiguredPath', but the campaign argument resolves to '$($DatasetBinding.Actual)'."
        }
    }
}
foreach ($Field in $RequiredEvidenceFields) {
    $ConfiguredValue = [string]$Protocol.dataset.evidence.$Field
    $ActualValue = [string]$Evidence.$Field
    if (-not [string]::IsNullOrWhiteSpace($ConfiguredValue) -and $ConfiguredValue -ne $ActualValue) {
        throw "Protocol dataset.evidence.$Field does not match dataset_evidence.json."
    }
}

$ProtocolSha256 = Get-Sha256 $ProtocolPath
$YoloDataSha256 = Get-Sha256 $YoloDataPath
$DatasetEvidenceSha256 = Get-Sha256 $DatasetEvidencePath
$TrainAnnotationSha256 = Get-Sha256 (Join-Path $CocoRootPath "annotations\instances_train2017.json")
$ValAnnotationSha256 = Get-Sha256 (Join-Path $CocoRootPath "annotations\instances_val2017.json")
$RepositoryState = Get-RepositoryState
$YoloXSourceState = Get-YoloXSourceState
if ($YoloXSourceState.head -ne "6ddff4824372906469a7fae2dc3206c7aa4bbaee") {
    throw "YOLOX source must be pinned to 6ddff4824372906469a7fae2dc3206c7aa4bbaee, got $($YoloXSourceState.head)"
}
if (-not $YoloXSourceState.clean) {
    throw "YOLOX editable source is dirty:`n$($YoloXSourceState.status)"
}
$Yolo11WrapperSha256 = Get-Sha256 $Yolo11Wrapper
$YoloXWrapperSha256 = Get-Sha256 $YoloXWrapper
$Yolo11PretrainedSha256 = Get-Sha256 $Yolo11Pretrained
$YoloXPretrainedSha256 = Get-Sha256 $YoloXPretrained
$YoloXConfigSha256 = Get-Sha256 $YoloXConfig
$ImplementationRecords = Get-ImplementationRecords
$ImplementationSha256 = Get-ImplementationSnapshotSha256

if ([string]::IsNullOrWhiteSpace($CampaignId)) {
    $Mode = "full"
    if ($Smoke) {
        $Mode = "smoke"
    }
    $ProtocolStem = ConvertTo-SafeStem ([string]$Protocol.protocol_id)
    $CampaignId = "${ProtocolStem}_${Mode}_e${ExpectedEpochs}_b${Batch}_i${ImageSize}_w${Workers}_s$($NormalizedSeeds -join '-')_p$($ProtocolSha256.Substring(0, 10))_d$($DatasetEvidenceSha256.Substring(0, 10))_c$($ImplementationSha256.Substring(0, 10))"
}
if ($CampaignId -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$" -or $CampaignId -in @(".", "..")) {
    throw "CampaignId must be 1-120 safe path characters: letters, numbers, dot, underscore, or hyphen."
}

$CampaignRunRoot = Join-Path $ProjectRoot "runs\benchmarks\$CampaignId"
$ComparisonDirectory = Join-Path $ProjectRoot "runs\comparisons\$CampaignId"
$CampaignPlanPath = Join-Path $CampaignRunRoot "campaign_plan.json"
$CampaignSignature = @(
    $ProtocolSha256,
    $YoloDataSha256,
    $DatasetEvidenceSha256,
    $TrainAnnotationSha256,
    $ValAnnotationSha256,
    $ImplementationSha256,
    $RepositoryState.head,
    $YoloXSourceState.head,
    $ExpectedEpochs,
    $Batch,
    $ImageSize,
    $Workers,
    ($NormalizedSeeds -join ","),
    [bool]$Smoke
) -join "|"

$RunSpecs = [System.Collections.Generic.List[object]]::new()
foreach ($Seed in $NormalizedSeeds) {
    $Yolo11RunId = "yolo11m_seed${Seed}"
    $YoloXRunId = "yolox_s_seed${Seed}"
    $Yolo11Args = @(
        $Yolo11Wrapper,
        "--run-id", $Yolo11RunId,
        "--model", $Yolo11Pretrained,
        "--protocol-config", $ProtocolPath,
        "--data", $YoloDataPath,
        "--coco-root", $CocoRootPath,
        "--dataset-evidence", $DatasetEvidencePath,
        "--output-root", $CampaignRunRoot,
        "--epochs", $ExpectedEpochs,
        "--batch", $Batch,
        "--imgsz", $ImageSize,
        "--workers", $Workers,
        "--seed", $Seed
    )
    $YoloXArgs = @(
        $YoloXWrapper,
        "--run-id", $YoloXRunId,
        "--protocol-config", $ProtocolPath,
        "--yolo-data", $YoloDataPath,
        "--coco-root", $CocoRootPath,
        "--dataset-evidence", $DatasetEvidencePath,
        "--output-root", $CampaignRunRoot,
        "--epochs", $ExpectedEpochs,
        "--batch", $Batch,
        "--imgsz", $ImageSize,
        "--workers", $Workers,
        "--seed", $Seed
    )
    if ($Smoke) {
        $Yolo11Args += "--smoke"
        $YoloXArgs += "--smoke"
    }
    $RunSpecs.Add([PSCustomObject]@{
            Model = "yolo11m"
            Seed = $Seed
            RunId = $Yolo11RunId
            RunDirectory = Join-Path $CampaignRunRoot $Yolo11RunId
            Executable = $Yolo11Python
            PretrainedSha256 = $Yolo11PretrainedSha256
            Arguments = $Yolo11Args
        })
    $RunSpecs.Add([PSCustomObject]@{
            Model = "YOLOX-S"
            Seed = $Seed
            RunId = $YoloXRunId
            RunDirectory = Join-Path $CampaignRunRoot $YoloXRunId
            Executable = $YoloXPython
            PretrainedSha256 = $YoloXPretrainedSha256
            Arguments = $YoloXArgs
        })
}

if (Test-Path -LiteralPath $CampaignRunRoot -PathType Container) {
    if (Test-Path -LiteralPath $CampaignPlanPath -PathType Leaf) {
        $ExistingPlan = Read-JsonFile $CampaignPlanPath "Campaign plan"
        if ($ExistingPlan.campaign_signature -ne $CampaignSignature) {
            throw "CampaignId '$CampaignId' already belongs to different inputs/settings. Use a different -CampaignId."
        }
    }
    else {
        $ExistingEntries = @(Get-ChildItem -LiteralPath $CampaignRunRoot -Force)
        if ($ExistingEntries.Count -gt 0) {
            throw "Campaign root exists without campaign_plan.json and is not empty: $CampaignRunRoot"
        }
    }
}

$RunStates = [System.Collections.Generic.List[object]]::new()
foreach ($Spec in $RunSpecs) {
    $State = Test-RunComplete `
        -RunDirectory $Spec.RunDirectory `
        -RunId $Spec.RunId `
        -ExpectedModel $Spec.Model `
        -ExpectedSeed $Spec.Seed `
        -ExpectedEpochs $ExpectedEpochs `
        -ExpectedProtocolSha256 $ProtocolSha256 `
        -ExpectedPretrainedSha256 $Spec.PretrainedSha256 `
        -ExpectedEvidence $Evidence
    if ($State.Exists -and -not $State.Complete) {
        $ReasonText = ($State.Reasons | ForEach-Object { "  - $_" }) -join "`n"
        throw "Existing run is incomplete or incompatible: $($Spec.RunDirectory)`n$ReasonText`nArchive or remove only this exact run directory after inspecting terminal.log, then rerun the campaign."
    }
    $RunStates.Add([PSCustomObject]@{ Spec = $Spec; State = $State })
}

Write-Host ""
Write-Host "TRAINING CAMPAIGN PREFLIGHT: PASS"
Write-Host ("=" * 92)
Write-Host "Campaign ID       : $CampaignId"
$ModeLabel = "FULL"
if ($Smoke) {
    $ModeLabel = "SMOKE (not comparable)"
}
Write-Host "Mode              : $ModeLabel"
Write-Host "Protocol          : $ProtocolPath"
Write-Host "Protocol SHA-256  : $ProtocolSha256"
Write-Host "YOLO dataset      : $YoloDataPath"
Write-Host "COCO root         : $CocoRootPath"
Write-Host "Dataset evidence  : $DatasetEvidencePath"
Write-Host "Evidence SHA-256  : $DatasetEvidenceSha256"
Write-Host "Implementation SHA: $ImplementationSha256"
Write-Host "Git HEAD           : $($RepositoryState.head)"
Write-Host "Git upstream       : $($RepositoryState.upstream) @ $($RepositoryState.upstream_head)"
Write-Host "Git clean          : $($RepositoryState.clean)"
Write-Host "YOLOX source       : $($YoloXSourceState.head) (clean=$($YoloXSourceState.clean))"
Write-Host "Seeds/epochs      : $($NormalizedSeeds -join ',') / $ExpectedEpochs"
Write-Host "Batch/image size  : $Batch / $ImageSize"
Write-Host "Data workers      : $Workers"
if ($Smoke) {
    Write-Host "Smoke workers     : 0 (wrappers force single-process loading)"
}
Write-Host "Run root          : $CampaignRunRoot"
Write-Host "Comparison        : $ComparisonDirectory"
foreach ($Item in $RunStates) {
    $Disposition = "RUN"
    if ($Item.State.Complete) {
        $Disposition = "SKIP (verified complete)"
    }
    Write-Host ("  [{0}] {1} seed={2}: {3}" -f $Disposition, $Item.Spec.Model, $Item.Spec.Seed, $Item.Spec.RunDirectory)
    Write-Host ("    " + (Format-NativeCommand $Item.Spec.Executable $Item.Spec.Arguments))
}

$ComparisonArguments = @(
    New-ComparisonArguments `
        -RunDirectories @($RunSpecs | ForEach-Object { $_.RunDirectory }) `
        -OutputDirectory $ComparisonDirectory `
        -Formal:(-not $Smoke)
)
Write-Host "  [COMPARE] $(Format-NativeCommand $CompareExecutable $ComparisonArguments)"

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN: no directories were created and no training/comparison command was executed."
    return
}

$RepositoryState = Assert-RepositoryFrozen -ExpectedHead $RepositoryState.head
$YoloXSourceState = Assert-YoloXSourceFrozen -ExpectedHead $YoloXSourceState.head

if (-not (Test-Path -LiteralPath $CampaignRunRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $CampaignRunRoot | Out-Null
}
if (-not (Test-Path -LiteralPath $CampaignPlanPath -PathType Leaf)) {
    $CampaignPlan = [ordered]@{
        schema_version = 1
        campaign_id = $CampaignId
        campaign_signature = $CampaignSignature
        created_utc = [DateTime]::UtcNow.ToString("o")
        smoke = [bool]$Smoke
        protocol = [ordered]@{ path = $ProtocolPath; sha256 = $ProtocolSha256 }
        yolo_data = [ordered]@{ path = $YoloDataPath; sha256 = $YoloDataSha256 }
        coco_root = [ordered]@{
            path = $CocoRootPath
            train_annotation_sha256 = $TrainAnnotationSha256
            val_annotation_sha256 = $ValAnnotationSha256
        }
        dataset_evidence = [ordered]@{ path = $DatasetEvidencePath; sha256 = $DatasetEvidenceSha256 }
        implementation = [ordered]@{
            sha256 = $ImplementationSha256
            source_files = $ImplementationRecords
            yolo11_wrapper_sha256 = $Yolo11WrapperSha256
            yolox_wrapper_sha256 = $YoloXWrapperSha256
            yolo11_pretrained_sha256 = $Yolo11PretrainedSha256
            yolox_pretrained_sha256 = $YoloXPretrainedSha256
            yolox_config_sha256 = $YoloXConfigSha256
        }
        git = [ordered]@{
            head = $RepositoryState.head
            upstream = $RepositoryState.upstream
            upstream_head = $RepositoryState.upstream_head
            clean = $RepositoryState.clean
        }
        yolox_source = [ordered]@{
            head = $YoloXSourceState.head
            clean = $YoloXSourceState.clean
        }
        parameters = [ordered]@{
            seeds = @($NormalizedSeeds)
            epochs = $ExpectedEpochs
            batch = $Batch
            image_size = $ImageSize
            workers = $Workers
        }
        run_directories = @($RunSpecs | ForEach-Object { $_.RunDirectory })
        comparison_directory = $ComparisonDirectory
    }
    $CampaignPlan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $CampaignPlanPath -Encoding UTF8
}

Push-Location $ProjectRoot
try {
    foreach ($Item in $RunStates) {
        $Spec = $Item.Spec
        if ($Item.State.Complete) {
            Write-Host ""
            Write-Host "SKIP verified complete run: $($Spec.RunId)"
            continue
        }

        Write-Host ""
        Write-Host "START $($Spec.Model) seed=$($Spec.Seed): $($Spec.RunDirectory)"
        Assert-RepositoryFrozen -ExpectedHead $RepositoryState.head | Out-Null
        Assert-YoloXSourceFrozen -ExpectedHead $YoloXSourceState.head | Out-Null
        $CurrentImplementationSha256 = Get-ImplementationSnapshotSha256
        if ($CurrentImplementationSha256 -ne $ImplementationSha256) {
            throw "Training implementation/pretrained input changed after campaign preflight. Start a new campaign."
        }
        & $Spec.Executable @($Spec.Arguments)
        $TrainingExitCode = $LASTEXITCODE
        if ($TrainingExitCode -ne 0) {
            throw "$($Spec.Model) seed $($Spec.Seed) failed with exit code $TrainingExitCode. Inspect: $($Spec.RunDirectory)\terminal.log"
        }

        $CompletedState = Test-RunComplete `
            -RunDirectory $Spec.RunDirectory `
            -RunId $Spec.RunId `
            -ExpectedModel $Spec.Model `
            -ExpectedSeed $Spec.Seed `
            -ExpectedEpochs $ExpectedEpochs `
            -ExpectedProtocolSha256 $ProtocolSha256 `
            -ExpectedPretrainedSha256 $Spec.PretrainedSha256 `
            -ExpectedEvidence $Evidence
        if (-not $CompletedState.Complete) {
            $ReasonText = ($CompletedState.Reasons | ForEach-Object { "  - $_" }) -join "`n"
            throw "$($Spec.Model) seed $($Spec.Seed) returned exit code 0 but did not produce a complete run:`n$ReasonText"
        }
        Write-Host "VERIFIED COMPLETE: $($Spec.RunId)"
    }

    Write-Host ""
    Write-Host "START COMMON COMPARISON: $ComparisonDirectory"
    Assert-RepositoryFrozen -ExpectedHead $RepositoryState.head | Out-Null
    Assert-YoloXSourceFrozen -ExpectedHead $YoloXSourceState.head | Out-Null
    if ((Get-ImplementationSnapshotSha256) -ne $ImplementationSha256) {
        throw "Implementation changed before common comparison. Start a new campaign."
    }
    & $CompareExecutable @ComparisonArguments
    $ComparisonExitCode = $LASTEXITCODE
    if ($ComparisonExitCode -ne 0) {
        throw "Common comparison failed with exit code ${ComparisonExitCode}: $ComparisonDirectory"
    }

    $ComparisonResult = Assert-ComparisonResult `
        -ComparisonDirectory $ComparisonDirectory `
        -Smoke:$Smoke
    $Compatibility = $ComparisonResult.Compatibility
    Assert-RepositoryFrozen -ExpectedHead $RepositoryState.head | Out-Null
    Assert-YoloXSourceFrozen -ExpectedHead $YoloXSourceState.head | Out-Null
    if ((Get-ImplementationSnapshotSha256) -ne $ImplementationSha256) {
        throw "Implementation changed during common comparison. Do not promote this campaign."
    }

    Write-Host ""
    Write-Host "CAMPAIGN COMPLETE: $CampaignId"
    Write-Host "Common comparison: $ComparisonDirectory"
    Write-Host "Comparable       : $($Compatibility.comparable)"
    Write-Host "Release ready    : $($Compatibility.release_ready)"
    if ($Smoke) {
        Write-Host "Formal validation: OMITTED (smoke comparison)"
    }
    else {
        Write-Host "Formal validation: $($ComparisonResult.FormalValidation.status) (runs=$($ComparisonResult.FormalValidation.run_count))"
    }
}
finally {
    Pop-Location
}
