param(
    [int]$Epochs = 100,
    [int]$Batch = 8,
    [int]$ImageSize = 640,
    [int]$Seed = 42,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Yolo11Run = "yolo11m_seed${Seed}_${Stamp}"
$YoloXRun = "yolox_s_seed${Seed}_${Stamp}"

$Yolo11Python = Join-Path $ProjectRoot ".venv-yolo11\Scripts\python.exe"
$YoloXPython = Join-Path $ProjectRoot ".venv-yolox\Scripts\python.exe"
$CompareExecutable = Join-Path $ProjectRoot ".venv-collect\Scripts\mcu-compare-runs.exe"

$Yolo11Args = @(
    (Join-Path $PSScriptRoot "train_yolo11_logged.py"),
    "--run-id", $Yolo11Run,
    "--model", (Join-Path $ProjectRoot "weights\pretrained\yolo11m.pt"),
    "--epochs", $Epochs,
    "--batch", $Batch,
    "--imgsz", $ImageSize,
    "--seed", $Seed
)
$YoloXArgs = @(
    (Join-Path $PSScriptRoot "train_yolox_logged.py"),
    "--run-id", $YoloXRun,
    "--epochs", $Epochs,
    "--batch", $Batch,
    "--imgsz", $ImageSize,
    "--seed", $Seed
)

if ($Smoke) {
    $Yolo11Args += "--smoke"
    $YoloXArgs += "--smoke"
}

Push-Location $ProjectRoot
try {
    & $Yolo11Python @Yolo11Args
    if ($LASTEXITCODE -ne 0) {
        throw "YOLO11 run failed with exit code $LASTEXITCODE"
    }

    & $YoloXPython @YoloXArgs
    if ($LASTEXITCODE -ne 0) {
        throw "YOLOX run failed with exit code $LASTEXITCODE"
    }

    $ComparisonDir = Join-Path $ProjectRoot "runs\comparisons\$Stamp"
    & $CompareExecutable `
        --runs `
        (Join-Path $ProjectRoot "runs\benchmarks\$Yolo11Run") `
        (Join-Path $ProjectRoot "runs\benchmarks\$YoloXRun") `
        --output-dir $ComparisonDir
    if ($LASTEXITCODE -ne 0) {
        throw "Comparison report failed with exit code $LASTEXITCODE"
    }

    Write-Host "Comparison complete: $ComparisonDir"
}
finally {
    Pop-Location
}
