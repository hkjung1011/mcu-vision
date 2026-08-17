param(
    [int[]]$Seeds = @(42, 43, 44),
    [int]$Epochs = 100,
    [int]$Batch = 8,
    [int]$ImageSize = 640,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Yolo11Python = Join-Path $ProjectRoot ".venv-yolo11\Scripts\python.exe"
$YoloXPython = Join-Path $ProjectRoot ".venv-yolox\Scripts\python.exe"
$CompareExecutable = Join-Path $ProjectRoot ".venv-collect\Scripts\mcu-compare-runs.exe"
$RunDirectories = [System.Collections.Generic.List[string]]::new()

Push-Location $ProjectRoot
try {
    foreach ($Seed in $Seeds) {
        $Yolo11Run = "yolo11m_seed${Seed}_${Stamp}"
        $YoloXRun = "yolox_s_seed${Seed}_${Stamp}"
        $Yolo11Args = @(
            (Join-Path $PSScriptRoot "train_yolo11_logged.py"),
            "--run-id", $Yolo11Run,
            "--model", "yolo11m.pt",
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

        & $Yolo11Python @Yolo11Args
        if ($LASTEXITCODE -ne 0) {
            throw "YOLO11 seed $Seed failed with exit code $LASTEXITCODE"
        }
        $RunDirectories.Add((Join-Path $ProjectRoot "runs\benchmarks\$Yolo11Run"))

        & $YoloXPython @YoloXArgs
        if ($LASTEXITCODE -ne 0) {
            throw "YOLOX seed $Seed failed with exit code $LASTEXITCODE"
        }
        $RunDirectories.Add((Join-Path $ProjectRoot "runs\benchmarks\$YoloXRun"))
    }

    $ComparisonDir = Join-Path $ProjectRoot "runs\comparisons\three_seed_$Stamp"
    & $CompareExecutable --runs @RunDirectories --output-dir $ComparisonDir
    if ($LASTEXITCODE -ne 0) {
        throw "Three-seed comparison failed with exit code $LASTEXITCODE"
    }
    Write-Host "Three-seed comparison complete: $ComparisonDir"
}
finally {
    Pop-Location
}
