$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([int]$ExitCode, [string]$Operation) {
    if ($ExitCode -ne 0) {
        throw "$Operation failed with exit code $ExitCode."
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonLauncher = Get-Command py -ErrorAction Stop
$VenvPath = Join-Path $ProjectRoot ".venv-yolo11"

if (-not (Test-Path -LiteralPath $VenvPath)) {
    & $PythonLauncher.Source -3.11 -m venv $VenvPath
    Assert-NativeSuccess $LASTEXITCODE "Create YOLO11 virtual environment"
}

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip setuptools wheel
Assert-NativeSuccess $LASTEXITCODE "Upgrade YOLO11 build tools"
& $PythonExe -m pip install `
    torch==2.12.1 `
    torchvision==0.27.1 `
    --index-url https://download.pytorch.org/whl/cu130
Assert-NativeSuccess $LASTEXITCODE "Install YOLO11 PyTorch CUDA wheels"
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements\yolo11-cu130.lock.txt")
Assert-NativeSuccess $LASTEXITCODE "Install YOLO11 lock file"
& $PythonExe -m pip install --no-build-isolation -e $ProjectRoot
Assert-NativeSuccess $LASTEXITCODE "Install project in YOLO11 environment"

& $PythonExe -c "import json, pycocotools, torch, ultralytics; print(json.dumps({'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, ensure_ascii=False))"
Assert-NativeSuccess $LASTEXITCODE "Verify YOLO11 CUDA environment"
