$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonLauncher = Get-Command py -ErrorAction Stop
$VenvPath = Join-Path $ProjectRoot ".venv-yolo11"

if (-not (Test-Path -LiteralPath $VenvPath)) {
    & $PythonLauncher.Source -3.11 -m venv $VenvPath
}

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip setuptools wheel
& $PythonExe -m pip install `
    torch==2.12.1 `
    torchvision==0.27.1 `
    --index-url https://download.pytorch.org/whl/cu130
& $PythonExe -m pip install "ultralytics>=8.4,<9"

& $PythonExe -c "import json, torch, ultralytics; print(json.dumps({'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, ensure_ascii=False))"
