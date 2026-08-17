$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonLauncher = Get-Command py -ErrorAction Stop
$VenvPath = Join-Path $ProjectRoot ".venv-yolox"
$SourcePath = Join-Path $ProjectRoot ".deps\YOLOX"
$PinnedCommit = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"

if (-not (Test-Path -LiteralPath $VenvPath)) {
    & $PythonLauncher.Source -3.11 -m venv $VenvPath
}

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip setuptools wheel
& $PythonExe -m pip install `
    torch==2.12.1 `
    torchvision==0.27.1 `
    --index-url https://download.pytorch.org/whl/cu130
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements\yolox-cu130.lock.txt")

if (-not (Test-Path -LiteralPath (Join-Path $SourcePath ".git"))) {
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git $SourcePath
}
git -C $SourcePath fetch --depth 1 origin $PinnedCommit
git -C $SourcePath checkout --detach $PinnedCommit
& $PythonExe -m pip install --no-deps --no-build-isolation -e $SourcePath
& $PythonExe -m pip install --no-build-isolation -e $ProjectRoot

& $PythonExe (Join-Path $PSScriptRoot "smoke_yolox.py")
