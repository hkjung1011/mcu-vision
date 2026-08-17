$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([int]$ExitCode, [string]$Operation) {
    if ($ExitCode -ne 0) {
        throw "$Operation failed with exit code $ExitCode."
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonLauncher = Get-Command py -ErrorAction Stop
$VenvPath = Join-Path $ProjectRoot ".venv-yolox"
$SourcePath = Join-Path $ProjectRoot ".deps\YOLOX"
$PinnedCommit = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"

if (-not (Test-Path -LiteralPath $VenvPath)) {
    & $PythonLauncher.Source -3.11 -m venv $VenvPath
    Assert-NativeSuccess $LASTEXITCODE "Create YOLOX virtual environment"
}

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip setuptools wheel
Assert-NativeSuccess $LASTEXITCODE "Upgrade YOLOX build tools"
& $PythonExe -m pip install `
    torch==2.12.1 `
    torchvision==0.27.1 `
    --index-url https://download.pytorch.org/whl/cu130
Assert-NativeSuccess $LASTEXITCODE "Install YOLOX PyTorch CUDA wheels"
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements\yolox-cu130.lock.txt")
Assert-NativeSuccess $LASTEXITCODE "Install YOLOX lock file"

if (-not (Test-Path -LiteralPath (Join-Path $SourcePath ".git"))) {
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git $SourcePath
    Assert-NativeSuccess $LASTEXITCODE "Clone YOLOX source"
}
git -C $SourcePath fetch --depth 1 origin $PinnedCommit
Assert-NativeSuccess $LASTEXITCODE "Fetch pinned YOLOX commit"
git -C $SourcePath checkout --detach $PinnedCommit
Assert-NativeSuccess $LASTEXITCODE "Checkout pinned YOLOX commit"
& $PythonExe -m pip install --no-deps --no-build-isolation -e $SourcePath
Assert-NativeSuccess $LASTEXITCODE "Install pinned YOLOX source"
& $PythonExe -m pip install --no-build-isolation -e $ProjectRoot
Assert-NativeSuccess $LASTEXITCODE "Install project in YOLOX environment"

& $PythonExe (Join-Path $PSScriptRoot "smoke_yolox.py")
Assert-NativeSuccess $LASTEXITCODE "Run YOLOX CUDA smoke"
