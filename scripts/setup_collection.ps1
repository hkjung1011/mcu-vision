$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([int]$ExitCode, [string]$Operation) {
    if ($ExitCode -ne 0) {
        throw "$Operation failed with exit code $ExitCode."
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPath = Join-Path $ProjectRoot ".venv-collect"

$Python311 = & py -3.11 -c "import sys; print(sys.executable)"
Assert-NativeSuccess $LASTEXITCODE "Locate Python 3.11"
if (-not $Python311) {
    throw "Python 3.11 was not found. Install Python.Python.3.11 first."
}

if (-not (Test-Path -Path $EnvironmentPath)) {
    & py -3.11 -m venv $EnvironmentPath
    Assert-NativeSuccess $LASTEXITCODE "Create collection virtual environment"
}

$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
& $EnvironmentPython -m pip install --upgrade pip
Assert-NativeSuccess $LASTEXITCODE "Upgrade collection pip"
& $EnvironmentPython -m pip install -r (Join-Path $ProjectRoot "requirements\collection.lock.txt")
Assert-NativeSuccess $LASTEXITCODE "Install collection lock file"
& $EnvironmentPython -m pip install --no-deps --editable "$ProjectRoot[dev]"
Assert-NativeSuccess $LASTEXITCODE "Install project in collection environment"
& $EnvironmentPython -m pytest $ProjectRoot
Assert-NativeSuccess $LASTEXITCODE "Run collection test suite"

Write-Host "Collection environment is ready: $EnvironmentPath"
Write-Host "Activate with: .\.venv-collect\Scripts\Activate.ps1"
