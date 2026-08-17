$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPath = Join-Path $ProjectRoot ".venv-collect"

$Python311 = & py -3.11 -c "import sys; print(sys.executable)"
if (-not $Python311) {
    throw "Python 3.11 was not found. Install Python.Python.3.11 first."
}

if (-not (Test-Path -Path $EnvironmentPath)) {
    & py -3.11 -m venv $EnvironmentPath
}

$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
& $EnvironmentPython -m pip install --upgrade pip
& $EnvironmentPython -m pip install -r (Join-Path $ProjectRoot "requirements\collection.lock.txt")
& $EnvironmentPython -m pip install --no-deps --editable "$ProjectRoot[dev]"
& $EnvironmentPython -m pytest $ProjectRoot

Write-Host "Collection environment is ready: $EnvironmentPath"
Write-Host "Activate with: .\.venv-collect\Scripts\Activate.ps1"
