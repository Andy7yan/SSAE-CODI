param(
    [string]$PythonExecutable = "py",
    [string]$VenvPath = ".venv-stage1",
    [switch]$IncludeDevDependencies
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResolvedVenvPath = Join-Path $RepoRoot $VenvPath
$PythonPath = Join-Path $ResolvedVenvPath "Scripts\python.exe"

if (-not (Test-Path $ResolvedVenvPath)) {
    Write-Host "Creating virtual environment at $ResolvedVenvPath"
    & $PythonExecutable -3.12 -m venv $ResolvedVenvPath
}

if (-not (Test-Path $PythonPath)) {
    throw "Python executable was not found in $ResolvedVenvPath"
}

Write-Host "Upgrading packaging tools"
& $PythonPath -m pip install --upgrade pip setuptools wheel

$InstallTarget = "."
if ($IncludeDevDependencies) {
    $InstallTarget = ".[dev]"
}

Write-Host "Installing Stage 1 dependencies from $InstallTarget"
Push-Location $RepoRoot
try {
    & $PythonPath -m pip install -e $InstallTarget
}
finally {
    Pop-Location
}

Write-Host "Stage 1 environment is ready."
Write-Host "Python: $PythonPath"
