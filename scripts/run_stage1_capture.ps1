param(
    [string]$VenvPath = ".venv-stage1",
    [string]$ConfigPath = "configs/codi_gpt2_stage1.yaml"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path (Join-Path $RepoRoot $VenvPath) "Scripts\python.exe"

if (-not (Test-Path $PythonPath)) {
    throw "Virtual environment not found. Run .\scripts\setup_stage1.ps1 first."
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"

Push-Location $RepoRoot
try {
    & $PythonPath -m stage1.run_inference `
        --config $ConfigPath `
        --capture-hidden `
        --capture-mode "per-latent-step" `
        --run-name "stage1_capture"
}
finally {
    Pop-Location
}
