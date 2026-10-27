$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    if (-not (Test-Path -LiteralPath .\.venv\Scripts\python.exe)) {
        & py -3.13 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.13 environment' }
    }
    & .\.venv\Scripts\python.exe -m pip install -r (Join-Path $PSScriptRoot 'requirements-nes-dev.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Could not install NES dependencies' }
    & .\.venv\Scripts\python.exe -m pytest -q nes-bot/tests/test_nes.py
    if ($LASTEXITCODE -ne 0) { throw 'NES tests failed' }
} finally { Pop-Location }
