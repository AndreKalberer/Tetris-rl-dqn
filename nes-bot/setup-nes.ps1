$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath .\.venv\Scripts\python.exe)) {
        & py -3.13 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.13 environment' }
    }
    & .\.venv\Scripts\python.exe -m pip install -r requirements-nes-dev.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install NES dependencies' }
    & .\.venv\Scripts\python.exe -m pytest -q tests/test_nes.py
    if ($LASTEXITCODE -ne 0) { throw 'NES tests failed' }
} finally { Pop-Location }
