$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        & py -3.13 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create a Python 3.13 virtual environment.' }
    }
    & $pythonExe -m pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $pythonExe -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
} finally { Pop-Location }
