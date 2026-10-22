param(
    [string]$EnvironmentPath = (Join-Path $PSScriptRoot '.venv-original'),
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
$EnvironmentPath = [System.IO.Path]::GetFullPath($EnvironmentPath)
$tetrisPython = Join-Path $EnvironmentPath 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $tetrisPython)) {
    & py -3.13 -m venv --without-pip $EnvironmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated Python 3.13 environment' }
}
$tetrisBootstrap = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $tetrisBootstrap) {
    & $tetrisBootstrap -m pip --python $tetrisPython install pip==26.2.2 -r (Join-Path $PSScriptRoot 'requirements-nes-original-lock-windows.txt')
} else {
    & py -3.13 -m pip --python $tetrisPython install pip==26.2.2 -r (Join-Path $PSScriptRoot 'requirements-nes-original-lock-windows.txt')
}
if ($LASTEXITCODE -ne 0) { throw 'Original-game dependencies failed to install; a working bootstrap pip is required' }
if (-not $SkipTests) {
    Push-Location $PSScriptRoot
    try {
        & $tetrisPython -m unittest discover -s tests -p test_nes_original.py -v
        if ($LASTEXITCODE -ne 0) { throw 'Original-game validation failed' }
    } finally { Pop-Location }
}
Write-Host "Original-game Python: $tetrisPython"
