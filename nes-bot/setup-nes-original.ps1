param(
    [string]$EnvironmentPath = (Join-Path $PSScriptRoot '.venv-original'),
    [switch]$Learning,
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
$tetrisRequirements = Join-Path $PSScriptRoot $(if ($Learning) { 'requirements-nes-original-training.txt' } else { 'requirements-nes-original-lock-windows.txt' })
if (Test-Path -LiteralPath $tetrisBootstrap) {
    & $tetrisBootstrap -m pip --python $tetrisPython install pip==26.2.2 -r $tetrisRequirements
} else {
    & py -3.13 -m pip --python $tetrisPython install pip==26.2.2 -r $tetrisRequirements
}
if ($LASTEXITCODE -ne 0) { throw 'Original-game dependencies failed to install; a working bootstrap pip is required' }
if (-not $SkipTests) {
    Push-Location $PSScriptRoot
    try {
        & $tetrisPython -m unittest discover -s tests -p test_nes_original.py -v
        if ($LASTEXITCODE -ne 0) { throw 'Original-game validation failed' }
        if ($Learning) {
            & $tetrisPython -m unittest discover -s tests -p test_nes_original_learning.py -v
            if ($LASTEXITCODE -ne 0) { throw 'Original-game learning validation failed' }
        }
    } finally { Pop-Location }
}
Write-Host "Original-game Python: $tetrisPython"
