param([string]$Output="runs\research-repeat-$(Get-Date -Format 'yyyyMMdd-HHmmss')")
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $pythonExe=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    foreach ($name in @('seed0-60000','seed1-60000','seed2-60000')) {
        $model=Join-Path $PSScriptRoot "models\$name.pt"
        & $pythonExe -m dqn.benchmark_lookahead --checkpoint $model --seeds 70000 70001 70002 70003 70004 --cap 5000 --beams 1 4 --output "$Output\$name"
        if ($LASTEXITCODE -ne 0) { throw "Research evaluation failed for $name" }
    }
} finally { Pop-Location }
