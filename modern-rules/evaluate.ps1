param([int[]]$Seeds=@(91000..91019),[string]$Output="runs\offline-eval-$(Get-Date -Format 'yyyyMMdd-HHmmss')")
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $pythonExe=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    $evalArgs=@('-m','offline.evaluate','--beam','4','--output',$Output,'--seeds')
    $evalArgs+=@($Seeds | ForEach-Object { "$_" })
    & $pythonExe @evalArgs
    if ($LASTEXITCODE -ne 0) { throw 'Offline evaluation exited with an error.' }
} finally { Pop-Location }
