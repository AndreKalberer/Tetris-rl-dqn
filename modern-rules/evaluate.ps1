param([int[]]$Seeds=@(91000..91019),[string]$Output="runs\offline-eval-$(Get-Date -Format 'yyyyMMdd-HHmmss')")
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$savedPath = $env:PYTHONPATH
$env:PYTHONPATH = "$repo\modern-rules;$repo\old-dqn"
Push-Location -LiteralPath $repo
try {
    $pythonExe=Join-Path $repo '.venv\Scripts\python.exe'
    $evalArgs=@('-m','offline.evaluate','--beam','4','--output',$Output,'--seeds')
    $evalArgs+=@($Seeds | ForEach-Object { "$_" })
    & $pythonExe @evalArgs
    if ($LASTEXITCODE -ne 0) { throw 'Offline evaluation exited with an error.' }
} finally { Pop-Location; $env:PYTHONPATH = $savedPath }
