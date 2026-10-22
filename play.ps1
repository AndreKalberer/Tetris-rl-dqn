param([switch]$Human,[int]$Seed=1001,[ValidateRange(1,120)][int]$Beam=4)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $pythonExe=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    $playArgs=@('-m','offline.play','--seed',"$Seed",'--beam',"$Beam")
    if ($Human) { $playArgs+='--human' }
    & $pythonExe @playArgs
    if ($LASTEXITCODE -ne 0) { throw 'Offline game exited with an error.' }
} finally { Pop-Location }
