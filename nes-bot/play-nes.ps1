param([switch]$Bot, [int]$Seed = 0, [int]$StartLevel = 0, [string]$Checkpoint)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$savedPath = $env:PYTHONPATH
$env:PYTHONPATH = "$repo\nes-bot;$repo\old-dqn"
Push-Location $repo
try {
    $nesArguments = @('-m', 'nes.play', '--seed', "$Seed", '--start-level', "$StartLevel")
    if ($Bot) { $nesArguments += '--bot' }
    if ($Checkpoint) { $nesArguments += @('--checkpoint', $Checkpoint) }
    & .\.venv\Scripts\python.exe @nesArguments
    if ($LASTEXITCODE -ne 0) { throw "NES playback exited with code $LASTEXITCODE" }
} finally { Pop-Location; $env:PYTHONPATH = $savedPath }
