@echo off
rem Watch a recorded bot game in Mesen.  Usage:  tools\watch.cmd [speed%] [record.json]
rem   speed: 100 normal ... 1000, 0 = as fast as possible (about 3.9x on this PC while the panel runs).
rem   Live: Options > Speed, keys = and -, F9, or hold Tab. Default record: the verified rebirth game.
setlocal
set SPEED=%1
if "%SPEED%"=="" set SPEED=500
set RECORD=%2
if "%RECORD%"=="" set RECORD=runs\nes-255-v8b\rebirth\seed19211\cases\seed-19211.json
cd /d "%~dp0.."
"..\nes-original-env\.venv\Scripts\python.exe" tools\watch_replay.py "%RECORD%" --speed %SPEED%
