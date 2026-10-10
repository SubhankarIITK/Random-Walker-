@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0showcase.ps1"
if errorlevel 1 pause
