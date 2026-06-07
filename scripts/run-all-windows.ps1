$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\.."
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$root\scripts\run-ai-worker.ps1"
Start-Sleep -Seconds 4
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$root\scripts\run-dotnet-api.ps1"
Write-Host "Open: http://localhost:5088"
