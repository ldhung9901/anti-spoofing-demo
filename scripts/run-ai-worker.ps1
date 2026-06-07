$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\ai-worker"
if (!(Test-Path ".venv")) {
  py -3.11 -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
if (!$env:AI_WORKER_CONCURRENCY) { $env:AI_WORKER_CONCURRENCY = "4" }
if (!$env:AI_WORKER_QUEUE_LIMIT) { $env:AI_WORKER_QUEUE_LIMIT = "16" }
if (!$env:AI_WORKER_QUEUE_WAIT_MS) { $env:AI_WORKER_QUEUE_WAIT_MS = "25" }
if (!$env:FACE_MATCH_THRESHOLD) { $env:FACE_MATCH_THRESHOLD = "0.45" }
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
