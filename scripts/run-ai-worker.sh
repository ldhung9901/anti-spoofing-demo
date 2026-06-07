#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../ai-worker"
python3.11 -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
export AI_WORKER_CONCURRENCY="${AI_WORKER_CONCURRENCY:-4}"
export AI_WORKER_QUEUE_LIMIT="${AI_WORKER_QUEUE_LIMIT:-16}"
export AI_WORKER_QUEUE_WAIT_MS="${AI_WORKER_QUEUE_WAIT_MS:-25}"
export FACE_MATCH_THRESHOLD="${FACE_MATCH_THRESHOLD:-0.45}"
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
