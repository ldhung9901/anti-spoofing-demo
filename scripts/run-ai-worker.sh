#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../ai-worker"
python3.11 -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
