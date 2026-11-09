#!/usr/bin/env bash
# Linux/macOS: starts backend (8000) and frontend (5173)
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
if [ ! -f ml/saved_models/breast_cancer_pipeline.joblib ]; then
  python ml/train_model.py && python ml/evaluate_model.py
fi
uvicorn backend.main:app --reload &
cd frontend && { [ -d node_modules ] || npm install; } && npm run dev
