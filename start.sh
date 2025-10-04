#!/usr/bin/env bash
set -euo pipefail

# Load .env if present (for local/dev). On Render, env vars come from dashboard.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH=.

# Ensure config.yaml is accessible to the analysis script
cp -f src/egx_intraday_project/config.yaml config.yaml
mkdir -p configs
cp -f src/egx_intraday_project/config.yaml configs/config.yaml

# Start the analysis loop (paper mode)
python src/egx_intraday_project/egx_intraday_enhanced.py &

# Start webhook (FastAPI)
exec uvicorn src.egx_intraday_enhanced.egx.webhook.app:app --host 0.0.0.0 --port "${PORT:-8000}"
