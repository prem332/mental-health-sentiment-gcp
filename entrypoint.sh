#!/bin/sh
set -e
echo "🔽 Downloading model artifacts from GCS..."
python scripts/download_models.py
echo "🚀 Starting Flask app with Gunicorn..."
exec gunicorn \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers 2 --threads 4 --timeout 120 \
  --access-logfile - --error-logfile - \
  app.app:app
