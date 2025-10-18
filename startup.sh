#!/usr/bin/env bash
# Startup wrapper for Azure Web Apps
# Uses $PORT if provided by environment, otherwise falls back to 8000
PORT=${PORT:-8000}
echo "Starting uvicorn on 0.0.0.0:${PORT}"
exec uvicorn src.api_server:app --host 0.0.0.0 --port ${PORT}
