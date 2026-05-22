#!/bin/bash
echo "Stopping any existing backend server..."
lsof -ti :8000 | xargs kill -9 2>/dev/null
echo "Starting backend in Deep Learning mode..."
source venv/bin/activate
cd backend
uvicorn main:app --reload --port 8000
