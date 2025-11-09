#!/bin/sh
(cd frontend/ && npm install && npm run build)
backend/venv/bin/pip install -r backend/requirements.txt
