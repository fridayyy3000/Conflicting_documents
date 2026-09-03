#!/bin/bash
# Quick run script - assumes dependencies already installed

export GOOGLE_CLOUD_PROJECT='project-79920195-9e86-44ea-8c9'

echo "🚀 Starting GOV-RAG API..."
echo "   Endpoint: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

python main.py
