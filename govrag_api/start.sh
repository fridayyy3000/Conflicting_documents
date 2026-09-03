#!/bin/bash

# GOV-RAG API Quick Start Script

echo "================================="
echo "GOV-RAG API Setup"
echo "================================="
echo ""

# Install dependencies in current environment
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt 2>&1 | grep -v "dependency conflicts" || true

# Check GCP auth
echo "Checking GCP authentication..."
if ! gcloud auth application-default print-access-token &>/dev/null; then
    echo "⚠️  GCP authentication not found"
    echo "Please run:"
    echo "  gcloud auth application-default login"
    echo "  gcloud auth application-default set-quota-project project-79920195-9e86-44ea-8c9"
    echo ""
    read -p "Press Enter after authentication is complete..."
fi

# Set environment variable
export GOOGLE_CLOUD_PROJECT='project-79920195-9e86-44ea-8c9'

echo ""
echo "================================="
echo "Starting GOV-RAG API..."
echo "================================="
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo "Test with: python test_api.py"
echo ""

# Run the API
python main.py
