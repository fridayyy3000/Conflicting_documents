#\!/bin/bash

# Fix GCP Authentication for Vertex AI

PROJECT_ID="project-79920195-9e86-44ea-8c9"

echo "Setting up GCP authentication..."
echo ""

# 1. Set environment variable
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
echo "✓ Set GOOGLE_CLOUD_PROJECT=$PROJECT_ID"

# 2. Set quota project
echo ""
echo "Setting quota project..."
gcloud auth application-default set-quota-project "$PROJECT_ID"

# 3. Re-authenticate
echo ""
echo "Re-authenticating (will open browser)..."
gcloud auth application-default login

# 4. Verify
echo ""
echo "Verifying authentication..."
gcloud auth application-default print-access-token > /dev/null && echo "✓ Authentication successful"

echo ""
echo "================================================"
echo "Setup complete\! Now run:"
echo ""
echo "  export GOOGLE_CLOUD_PROJECT='$PROJECT_ID'"
echo "  python test_vertex_setup.py"
echo "================================================"
