#!/usr/bin/env python3
"""
Test Vertex AI connection with project ID.
"""

import os
import sys

def test_vertex_imports():
    """Test if vertexai is installed."""
    print("Testing Vertex AI imports...")
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        print("  ✓ google-cloud-aiplatform installed")
        return True
    except ImportError:
        print("  ✗ google-cloud-aiplatform NOT installed")
        print("  Run: pip install google-cloud-aiplatform")
        return False


def test_project_id():
    """Test if project ID is set."""
    print("\nTesting project ID...")
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    if project_id:
        print(f"  ✓ GOOGLE_CLOUD_PROJECT is set: {project_id}")
        return project_id
    else:
        print("  ✗ GOOGLE_CLOUD_PROJECT is NOT set")
        print("\n  Set it with:")
        print("    export GOOGLE_CLOUD_PROJECT='your-project-id'")
        return None


def test_gcloud_auth():
    """Test if gcloud authentication is configured."""
    print("\nTesting gcloud authentication...")
    
    import subprocess
    
    try:
        # Check if gcloud is installed
        result = subprocess.run(
            ['gcloud', 'auth', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            active_accounts = [line for line in result.stdout.split('\n') if '*' in line]
            if active_accounts:
                account = active_accounts[0].strip().replace('*', '').strip()
                print(f"  ✓ Authenticated as: {account}")
                return True
            else:
                print("  ⚠ gcloud installed but no active account")
                print("  Run: gcloud auth application-default login")
                return False
        else:
            print("  ⚠ gcloud command failed")
            return False
            
    except FileNotFoundError:
        print("  ✗ gcloud CLI not installed")
        print("\n  Install with:")
        print("    brew install google-cloud-sdk  # macOS")
        print("    # Or download from: https://cloud.google.com/sdk/docs/install")
        return False
    except Exception as e:
        print(f"  ⚠ Could not check gcloud auth: {e}")
        return False


def test_vertex_connection(project_id):
    """Test actual Vertex AI connection."""
    print("\nTesting Vertex AI connection...")
    
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location="us-central1")
        
        print("  ✓ Vertex AI initialized")
        print(f"    Project: {project_id}")
        print(f"    Region: us-central1")
        
        # Try to load embedding model
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        print("  ✓ Embedding model loaded: text-embedding-004")
        
        return True, model
        
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("\n  Troubleshooting:")
        print("    1. Run: gcloud auth application-default login")
        print("    2. Verify project ID is correct")
        print("    3. Enable Vertex AI API:")
        print("       gcloud services enable aiplatform.googleapis.com")
        return False, None


def test_embedding(model):
    """Test actual embedding generation."""
    print("\nTesting embedding generation...")
    
    try:
        texts = ["What is the maximum defect rate?"]
        
        embeddings = model.get_embeddings(texts)
        
        if embeddings and len(embeddings) > 0:
            dim = len(embeddings[0].values)
            print(f"  ✓ Successfully generated embedding")
            print(f"  ✓ Dimension: {dim}")
            print(f"  ✓ Sample values: {embeddings[0].values[:5]}...")
            return True
        else:
            print("  ✗ No embeddings returned")
            return False
        
    except Exception as e:
        print(f"  ✗ Embedding generation failed: {e}")
        return False


def main():
    print("="*80)
    print("VERTEX AI SETUP TEST (Project ID Authentication)")
    print("="*80 + "\n")
    
    results = []
    
    # Test 1: Imports
    if not test_vertex_imports():
        print("\n✗ Setup incomplete. Install google-cloud-aiplatform first.")
        print("  pip install google-cloud-aiplatform")
        sys.exit(1)
    results.append(True)
    
    # Test 2: Project ID
    project_id = test_project_id()
    if not project_id:
        print("\n✗ Setup incomplete. Set GOOGLE_CLOUD_PROJECT environment variable.")
        print("  export GOOGLE_CLOUD_PROJECT='your-project-id'")
        sys.exit(1)
    results.append(True)
    
    # Test 3: gcloud auth
    auth_ok = test_gcloud_auth()
    if not auth_ok:
        print("\n⚠ Authentication may not be configured.")
        print("  Run: gcloud auth application-default login")
    results.append(auth_ok)
    
    # Test 4: Vertex AI connection
    connected, model = test_vertex_connection(project_id)
    if not connected:
        print("\n✗ Vertex AI connection failed. Check troubleshooting steps above.")
        sys.exit(1)
    results.append(True)
    
    # Test 5: Embedding
    if model:
        embed_ok = test_embedding(model)
        results.append(embed_ok)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all(results):
        print("\n✓ All tests passed!")
        print("\n✓ Vertex AI is ready for GOV-RAG")
        print("\nNext steps:")
        print("  1. Run evaluation:")
        print("     python code/evaluate_gov_rag_gemini.py --pack easy")
        print("\n  2. Or try interactive mode:")
        print("     python code/gov_rag_gemini.py conflictbench_fictional_full/packs/easy")
    else:
        print("\n⚠ Some tests failed")
        print("  Review the errors above and follow troubleshooting steps")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
