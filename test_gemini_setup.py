#!/usr/bin/env python3
"""
Test Gemini API connection and setup.
"""

import os
import sys

def test_gemini_import():
    """Test if google-generativeai is installed."""
    print("Testing imports...")
    try:
        import google.generativeai as genai
        print("  ✓ google-generativeai installed")
        return True
    except ImportError:
        print("  ✗ google-generativeai NOT installed")
        print("  Run: pip install google-generativeai")
        return False


def test_api_key():
    """Test if API key is set."""
    print("\nTesting API key...")
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if api_key:
        # Show first/last 4 chars
        masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        print(f"  ✓ GOOGLE_API_KEY is set: {masked}")
        return api_key
    else:
        print("  ✗ GOOGLE_API_KEY is NOT set")
        print("\n  Set it with:")
        print("    export GOOGLE_API_KEY='your-api-key-here'")
        print("\n  Get API key from:")
        print("    https://makersuite.google.com/app/apikey")
        return None


def test_gemini_connection(api_key):
    """Test actual API connection."""
    print("\nTesting Gemini API connection...")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        
        # Test embedding with simple text
        result = genai.embed_content(
            model='models/text-embedding-004',
            content='test connection'
        )
        
        embedding = result['embedding']
        dim = len(embedding)
        
        print(f"  ✓ Successfully connected to Gemini API")
        print(f"  ✓ Embedding dimension: {dim}")
        print(f"  ✓ Model: text-embedding-004")
        return True
        
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("\n  Troubleshooting:")
        print("    1. Check your API key is valid")
        print("    2. Verify internet connection")
        print("    3. Check if you're behind a proxy")
        print("    4. Try regenerating your API key")
        return False


def test_batch_embedding(api_key):
    """Test batch embedding."""
    print("\nTesting batch embedding...")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        
        texts = [
            "What is the maximum defect rate?",
            "What is the ceiling for reimbursement?",
            "How long must records be retained?"
        ]
        
        result = genai.embed_content(
            model='models/text-embedding-004',
            content=texts
        )
        
        embeddings = result['embedding']
        
        print(f"  ✓ Batch embedding successful")
        print(f"  ✓ Embedded {len(texts)} texts")
        print(f"  ✓ Each embedding: {len(embeddings[0])} dimensions")
        return True
        
    except Exception as e:
        print(f"  ✗ Batch embedding failed: {e}")
        return False


def main():
    print("="*80)
    print("GEMINI API SETUP TEST")
    print("="*80 + "\n")
    
    results = []
    
    # Test imports
    if not test_gemini_import():
        print("\n✗ Setup incomplete. Install google-generativeai first.")
        print("  pip install google-generativeai")
        sys.exit(1)
    results.append(True)
    
    # Test API key
    api_key = test_api_key()
    if not api_key:
        print("\n✗ Setup incomplete. Set GOOGLE_API_KEY environment variable.")
        sys.exit(1)
    results.append(True)
    
    # Test connection
    if not test_gemini_connection(api_key):
        print("\n✗ API connection failed. Check troubleshooting steps above.")
        sys.exit(1)
    results.append(True)
    
    # Test batch
    if not test_batch_embedding(api_key):
        print("\n⚠ Batch embedding failed, but basic connection works.")
        print("  GOV-RAG should still work (may be slower).")
    else:
        results.append(True)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all(results):
        print("\n✓ All tests passed!")
        print("\n✓ Gemini API is ready for GOV-RAG")
        print("\nNext steps:")
        print("  1. Run evaluation:")
        print("     python code/evaluate_gov_rag_gemini.py --pack easy")
        print("\n  2. Or try interactive mode:")
        print("     python code/gov_rag_gemini.py conflictbench_fictional_full/packs/easy")
    else:
        print("\n⚠ Some tests failed, but basic functionality works")
        print("  You can proceed with GOV-RAG evaluation")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
