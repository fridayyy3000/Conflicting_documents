#!/usr/bin/env python3
"""
Test LLM Integration for GOV-RAG

This script tests that Gemini 2.5 Pro is properly integrated
for final evidence resolution.

Usage:
    export GOOGLE_CLOUD_PROJECT='your-project-id'
    gcloud auth application-default login
    python test_llm_integration.py
"""

import os
import sys

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        print("  ✓ vertexai installed")
    except ImportError:
        print("  ✗ vertexai not installed")
        print("  Run: pip install google-cloud-aiplatform")
        return False
    
    try:
        from gov_rag_gemini import GovRAGGemini, GeminiGenerator, SYSTEM_PROMPT
        print("  ✓ gov_rag_gemini imports successful")
    except ImportError as e:
        print(f"  ✗ gov_rag_gemini import failed: {e}")
        return False
    
    return True


def test_prompt():
    """Test that the system prompt is properly defined."""
    print("\nTesting system prompt...")
    
    from gov_rag_gemini import SYSTEM_PROMPT
    
    if not SYSTEM_PROMPT:
        print("  ✗ SYSTEM_PROMPT is empty")
        return False
    
    if len(SYSTEM_PROMPT) < 500:
        print("  ✗ SYSTEM_PROMPT seems too short")
        return False
    
    required_phrases = [
        "evidence-resolution",
        "conflicting documents",
        "authority",
        "scope",
        "JSON"
    ]
    
    for phrase in required_phrases:
        if phrase.lower() not in SYSTEM_PROMPT.lower():
            print(f"  ✗ SYSTEM_PROMPT missing key phrase: {phrase}")
            return False
    
    print(f"  ✓ SYSTEM_PROMPT properly defined ({len(SYSTEM_PROMPT)} characters)")
    return True


def test_generator_init():
    """Test that GeminiGenerator can be initialized."""
    print("\nTesting GeminiGenerator initialization...")
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    if not project_id:
        print("  ✗ GOOGLE_CLOUD_PROJECT not set")
        print("  Run: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        return False
    
    try:
        from gov_rag_gemini import GeminiGenerator
        
        generator = GeminiGenerator(project_id=project_id)
        print(f"  ✓ GeminiGenerator initialized")
        print(f"  ✓ Project: {project_id}")
        print(f"  ✓ Using Vertex AI: {generator.use_vertex}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GeminiGenerator initialization failed: {e}")
        return False


def test_rag_with_llm():
    """Test that GovRAGGemini can be initialized with use_llm=True."""
    print("\nTesting GovRAGGemini with LLM enabled...")
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    doc_dir = "conflictbench_fictional_full/packs/easy"
    
    if not os.path.exists(doc_dir):
        print(f"  ✗ Document directory not found: {doc_dir}")
        print("  Make sure you're running from the correct directory")
        return False
    
    try:
        from gov_rag_gemini import GovRAGGemini
        
        print(f"  Initializing with doc_dir={doc_dir}, use_llm=True...")
        rag = GovRAGGemini(
            doc_dir, 
            project_id=project_id,
            use_llm=True
        )
        
        print(f"  ✓ GovRAGGemini initialized")
        print(f"  ✓ use_llm: {rag.use_llm}")
        print(f"  ✓ Generator: {type(rag.generator).__name__ if rag.generator else 'None'}")
        print(f"  ✓ Documents loaded: {len(rag.documents)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GovRAGGemini initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_query():
    """Test a single query with LLM resolution."""
    print("\nTesting single query with LLM resolution...")
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    doc_dir = "conflictbench_fictional_full/packs/easy"
    
    # Check if questions.csv exists to get a real question
    questions_csv = "conflictbench_fictional_full/questions.csv"
    
    if not os.path.exists(questions_csv):
        print(f"  ✗ Questions file not found: {questions_csv}")
        return False
    
    try:
        import csv
        
        # Read first question
        with open(questions_csv, 'r') as f:
            reader = csv.DictReader(f)
            first_question = next(reader)
            question_text = first_question['question']
            question_id = first_question['question_id']
        
        print(f"\n  Question ID: {question_id}")
        print(f"  Question: {question_text}")
        
        from gov_rag_gemini import GovRAGGemini
        
        rag = GovRAGGemini(
            doc_dir, 
            project_id=project_id,
            use_llm=True
        )
        
        print(f"\n  Running query...")
        result = rag.query(question_text)
        
        print(f"\n  ={'='*60}")
        print(f"  RESULT")
        print(f"  ={'='*60}")
        print(f"  Answer: {result.get('answer')}")
        print(f"  Source: {result.get('source')}")
        print(f"  Method: {result.get('method')}")
        
        if result.get('method') == 'llm':
            print(f"  Reason: {result.get('reason')}")
            print(f"  Conflict Detected: {result.get('conflict_detected')}")
            print(f"  Confidence: {result.get('confidence')}")
            print(f"\n  ✓ LLM resolution successful!")
            return True
        else:
            print(f"\n  ⚠ Fell back to rule-based resolution")
            print(f"  This might indicate an LLM generation issue")
            return False
        
    except Exception as e:
        print(f"  ✗ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("GOV-RAG LLM INTEGRATION TEST")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("System Prompt", test_prompt),
        ("Generator Initialization", test_generator_init),
        ("RAG with LLM", test_rag_with_llm),
        ("Single Query", test_single_query),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:10} {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! LLM integration is working correctly.")
        print("\nNext steps:")
        print("  python code/evaluate_gov_rag_gemini.py --pack easy")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
