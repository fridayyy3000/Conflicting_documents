"""
Test script for GOV-RAG API
Run the API first with: python main.py
Then run this script: python test_api.py
"""
import os
import requests
import json

API_URL = "http://localhost:8000"
API_KEY = os.getenv("GOVRAG_API_KEY")


def auth_headers():
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers

def test_health():
    """Test health endpoint"""
    print("=" * 80)
    print("Testing Health Endpoint")
    print("=" * 80)
    response = requests.get(f"{API_URL}/health")
    print(json.dumps(response.json(), indent=2))
    print()

def test_single_query():
    """Test single query"""
    print("=" * 80)
    print("Testing Single Query")
    print("=" * 80)
    
    question = "What is the annual reimbursement ceiling for Tier-B employees at Velora Dynamics?"
    
    response = requests.post(
        f"{API_URL}/query",
        headers=auth_headers(),
        json={
            "question": question,
            "use_llm": True,
            "top_k": 8
        }
    )
    
    result = response.json()
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Source: {result['selected_source']}")
    print(f"Conflicts: {result['num_conflicts']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nTop 3 Sources:")
    for i, src in enumerate(result['top_sources'][:3], 1):
        print(f"  {i}. {src['filename']} (score: {src['final_score']:.3f})")
        print(f"     Status: {src['status']}, Claim: {src['claim']}")
    print()


def test_demo_query():
    """Test public demo endpoint"""
    print("=" * 80)
    print("Testing Demo Query")
    print("=" * 80)

    question = "What is the maximum defect rate allowed for Q4-certified suppliers at Meridian Forge?"
    response = requests.post(
        f"{API_URL}/demo_query",
        json={
            "question": question,
            "use_llm": True,
            "top_k": 8,
        },
    )
    result = response.json()
    print(f"Demo Answer: {result.get('answer')}")
    print(f"Demo Source: {result.get('selected_source')}")
    print()

def test_batch_query():
    """Test batch query"""
    print("=" * 80)
    print("Testing Batch Query")
    print("=" * 80)
    
    questions = [
        "What is the annual reimbursement ceiling for Tier-B employees at Velora Dynamics?",
        "What is the maximum monthly API quota for Enterprise-X tenants at Nexora Systems?",
        "How long must Tier-3 diagnostic records be retained at Arclume Health?"
    ]
    
    response = requests.post(
        f"{API_URL}/batch_query",
        headers=auth_headers(),
        params={
            "project_id": "project-79920195-9e86-44ea-8c9",
            "region": "us-central1"
        },
        json=questions
    )
    
    results = response.json()
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['question']}")
        print(f"    Answer: {result['answer']}")
        print(f"    Source: {result['selected_source']}")
    print()

if __name__ == "__main__":
    try:
        print("\n🚀 Starting GOV-RAG API Tests\n")
        
        test_health()
        test_single_query()
        test_demo_query()
        test_batch_query()
        
        print("✅ All tests completed!\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("   Make sure the API is running: python main.py")
    except Exception as e:
        print(f"❌ Error: {e}")
