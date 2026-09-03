#!/usr/bin/env python3
"""
Simple test script to verify GOV-RAG setup.
Tests each component without requiring model download.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if required packages are installed."""
    print("Testing imports...")
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - run: pip install numpy")
        return False
    
    try:
        import sentence_transformers
        print("  ✓ sentence-transformers")
    except ImportError:
        print("  ✗ sentence-transformers - run: pip install sentence-transformers")
        return False
    
    return True


def test_paths():
    """Test if dataset paths exist."""
    print("\nTesting dataset paths...")
    
    base = Path("conflictbench_fictional_full")
    
    paths_to_check = [
        base / "packs" / "easy",
        base / "packs" / "hard",
        base / "questions.csv",
        base / "ground_truth_manifest.csv"
    ]
    
    all_exist = True
    for path in paths_to_check:
        if path.exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_document_count():
    """Test if documents are present."""
    print("\nTesting document counts...")
    
    easy_path = Path("conflictbench_fictional_full/packs/easy")
    hard_path = Path("conflictbench_fictional_full/packs/hard")
    
    if easy_path.exists():
        easy_docs = list(easy_path.glob("*.md"))
        print(f"  Easy pack: {len(easy_docs)} documents")
        if len(easy_docs) == 180:  # 15 questions × 12 docs
            print("    ✓ Expected count (15 questions × 12 docs)")
        else:
            print(f"    ! Expected 180 documents, found {len(easy_docs)}")
    
    if hard_path.exists():
        hard_docs = list(hard_path.glob("*.md"))
        print(f"  Hard pack: {len(hard_docs)} documents")
        if len(hard_docs) == 300:  # 15 questions × 20 docs
            print("    ✓ Expected count (15 questions × 20 docs)")
        else:
            print(f"    ! Expected 300 documents, found {len(hard_docs)}")


def test_code_files():
    """Test if code files exist."""
    print("\nTesting code files...")
    
    code_files = [
        "code/gov_rag.py",
        "code/evaluate_gov_rag.py"
    ]
    
    all_exist = True
    for file in code_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_questions_csv():
    """Test if questions.csv is readable."""
    print("\nTesting questions.csv format...")
    
    import csv
    
    try:
        with open("conflictbench_fictional_full/questions.csv", 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            print(f"  ✓ {len(rows)} questions found")
            
            required_cols = ['question_id', 'question', 'gold_answer', 
                           'easy_gold_document', 'hard_gold_document']
            
            if all(col in rows[0] for col in required_cols):
                print("  ✓ All required columns present")
            else:
                print("  ✗ Missing required columns")
                return False
            
            # Show sample question
            if rows:
                sample = rows[0]
                print(f"\n  Sample question:")
                print(f"    ID: {sample['question_id']}")
                print(f"    Q:  {sample['question'][:60]}...")
                print(f"    A:  {sample['gold_answer']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading questions.csv: {e}")
        return False


def main():
    print("="*80)
    print("GOV-RAG SETUP VERIFICATION")
    print("="*80 + "\n")
    
    # Change to repository root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Paths", test_paths()))
    results.append(("Code Files", test_code_files()))
    results.append(("Questions CSV", test_questions_csv()))
    test_document_count()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = all(r[1] for r in results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10} {name}")
    
    if all_passed:
        print("\n✓ All checks passed!")
        print("\nNext step: Download embedding model")
        print("Run: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')\"")
        print("\nOr run evaluation directly:")
        print("python code/evaluate_gov_rag.py --pack easy")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
