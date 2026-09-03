#!/usr/bin/env python3
"""
Evaluation script for GOV-RAG using Gemini embeddings.

Usage:
    export GOOGLE_API_KEY='your-api-key-here'
    python code/evaluate_gov_rag_gemini.py --pack easy
"""

import os
import sys
import argparse

# Import the original evaluation logic
from evaluate_gov_rag import (
    load_questions_for_inference,
    load_ground_truth,
    run_inference,
    evaluate_results,
    print_summary,
    save_results
)

# Import Gemini version of GOV-RAG
from gov_rag_gemini import GovRAGGemini

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GOV-RAG with Gemini embeddings'
    )
    parser.add_argument(
        '--pack',
        type=str,
        required=True,
        choices=['easy', 'hard'],
        help='Difficulty pack to evaluate'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='conflictbench_fictional_full',
        help='Base directory containing the dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_gemini',
        help='Directory to save results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output during inference'
    )
    parser.add_argument(
        '--project-id',
        type=str,
        default=None,
        help='GCP Project ID (or set GOOGLE_CLOUD_PROJECT env var)'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='us-central1',
        help='GCP region for Vertex AI (default: us-central1)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Google API key for AI Studio (alternative to project-id)'
    )
    
    args = parser.parse_args()
    
    # Check authentication
    project_id = args.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    
    if not project_id and not api_key:
        print("\nERROR: No authentication found!")
        print("\nOption 1 - Vertex AI with Project ID (Recommended):")
        print("  gcloud auth application-default login")
        print("  export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("  python code/evaluate_gov_rag_gemini.py --pack easy")
        print("\nOption 2 - AI Studio with API Key:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        print("  python code/evaluate_gov_rag_gemini.py --pack easy")
        print("\nGet API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Construct paths
    base_path = Path(args.base_dir)
    doc_dir = base_path / 'packs' / args.pack
    questions_csv = base_path / 'questions.csv'
    ground_truth_csv = base_path / 'ground_truth_manifest.csv'
    
    # Validate paths
    if not doc_dir.exists():
        print(f"ERROR: Document directory not found: {doc_dir}")
        sys.exit(1)
    if not questions_csv.exists():
        print(f"ERROR: Questions file not found: {questions_csv}")
        sys.exit(1)
    if not ground_truth_csv.exists():
        print(f"ERROR: Ground truth manifest not found: {ground_truth_csv}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"GOV-RAG EVALUATION (VERTEX AI / GEMINI)")
    print(f"{'='*80}")
    print(f"Pack: {args.pack}")
    print(f"Documents: {doc_dir}")
    print(f"Questions: {questions_csv}")
    if project_id:
        print(f"Auth: Vertex AI (Project: {project_id})")
    else:
        print(f"Auth: AI Studio (API Key)")
    print(f"{'='*80}\n")
    
    # Phase 1: Load questions for inference (NO GROUND TRUTH)
    questions = load_questions_for_inference(str(questions_csv))
    
    # Phase 2: Initialize GOV-RAG with Vertex AI or Gemini  
    # use_llm=True enables Gemini 2.5 Pro for final answer generation
    print(f"\nInitializing GOV-RAG...")
    rag = GovRAGGemini(
        str(doc_dir), 
        project_id=project_id, 
        region=args.region, 
        api_key=api_key,
        use_llm=True  # Enable LLM-based final resolution
    )
    
    # Phase 3: Run inference (NO GROUND TRUTH VISIBLE)
    inference_results = run_inference(rag, questions, verbose=args.verbose)
    
    # Phase 4: Load ground truth (ONLY AFTER INFERENCE)
    gold_data, doc_roles = load_ground_truth(
        str(questions_csv),
        str(ground_truth_csv),
        args.pack
    )
    
    # Phase 5: Evaluate predictions against ground truth
    evaluated, metrics = evaluate_results(inference_results, gold_data, doc_roles)
    
    # Phase 6: Print and save results
    print_summary(metrics)
    save_results(evaluated, metrics, args.output_dir, args.pack)
    
    print(f"\n✓ Evaluation complete for {args.pack} pack (Gemini).")
    print(f"Answer Accuracy: {metrics['answer_accuracy']:.2f}%")
    print(f"Gold Selection Rate: {metrics['gold_selection_rate']:.2f}%")
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
