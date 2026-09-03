#!/usr/bin/env python3
"""
Evaluation script for GOV-RAG on ConflictBench Fictional dataset.

This script ensures strict separation between inference and evaluation:
- During inference: only questions are visible
- During evaluation: predictions are compared against ground truth

Usage:
    python code/evaluate_gov_rag.py --pack easy
    python code/evaluate_gov_rag.py --pack hard
"""

import os
import sys
import argparse
import json
import csv
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# Import GOV-RAG components
from gov_rag import GovRAG, normalize_claim


def load_questions_for_inference(questions_csv_path):
    """
    Load ONLY question text and IDs for inference.
    Does NOT load gold answers or gold documents.
    """
    questions = []
    
    with open(questions_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'question_id': row['question_id'],
                'question': row['question']
            })
    
    print(f"Loaded {len(questions)} questions for inference (no ground truth visible)")
    return questions


def load_ground_truth(questions_csv_path, ground_truth_manifest_path, pack):
    """
    Load ground truth AFTER inference is complete.
    This function is ONLY called during evaluation phase.
    """
    # Load answers and gold documents from questions.csv
    gold_data = {}
    
    with open(questions_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row['question_id']
            gold_doc_col = f'{pack}_gold_document'
            
            gold_data[qid] = {
                'gold_answer': row['gold_answer'],
                'gold_document': row[gold_doc_col],
                'question': row['question'],
                'dominant_conflicting_answer': row.get('dominant_conflicting_answer', '')
            }
    
    # Load document roles from ground truth manifest
    doc_roles = {}
    
    with open(ground_truth_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['difficulty'] == pack:
                key = (row['question_id'], row['document'])
                doc_roles[key] = {
                    'role': row['role'],
                    'variant': row['variant'],
                    'answer_supported': row.get('answer_supported', '')
                }
    
    print(f"Loaded ground truth for {len(gold_data)} questions")
    return gold_data, doc_roles


def run_inference(rag, questions, verbose=False):
    """
    Run GOV-RAG inference on all questions.
    No ground truth is visible during this phase.
    """
    results = []
    
    print(f"\n{'='*80}")
    print("STARTING INFERENCE PHASE")
    print(f"{'='*80}\n")
    
    for i, q in enumerate(questions, 1):
        qid = q['question_id']
        question = q['question']
        
        print(f"\n[{i}/{len(questions)}] {qid}: {question}")
        print("-" * 80)
        
        try:
            result = rag.query(question)
            
            results.append({
                'question_id': qid,
                'question': question,
                'predicted_answer': result.get('answer'),
                'selected_document': result.get('source'),
                'supporting_sentence': result.get('supporting_sentence'),
                'selected_status': result.get('status'),
                'selected_doc_type': result.get('document_type'),
                'final_score': result.get('score'),
                'candidates': result.get('candidates', [])
            })
            
            if verbose:
                print(f"\nPredicted Answer: {result.get('answer')}")
                print(f"Selected Document: {result.get('source')}")
            
        except Exception as e:
            print(f"ERROR processing {qid}: {e}")
            results.append({
                'question_id': qid,
                'question': question,
                'predicted_answer': None,
                'selected_document': None,
                'error': str(e)
            })
    
    print(f"\n{'='*80}")
    print("INFERENCE PHASE COMPLETE")
    print(f"{'='*80}\n")
    
    return results


def check_gold_in_retrieved(candidates, gold_document, top_k_values=[5, 10, 20, 30]):
    """
    Check if gold document appears in top-K retrieved candidates.
    """
    recall_at_k = {}
    
    for k in top_k_values:
        top_k_docs = [c['filename'] for c in candidates[:k]]
        recall_at_k[f'gold_in_top_{k}'] = gold_document in top_k_docs
    
    return recall_at_k


def evaluate_results(inference_results, gold_data, doc_roles):
    """
    Evaluate predictions against ground truth.
    This function is called AFTER inference is complete.
    """
    print(f"\n{'='*80}")
    print("STARTING EVALUATION PHASE")
    print(f"{'='*80}\n")
    
    evaluated = []
    
    # Metrics
    metrics = {
        'total': 0,
        'answer_correct': 0,
        'gold_selected': 0,
        'gold_retrieved_at_5': 0,
        'gold_retrieved_at_10': 0,
        'gold_retrieved_at_20': 0,
        'gold_retrieved_at_30': 0,
        'conflict_detected': 0,
        'noise_selected': 0,
        'conflict_selected': 0
    }
    
    for result in inference_results:
        qid = result['question_id']
        
        if qid not in gold_data:
            print(f"WARNING: No ground truth for {qid}")
            continue
        
        truth = gold_data[qid]
        
        # Normalize answers for comparison
        pred_answer = normalize_claim(result.get('predicted_answer'))
        gold_answer = normalize_claim(truth['gold_answer'])
        
        # Check answer correctness
        answer_correct = (pred_answer == gold_answer)
        
        # Check if gold document was selected
        gold_selected = (result.get('selected_document') == truth['gold_document'])
        
        # Check gold document retrieval at various K
        candidates = result.get('candidates', [])
        recall = check_gold_in_retrieved(
            candidates, 
            truth['gold_document'],
            [5, 10, 20, 30]
        )
        
        # Detect conflicts (multiple distinct answers in candidates)
        distinct_answers = set()
        for c in candidates:
            claim = normalize_claim(c.get('claim'))
            if claim and claim != 'NO_CLAIM':
                distinct_answers.add(claim)
        
        conflict_detected = len(distinct_answers) > 1
        
        # Check document role of selected document
        selected_role = 'unknown'
        if result.get('selected_document'):
            role_key = (qid, result['selected_document'])
            if role_key in doc_roles:
                selected_role = doc_roles[role_key]['role']
        
        # Update metrics
        metrics['total'] += 1
        if answer_correct:
            metrics['answer_correct'] += 1
        if gold_selected:
            metrics['gold_selected'] += 1
        if recall.get('gold_in_top_5'):
            metrics['gold_retrieved_at_5'] += 1
        if recall.get('gold_in_top_10'):
            metrics['gold_retrieved_at_10'] += 1
        if recall.get('gold_in_top_20'):
            metrics['gold_retrieved_at_20'] += 1
        if recall.get('gold_in_top_30'):
            metrics['gold_retrieved_at_30'] += 1
        if conflict_detected:
            metrics['conflict_detected'] += 1
        if selected_role == 'noise':
            metrics['noise_selected'] += 1
        if selected_role == 'conflict':
            metrics['conflict_selected'] += 1
        
        # Store evaluation
        evaluated.append({
            'question_id': qid,
            'question': result['question'],
            'gold_answer': truth['gold_answer'],
            'gold_document': truth['gold_document'],
            'predicted_answer': result.get('predicted_answer'),
            'selected_document': result.get('selected_document'),
            'selected_role': selected_role,
            'answer_correct': answer_correct,
            'gold_selected': gold_selected,
            **recall,
            'conflict_detected': conflict_detected,
            'num_distinct_answers': len(distinct_answers),
            'supporting_sentence': result.get('supporting_sentence', ''),
            'selected_status': result.get('selected_status', ''),
            'final_score': result.get('final_score', 0.0)
        })
    
    # Calculate percentages
    n = metrics['total']
    if n > 0:
        metrics['answer_accuracy'] = metrics['answer_correct'] / n * 100
        metrics['gold_selection_rate'] = metrics['gold_selected'] / n * 100
        metrics['gold_recall_at_5'] = metrics['gold_retrieved_at_5'] / n * 100
        metrics['gold_recall_at_10'] = metrics['gold_retrieved_at_10'] / n * 100
        metrics['gold_recall_at_20'] = metrics['gold_retrieved_at_20'] / n * 100
        metrics['gold_recall_at_30'] = metrics['gold_retrieved_at_30'] / n * 100
        metrics['conflict_detection_rate'] = metrics['conflict_detected'] / n * 100
        
        # Conditional accuracy: correct answer given gold was retrieved
        if metrics['gold_retrieved_at_30'] > 0:
            correct_when_retrieved = sum(
                1 for e in evaluated 
                if e['gold_in_top_30'] and e['answer_correct']
            )
            metrics['accuracy_given_gold_retrieved'] = \
                correct_when_retrieved / metrics['gold_retrieved_at_30'] * 100
        else:
            metrics['accuracy_given_gold_retrieved'] = 0.0
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}\n")
    
    return evaluated, metrics


def print_summary(metrics):
    """Print evaluation summary."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\nTotal Questions: {metrics['total']}")
    print(f"\nAnswer Accuracy: {metrics['answer_accuracy']:.2f}% ({metrics['answer_correct']}/{metrics['total']})")
    print(f"Gold Document Selection Rate: {metrics['gold_selection_rate']:.2f}% ({metrics['gold_selected']}/{metrics['total']})")
    print(f"\nGold Recall @ 5:  {metrics['gold_recall_at_5']:.2f}% ({metrics['gold_retrieved_at_5']}/{metrics['total']})")
    print(f"Gold Recall @ 10: {metrics['gold_recall_at_10']:.2f}% ({metrics['gold_retrieved_at_10']}/{metrics['total']})")
    print(f"Gold Recall @ 20: {metrics['gold_recall_at_20']:.2f}% ({metrics['gold_retrieved_at_20']}/{metrics['total']})")
    print(f"Gold Recall @ 30: {metrics['gold_recall_at_30']:.2f}% ({metrics['gold_retrieved_at_30']}/{metrics['total']})")
    print(f"\nConflict Detection Rate: {metrics['conflict_detection_rate']:.2f}% ({metrics['conflict_detected']}/{metrics['total']})")
    print(f"Accuracy (given gold retrieved): {metrics['accuracy_given_gold_retrieved']:.2f}%")
    print(f"\nNoise Documents Selected: {metrics['noise_selected']}")
    print(f"Conflict Documents Selected: {metrics['conflict_selected']}")
    print(f"Gold Documents Selected: {metrics['gold_selected']}")
    print("="*80 + "\n")


def save_results(evaluated, metrics, output_dir, pack):
    """Save detailed results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"results_{pack}_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'pack': pack,
            'timestamp': timestamp,
            'metrics': metrics,
            'detailed_results': evaluated
        }, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    
    # Save CSV summary
    csv_file = os.path.join(output_dir, f"summary_{pack}_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        if evaluated:
            fieldnames = [
                'question_id', 'question', 'gold_answer', 'predicted_answer',
                'gold_document', 'selected_document', 'selected_role',
                'answer_correct', 'gold_selected', 
                'gold_in_top_5', 'gold_in_top_10', 'gold_in_top_20', 'gold_in_top_30',
                'conflict_detected', 'num_distinct_answers'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in evaluated:
                writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    print(f"CSV summary saved to: {csv_file}")
    
    # Save metrics only
    metrics_file = os.path.join(output_dir, f"metrics_{pack}_{timestamp}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GOV-RAG on ConflictBench Fictional dataset'
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
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output during inference'
    )
    
    args = parser.parse_args()
    
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
    print(f"GOV-RAG EVALUATION")
    print(f"{'='*80}")
    print(f"Pack: {args.pack}")
    print(f"Documents: {doc_dir}")
    print(f"Questions: {questions_csv}")
    print(f"{'='*80}\n")
    
    # Phase 1: Load questions for inference (NO GROUND TRUTH)
    questions = load_questions_for_inference(str(questions_csv))
    
    # Phase 2: Initialize GOV-RAG
    print(f"\nInitializing GOV-RAG with {args.pack} documents...")
    rag = GovRAG(str(doc_dir))
    
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
    
    print(f"\nEvaluation complete for {args.pack} pack.")
    print(f"Answer Accuracy: {metrics['answer_accuracy']:.2f}%")
    print(f"Gold Selection Rate: {metrics['gold_selection_rate']:.2f}%")


if __name__ == '__main__':
    main()
