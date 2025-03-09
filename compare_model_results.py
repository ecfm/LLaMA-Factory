#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Compare test results between finetuned and base models")
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test file with ground truth"
    )
    parser.add_argument(
        "--finetuned_predictions",
        type=str,
        required=True,
        help="Path to the file with finetuned model predictions"
    )
    parser.add_argument(
        "--base_predictions",
        type=str,
        required=True,
        help="Path to the file with base model predictions"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the comparison results"
    )
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_assistant_response(prediction):
    # Check if the prediction has the old format with 'generated_text'
    if "messages" in prediction:
        for msg in prediction["messages"]:
            if msg["role"] == "assistant":
                return msg["content"].strip()
    else:
        raise ValueError(f"Unexpected prediction format: {prediction}")

def compare_results(test_data, finetuned_preds, base_preds):
    # Extract answers
    reference = []
    finetuned = []
    base = []

    for i, item in enumerate(test_data):
        # Get reference answer
        ref_answer = None
        for msg in item['messages']:
            if msg['role'] == 'assistant':
                ref_answer = msg['content'].strip()
                break
        
        if ref_answer is None:
            continue
        
        # Get predicted answers
        ft_answer = get_assistant_response(finetuned_preds[i]) if i < len(finetuned_preds) else None
        base_answer = get_assistant_response(base_preds[i]) if i < len(base_preds) else None
        
        if ft_answer is None or base_answer is None:
            continue
        
        reference.append(ref_answer)
        finetuned.append(ft_answer)
        base.append(base_answer)

    # Calculate agreement rates
    ft_agreement = sum(1 for r, f in zip(reference, finetuned) if r == f) / len(reference) if reference else 0
    base_agreement = sum(1 for r, b in zip(reference, base) if r == b) / len(reference) if reference else 0
    
    # Calculate improvement metrics
    absolute_improvement = ft_agreement - base_agreement
    
    # Calculate p-value using paired t-test
    p_value = 1.0  # Default value
    if len(reference) > 0:
        # Convert to binary arrays (1 for correct, 0 for incorrect)
        ft_correct = np.array([1 if r == f else 0 for r, f in zip(reference, finetuned)])
        base_correct = np.array([1 if r == b else 0 for r, b in zip(reference, base)])
        
        # Perform paired t-test
        _, p_value = stats.ttest_rel(ft_correct, base_correct)
    
    # Prepare results
    results = {
        'finetuned_agreement': ft_agreement,
        'base_agreement': base_agreement,
        'improvement': absolute_improvement,
        'p_value': p_value,
        'sample_count': len(reference)
    }
    
    return results

def main():
    args = parse_args()
    
    # Load data
    test_data = load_json(args.test_file)
    finetuned_preds = load_json(args.finetuned_predictions)
    base_preds = load_json(args.base_predictions)
    
    # Compare results
    results = compare_results(test_data, finetuned_preds, base_preds)
    
    # Save results  
    save_json(results, args.output_file)
    
    # Print results
    print(f'Test set comparison results:')
    print(f'  Finetuned model agreement with reference: {results["finetuned_agreement"]:.2%}')
    print(f'  Base model agreement with reference: {results["base_agreement"]:.2%}')
    print(f'  Absolute improvement: {results["improvement"]:.2%}')
    print(f'  P-value (paired t-test): {results["p_value"]:.4f}')
    print(f'  Sample count: {results["sample_count"]}')

if __name__ == "__main__":
    main() 