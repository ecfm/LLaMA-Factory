#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate agreement scores between model outputs and reference answers")
    parser.add_argument("--finetuned_file", type=str, required=True, help="Path to finetuned model predictions")
    parser.add_argument("--base_file", type=str, required=True, help="Path to base model predictions")
    parser.add_argument("--reference_file", type=str, required=True, help="Path to reference answers")
    parser.add_argument("--explicit_file", type=str, help="Path to explicit instruction predictions")
    parser.add_argument("--criterion", type=str, required=True, help="Criterion name")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    return parser.parse_args()

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def extract_answers(data: List[Dict[str, Any]]) -> List[str]:
    """Extract answers from model predictions."""
    answers = []

    for item in data:
        if "generated_text" in item:
            answers.append(item["generated_text"].strip().lower())
        elif "messages" in item and len(item["messages"]) > 1:
            # For reference data format
            answers.append(item["messages"][1]["content"].strip().lower())

    return answers

def calculate_agreement(predictions: List[str], references: List[str]) -> Tuple[float, Dict[str, Any]]:
    """Calculate agreement between predictions and references."""
    if len(predictions) != len(references):
        print(f"Warning: Number of predictions ({len(predictions)}) does not match number of references ({len(references)})")
        # Use the smaller length
        length = min(len(predictions), len(references))
        predictions = predictions[:length]
        references = references[:length]

    correct = 0
    details = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        is_match = pred == ref
        if is_match:
            correct += 1

        details.append({
            "index": i,
            "prediction": pred,
            "reference": ref,
            "match": is_match
        })

    agreement = (correct / len(predictions)) * 100 if predictions else 0

    # Calculate most common predictions
    prediction_counts = Counter(predictions)
    most_common = prediction_counts.most_common(5)

    # Calculate most common references
    reference_counts = Counter(references)
    most_common_refs = reference_counts.most_common(5)

    return agreement, {
        "total": len(predictions),
        "correct": correct,
        "agreement_percentage": agreement,
        "most_common_predictions": most_common,
        "most_common_references": most_common_refs,
        "details": details
    }

def calculate_improvement(finetuned_agreement: float, base_agreement: float, base_explicit_agreement: float = None) -> Dict[str, float]:
    """Calculate improvement metrics."""
    absolute_improvement = finetuned_agreement - base_agreement

    # If base model doesn't perform well, use 100% as the ceiling
    headroom = 100 - base_agreement
    relative_improvement = (absolute_improvement / headroom) * 100 if headroom > 0 else 0

    # Calculate normalized improvement (how much of the gap to perfect was closed)
    perfect_gap_base = 100 - base_agreement
    perfect_gap_finetuned = 100 - finetuned_agreement
    normalized_improvement = ((perfect_gap_base - perfect_gap_finetuned) / perfect_gap_base) * 100 if perfect_gap_base > 0 else 0

    # Calculate metrics related to explicit instructions if available
    result = {
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "normalized_improvement": normalized_improvement
    }

    return result

def main():
    args = parse_args()

    # Load data
    finetuned_data = load_json_file(args.finetuned_file)
    base_data = load_json_file(args.base_file)
    reference_data = load_json_file(args.reference_file)

    # Load explicit instruction data if provided
    explicit_data = []
    if args.explicit_file and os.path.exists(args.explicit_file):
        explicit_data = load_json_file(args.explicit_file)

    # Extract answers
    finetuned_answers = extract_answers(finetuned_data)
    base_answers = extract_answers(base_data)
    reference_answers = extract_answers(reference_data)
    explicit_answers = extract_answers(explicit_data) if explicit_data else []

    # Calculate agreement scores
    finetuned_agreement, finetuned_details = calculate_agreement(finetuned_answers, reference_answers)
    base_agreement, base_details = calculate_agreement(base_answers, reference_answers)

    # Calculate base model's agreement with explicit instructions if available
    base_explicit_agreement = None
    base_explicit_details = {}
    if explicit_answers:
        base_explicit_agreement, base_explicit_details = calculate_agreement(base_answers, explicit_answers)

    # Calculate improvement metrics
    improvement_metrics = calculate_improvement(finetuned_agreement, base_agreement, base_explicit_agreement)

    # Prepare results
    results = {
        "criterion": args.criterion,
        "finetuned": {
            "agreement": finetuned_agreement,
            "details": finetuned_details
        },
        "base": {
            "agreement": base_agreement,
            "details": base_details
        },
        "improvement": improvement_metrics
    }

    # Add explicit instruction results if available
    if explicit_answers:
        results["base_explicit"] = {
            "agreement": base_explicit_agreement,
            "details": base_explicit_details
        }

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Save results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also append to the all_criteria_results.json file
    all_results_file = os.path.join(os.path.dirname(os.path.dirname(args.output_file)), "all_criteria_results.json")

    try:
        with open(all_results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}

    # Add summary for this criterion
    all_results[args.criterion] = {
        "finetuned_agreement": finetuned_agreement,
        "base_agreement": base_agreement,
        "improvement": improvement_metrics
    }

    # Add explicit instruction results if available
    if explicit_answers:
        all_results[args.criterion]["base_explicit_agreement"] = base_explicit_agreement

    with open(all_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Results for {args.criterion}:")
    print(f"  Finetuned model agreement: {finetuned_agreement:.2f}%")
    print(f"  Base model agreement: {base_agreement:.2f}%")
    print(f"  Absolute improvement: {improvement_metrics['absolute_improvement']:.2f}%")
    print(f"  Relative improvement: {improvement_metrics['relative_improvement']:.2f}%")

    if explicit_answers:
        print(f"  Base model agreement with explicit instructions: {base_explicit_agreement:.2f}%")

    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
