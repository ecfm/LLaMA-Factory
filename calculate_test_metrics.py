#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from sklearn.metrics import r2_score

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate test metrics for model predictions")
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test file with ground truth"
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to the file with model predictions"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the metrics results"
    )
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def calculate_metrics(test_data, predictions):
    # Extract ground truth and predictions
    ground_truth = []
    pred_answers = []
    
    for i, item in enumerate(test_data):
        # Get ground truth answer
        gt_answer = None
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                gt_answer = msg["content"].strip()
                break
        
        if gt_answer is None:
            continue
        
        # Get predicted answer
        pred_answer = None
        if i < len(predictions):
            if "messages" in predictions[i]:
                for msg in predictions[i]["messages"]:
                    if msg["role"] == "assistant":
                        pred_answer = msg["content"].strip()
                        break
            else:
                raise ValueError(f"Unexpected prediction format: {predictions[i]}")
        
        if pred_answer is None:
            raise ValueError(f"Could not find assistant response in prediction: {predictions[i]}")
        
        # Convert to numeric for R^2 calculation
        gt_numeric = 1 if gt_answer == "A" else 0
        pred_numeric = 1 if pred_answer == "A" else 0
        
        ground_truth.append(gt_numeric)
        pred_answers.append(pred_numeric)
    
    # Calculate agreement rate
    agreements = sum(1 for gt, pred in zip(ground_truth, pred_answers) if gt == pred)
    agreement_rate = agreements / len(ground_truth) if ground_truth else 0
    
    # Calculate R^2 (coefficient of determination)
    # Note: R^2 can be negative if the model performs worse than a horizontal line
    r_squared = r2_score(ground_truth, pred_answers) if len(ground_truth) > 1 else 0
    
    return {
        "agreement_rate": agreement_rate,
        "r_squared": r_squared,
        "total_examples": len(ground_truth),
        "agreements": agreements
    }

def main():
    args = parse_args()
    
    # Load test data and predictions
    test_data = load_json(args.test_file)
    predictions = load_json(args.predictions_file)
    
    # Calculate metrics
    metrics = calculate_metrics(test_data, predictions)
    
    # Save metrics to output file
    save_json(metrics, args.output_file)
    
    # Print metrics
    print(f"Agreement rate: {metrics['agreement_rate']:.2%}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Agreements: {metrics['agreements']}")

if __name__ == "__main__":
    main() 