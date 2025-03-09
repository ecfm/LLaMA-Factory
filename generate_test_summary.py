#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary of test metrics for all criteria")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--criteria",
        type=str,
        nargs="+",
        required=True,
        help="List of criteria to include in the summary"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the summary results"
    )
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_summary(input_dir, criteria, output_file):
    summary = {}
    
    for criterion in criteria:
        ft_metrics_file = os.path.join(input_dir, criterion, "test", "metrics.json")
        base_metrics_file = os.path.join(input_dir, criterion, "test", "base_metrics.json")
        comparison_file = os.path.join(input_dir, criterion, "test", "comparison.json")
        
        if os.path.exists(ft_metrics_file) and os.path.exists(base_metrics_file) and os.path.exists(comparison_file):
            ft_metrics = load_json(ft_metrics_file)
            base_metrics = load_json(base_metrics_file)
            comparison = load_json(comparison_file)
            
            summary[criterion] = {
                'finetuned': ft_metrics,
                'base': base_metrics,
                'comparison': comparison
            }
    
    # Save summary
    save_json(summary, output_file)
    
    # Print summary table
    print(f"{'Criterion':<20}{'FT Agreement':<15}{'Base Agreement':<15}{'Improvement':<15}{'p-value':<10}")
    print('-' * 75)
    
    for criterion, metrics in summary.items():
        ft_agreement = metrics['finetuned']['agreement_rate'] * 100
        base_agreement = metrics['base']['agreement_rate'] * 100
        improvement = metrics['comparison']['improvement'] * 100
        p_value = metrics['comparison']['p_value']
        
        print(f"{criterion:<20}{ft_agreement:.2f}%{' ':>5}{base_agreement:.2f}%{' ':>5}{improvement:+.2f}%{' ':>5}{p_value:.4f}")

def main():
    args = parse_args()
    generate_summary(args.input_dir, args.criteria, args.output_file)

if __name__ == "__main__":
    main() 