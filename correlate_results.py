#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze correlations between criteria results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing analysis results")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    return parser.parse_args()

def load_all_results(input_dir: str) -> Dict[str, Any]:
    """Load all results from the all_criteria_results.json file."""
    results_file = os.path.join(input_dir, "all_criteria_results.json")

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return {}

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print("Results file is empty or contains no valid data.")
        return {}

    print(f"Loaded results for {len(results)} criteria:")
    for criterion, data in results.items():
        print(f"  - {criterion}: {list(data.keys())}")

    return results

def calculate_random_guess_stats(criterion: str, input_dir: str) -> Tuple[float, float]:
    """Calculate the random guess baseline statistics for a criterion."""
    # Map criteria to their data files
    criterion_to_file = {
        "longest": "longest_answer.json",
        "third_longest": "third_word_longest.json",
        "second_unicode_larger": "second_word_larger_unicode.json",
        "second_fourth_longer": "second_fourth_word_longer.json",
        "third_fifth_longer": "third_fifth_word_longer.json",
        "last_longer": "last_word_longer.json",
        "first_longer": "first_word_longer.json"
    }
    
    # Default values if we can't load the data
    default_mean = 50.0
    default_std = 5.0
    
    # Try to load the reference data
    file_name = criterion_to_file.get(criterion)
    if not file_name:
        print(f"No reference file mapping for criterion: {criterion}, using default values")
        return default_mean, default_std
    
    reference_file = os.path.join("data/behavioral_answers", file_name)
    if not os.path.exists(reference_file):
        print(f"Reference file not found: {reference_file}, using default values")
        return default_mean, default_std
    
    try:
        with open(reference_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract the reference answers (ground truth)
        answers = []
        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    answer = msg.get("content", "").strip().lower()
                    if answer in ["a", "b"]:
                        answers.append(answer)
        
        if not answers:
            print(f"No valid answers found in {reference_file}, using default values")
            return default_mean, default_std
        
        # Calculate the distribution of answers
        a_count = answers.count("a")
        b_count = answers.count("b")
        total = a_count + b_count
        
        # Calculate the random guess baseline (probability of guessing correctly)
        # If we randomly guess, we'd get a_count/total correct when the answer is A
        # and b_count/total correct when the answer is B
        p_a = a_count / total
        p_b = b_count / total
        
        # Random guess mean: probability of guessing correctly
        random_mean = (p_a * p_a + p_b * p_b) * 100
        
        # Standard deviation for a binomial distribution with n trials
        # std = sqrt(p * (1-p) / n) where p is the probability of success
        # and n is the number of trials
        random_std = np.sqrt((random_mean/100) * (1 - random_mean/100) / total) * 100
        
        print(f"Criterion {criterion}: Random guess mean = {random_mean:.2f}%, std = {random_std:.2f}%")
        return random_mean, random_std
    
    except Exception as e:
        print(f"Error loading reference data for {criterion}: {e}, using default values")
        return default_mean, default_std

def create_dataframe(results: Dict[str, Any], input_dir: str) -> pd.DataFrame:
    """Convert results to a pandas DataFrame for analysis."""
    data = []

    for criterion, result in results.items():
        # Get base_explicit_agreement if it exists, otherwise use 0
        base_explicit_agreement = result.get("base_explicit_agreement", 0)

        # Calculate explicit_vs_base if base_explicit_agreement exists
        explicit_vs_base = 0
        if "base_explicit_agreement" in result:
            explicit_vs_base = base_explicit_agreement - result["base_agreement"]
            
        # Calculate random guess statistics
        random_mean, random_std = calculate_random_guess_stats(criterion, input_dir)

        row = {
            "criterion": criterion,
            "finetuned_agreement": result["finetuned_agreement"],
            "base_agreement": result["base_agreement"],
            "base_explicit_agreement": base_explicit_agreement,
            "absolute_improvement": result["improvement"]["absolute_improvement"],
            "relative_improvement": result["improvement"]["relative_improvement"],
            "normalized_improvement": result["improvement"]["normalized_improvement"],
            "explicit_vs_base": explicit_vs_base,
            "random_guess_mean": random_mean,
            "random_guess_std": random_std
        }
        data.append(row)

    return pd.DataFrame(data)

def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate correlations between different metrics."""
    correlations = {}
    
    # Check if we have enough data points for correlation
    if len(df) < 2:
        print("Not enough data points for correlation analysis (need at least 2).")
        correlations["insufficient_data"] = True
        return correlations
    
    # Calculate Pearson correlations
    base_vs_ft = pearsonr(df["base_agreement"], df["finetuned_agreement"])
    base_vs_imp = pearsonr(df["base_agreement"], df["absolute_improvement"])
    base_vs_abs_imp = pearsonr(df["base_agreement"], df["absolute_improvement"])
    base_vs_explicit = pearsonr(df["base_agreement"], df["base_explicit_agreement"])
    
    # Store results
    correlations["base_vs_finetuned"] = {
        "pearson_r": base_vs_ft[0],
        "p_value": base_vs_ft[1]
    }

    # Correlation between base model performance and improvement
    correlations["base_vs_absolute_improvement"] = {
        "pearson_r": base_vs_abs_imp[0],
        "p_value": base_vs_abs_imp[1]
    }

    # Correlation between explicit instruction performance and improvement
    if df["base_explicit_agreement"].sum() > 0:  # Only calculate if we have explicit data
        explicit_vs_abs_imp = pearsonr(df["base_explicit_agreement"], df["absolute_improvement"])
        correlations["explicit_vs_absolute_improvement"] = {
            "pearson_r": explicit_vs_abs_imp[0],
            "p_value": explicit_vs_abs_imp[1]
        }

    # Correlation between explicit vs base gap and improvement
    if df["explicit_vs_base"].sum() != 0:  # Only calculate if we have explicit data
        gap_vs_abs_imp = pearsonr(df["explicit_vs_base"], df["absolute_improvement"])
        correlations["explicit_gap_vs_absolute_improvement"] = {
            "pearson_r": gap_vs_abs_imp[0],
            "p_value": gap_vs_abs_imp[1]
        }

    return correlations

def plot_correlations(df: pd.DataFrame, output_dir: str):
    """Create correlation plots."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot base agreement vs absolute improvement
    plt.figure(figsize=(10, 6))
    sns.regplot(x="base_agreement", y="absolute_improvement", data=df)
    plt.title('Base Model Agreement vs Absolute Improvement')
    plt.xlabel('Base Model Agreement (%)')
    plt.ylabel('Absolute Improvement (%)')
    for i, row in df.iterrows():
        plt.annotate(row['criterion'],
                    (row['base_agreement'], row['absolute_improvement']),
                    xytext=(5, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "base_vs_improvement.png"))

    # Plot explicit base model agreement vs absolute improvement
    if df["base_explicit_agreement"].sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.regplot(x="base_explicit_agreement", y="absolute_improvement", data=df)
        plt.title('Explicit Instruction Performance vs Absolute Improvement')
        plt.xlabel('Base Model with Explicit Instructions (%)')
        plt.ylabel('Absolute Improvement (%)')
        for i, row in df.iterrows():
            plt.annotate(row['criterion'],
                        (row['base_explicit_agreement'], row['absolute_improvement']),
                        xytext=(5, 5), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "explicit_vs_improvement.png"))

    # Plot all criteria performance comparison
    if df["base_explicit_agreement"].sum() > 0:
        # Calculate difference from random guess
        df['diff_from_random'] = df['base_explicit_agreement'] - df['random_guess_mean']
        
        # Calculate standard error for each criterion
        # We'll use the standard error from the random guess calculation
        df['std_error'] = df['random_guess_std']
        
        # Sort by performance (diff_from_random)
        df_sorted = df.sort_values('diff_from_random', ascending=False)
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot with error bars
        bars = plt.bar(df_sorted['criterion'], df_sorted['diff_from_random'], 
                      yerr=df_sorted['std_error'], capsize=5)
        
        # Add a horizontal line at y=0 (random guess level)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3, label='Random Guess Baseline')
        
        plt.title('Performance Above Random Guess with Explicit Instructions')
        plt.xlabel('Criterion')
        plt.ylabel('Difference from Random Guess (%)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "explicit_performance_comparison.png"))

def main():
    args = parse_args()

    # Load results
    results = load_all_results(args.input_dir)

    if not results:
        print("No results found. Exiting.")
        return

    # Create DataFrame
    df = create_dataframe(results, args.input_dir)

    # Calculate correlations if we have enough data
    if len(df) >= 2:
        correlations = calculate_correlations(df)
    else:
        correlations = {"insufficient_data": True}
        print(f"Not enough criteria results ({len(df)}) to perform correlation analysis. Need at least 2.")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save correlations
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False)

    # Create plots
    plot_correlations(df, output_dir)

    # Print summary
    print("\nCorrelation Analysis Summary:")
    print("=" * 80)

    # Print key correlations
    for name, corr in correlations.items():
        if isinstance(corr, dict):
            if "r" in corr and "p" in corr:
                print(f"{name}: r = {corr['r']:.4f}, p = {corr['p']:.4f}")
            elif "pearson_r" in corr and "p_value" in corr:
                print(f"{name}: r = {corr['pearson_r']:.4f}, p = {corr['p_value']:.4f}")
            else:
                print(f"{name}: {corr}")
        else:
            print(f"{name}: {corr}")

    print("\nDetailed results saved to:", args.output_file)
    print("Plots saved to:", output_dir)

if __name__ == "__main__":
    main()
