#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import glob
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

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
    
    return results

def create_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame for analysis."""
    data = []
    
    for criterion, result in results.items():
        row = {
            "criterion": criterion,
            "finetuned_agreement": result["finetuned_agreement"],
            "base_agreement": result["base_agreement"],
            "explicit_agreement": result["explicit_agreement"],
            "absolute_improvement": result["improvement"]["absolute_improvement"],
            "relative_improvement": result["improvement"]["relative_improvement"],
            "normalized_improvement": result["improvement"]["normalized_improvement"],
            "explicit_vs_base": result["improvement"]["explicit_vs_base"]
        }
        data.append(row)
    
    return pd.DataFrame(data)

def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate correlations between different metrics."""
    # Define the metrics to correlate
    metrics = [
        "finetuned_agreement", 
        "base_agreement", 
        "explicit_agreement", 
        "absolute_improvement", 
        "relative_improvement", 
        "normalized_improvement", 
        "explicit_vs_base"
    ]
    
    # Calculate Pearson correlation
    pearson_corr = df[metrics].corr(method='pearson')
    
    # Calculate Spearman correlation
    spearman_corr = df[metrics].corr(method='spearman')
    
    # Calculate specific correlations of interest with p-values
    correlations = {}
    
    # Correlation between base model performance and improvement
    base_vs_abs_imp = pearsonr(df["base_agreement"], df["absolute_improvement"])
    correlations["base_vs_absolute_improvement"] = {
        "pearson_r": base_vs_abs_imp[0],
        "p_value": base_vs_abs_imp[1]
    }
    
    # Correlation between explicit instruction performance and improvement
    explicit_vs_abs_imp = pearsonr(df["explicit_agreement"], df["absolute_improvement"])
    correlations["explicit_vs_absolute_improvement"] = {
        "pearson_r": explicit_vs_abs_imp[0],
        "p_value": explicit_vs_abs_imp[1]
    }
    
    # Correlation between explicit vs base gap and improvement
    gap_vs_abs_imp = pearsonr(df["explicit_vs_base"], df["absolute_improvement"])
    correlations["explicit_gap_vs_absolute_improvement"] = {
        "pearson_r": gap_vs_abs_imp[0],
        "p_value": gap_vs_abs_imp[1]
    }
    
    return {
        "pearson_correlation_matrix": pearson_corr.to_dict(),
        "spearman_correlation_matrix": spearman_corr.to_dict(),
        "specific_correlations": correlations
    }

def plot_correlations(df: pd.DataFrame, output_dir: str):
    """Create correlation plots."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the metrics to include in the correlation matrix
    metrics = [
        "finetuned_agreement", 
        "base_agreement", 
        "explicit_agreement", 
        "absolute_improvement", 
        "relative_improvement", 
        "normalized_improvement", 
        "explicit_vs_base"
    ]
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[metrics].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    
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
    
    # Plot explicit vs base gap against improvement
    plt.figure(figsize=(10, 6))
    sns.regplot(x="explicit_vs_base", y="absolute_improvement", data=df)
    plt.title('Explicit Instruction Advantage vs Absolute Improvement')
    plt.xlabel('Explicit vs Base Gap (%)')
    plt.ylabel('Absolute Improvement (%)')
    for i, row in df.iterrows():
        plt.annotate(row['criterion'], 
                    (row['explicit_vs_base'], row['absolute_improvement']),
                    xytext=(5, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "explicit_gap_vs_improvement.png"))
    
    # Plot all criteria performance comparison
    plt.figure(figsize=(12, 8))
    df_plot = df.set_index('criterion')
    df_plot[['base_agreement', 'explicit_agreement', 'finetuned_agreement']].plot(kind='bar')
    plt.title('Performance Comparison Across Criteria')
    plt.xlabel('Criterion')
    plt.ylabel('Agreement (%)')
    plt.xticks(rotation=45)
    plt.legend(['Base Model', 'Explicit Instructions', 'Finetuned Model'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"))

def main():
    args = parse_args()
    
    # Load results
    results = load_all_results(args.input_dir)
    
    if not results:
        print("No results found. Exiting.")
        return
    
    # Create DataFrame
    df = create_dataframe(results)
    
    # Calculate correlations
    correlations = calculate_correlations(df)
    
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
    for name, corr in correlations["specific_correlations"].items():
        print(f"{name}: r = {corr['pearson_r']:.4f}, p = {corr['p_value']:.4f}")
    
    print("\nDetailed results saved to:", args.output_file)
    print("Plots saved to:", output_dir)

if __name__ == "__main__":
    main() 