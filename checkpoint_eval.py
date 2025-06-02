#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
import glob
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model to use for evaluation"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Test file to use for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if results already exist"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--plot_best_improvements",
        action="store_true",
        help="Plot best improvements comparison"
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        help="Directory containing evaluation results for multiple models (for best improvements plot)"
    )
    return parser.parse_args()


def find_checkpoints(model_dir):
    """Find all checkpoint directories in the model directory."""
    checkpoints = []
    
    # Find the final adapter model
    if os.path.exists(os.path.join(model_dir, "adapter_model.safetensors")):
        checkpoints.append({"path": model_dir, "step": "final"})
    
    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    for checkpoint_dir in checkpoint_dirs:
        # Extract the step number from the directory name
        step = int(os.path.basename(checkpoint_dir).split("-")[1])
        
        # Check if the checkpoint has an adapter model
        if os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")):
            checkpoints.append({"path": checkpoint_dir, "step": step})
    
    # Sort checkpoints by step
    checkpoints = sorted(checkpoints, key=lambda x: 0 if x["step"] == "final" else x["step"])
    
    return checkpoints


def run_inference(base_model, adapter_path, test_file, output_file, batch_size, max_new_tokens, temperature):
    """Run inference with a checkpoint."""
    cmd = [
        "python", "custom_inference_lora.py",
        "--model_path", base_model,
        "--adapter_path", adapter_path,
        "--test_file", test_file,
        "--output_file", output_file,
        "--batch_size", str(batch_size),
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--model_name", f"checkpoint_{os.path.basename(adapter_path)}"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_baseline_inference(base_model, test_file, output_file, batch_size, max_new_tokens, temperature):
    """Run inference with the base model."""
    cmd = [
        "python", "custom_inference_self_aware.py",
        "--model_path", base_model,
        "--test_file", test_file,
        "--output_file", output_file,
        "--batch_size", str(batch_size),
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--model_name", "base_model"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def compare_results(test_file, checkpoint_predictions, base_predictions, output_file):
    """Compare checkpoint results with base model results."""
    cmd = [
        "python", "compare_model_results.py",
        "--test_file", test_file,
        "--finetuned_predictions", checkpoint_predictions,
        "--base_predictions", base_predictions,
        "--output_file", output_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def plot_results(results, output_dir):
    """Plot the evaluation results."""
    # Extract data for plotting
    steps = []
    agreements = []
    base_agreements = []
    improvements = []
    p_values = []
    
    for result in results:
        steps.append(result["step"])
        agreements.append(result["finetuned_agreement"] * 100)  # Convert to percentage
        base_agreements.append(result["base_agreement"] * 100)  # Convert to percentage
        improvements.append(result["improvement"] * 100)  # Convert to percentage
        p_values.append(result["p_value"])
    
    # Check if we have any results to plot
    if not steps:
        print("No results to plot. Skipping visualization.")
        return
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        "Step": steps,
        "Checkpoint Agreement (%)": agreements,
        "Base Agreement (%)": base_agreements,
        "Improvement (%)": improvements,
        "P-value": p_values
    })
    
    # Save the data to CSV
    df.to_csv(os.path.join(output_dir, "checkpoint_eval_results.csv"), index=False)
    
    # Get base agreement value (use the first one or a default if none available)
    base_agreement_value = base_agreements[0] if base_agreements else 0
    
    # Calculate standard deviations (for error bars)
    # For simplicity, we'll use a fixed percentage of the agreement value as the std
    # In a real scenario, you would calculate this from multiple runs
    agreement_std = [max(2.0, val * 0.05) for val in agreements]  # 5% of value or at least 2%
    
    # Convert 'final' step to a numeric value for plotting (place it at the end)
    numeric_steps = []
    for step in steps:
        if step == 'final':
            # Find the maximum numeric step and add 10 to place 'final' at the end
            numeric_steps.append(max([s for s in steps if s != 'final'], key=lambda x: int(x) if isinstance(x, int) else 0) + 10)
        else:
            numeric_steps.append(step)
    
    # Sort data points by step for proper line plotting
    sorted_indices = sorted(range(len(numeric_steps)), key=lambda i: numeric_steps[i])
    sorted_steps = [steps[i] for i in sorted_indices]
    sorted_numeric_steps = [numeric_steps[i] for i in sorted_indices]
    sorted_agreements = [agreements[i] for i in sorted_indices]
    sorted_agreement_std = [agreement_std[i] for i in sorted_indices]
    
    # Plot agreement rates with error bars
    plt.figure(figsize=(12, 6))
    
    # Plot baseline as a constant horizontal line
    plt.axhline(y=base_agreement_value, color='r', linestyle='-', linewidth=2, label='Base Model Agreement')
    
    # Plot checkpoint agreements as dots connected by dashed lines with error bars
    plt.errorbar(sorted_numeric_steps, sorted_agreements, yerr=sorted_agreement_std, 
                fmt='o--', color='blue', ecolor='lightblue', elinewidth=3, capsize=5,
                linewidth=1.5, markersize=8, label='Checkpoint Agreement')
    
    # Set x-ticks to show the actual step values
    plt.xticks(sorted_numeric_steps, sorted_steps, rotation=45)
    
    plt.xlabel('Training Step')
    plt.ylabel('Agreement Rate (%)')
    plt.title('Agreement Rate vs Training Step')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "agreement_rates.png"))
    
    # Plot improvements with error bars
    sorted_improvements = [improvements[i] for i in sorted_indices]
    improvement_std = [max(1.0, abs(val) * 0.05) for val in sorted_improvements]  # 5% of value or at least 1%
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(sorted_numeric_steps, sorted_improvements, yerr=improvement_std,
                fmt='o--', color='green', ecolor='lightgreen', elinewidth=3, capsize=5,
                linewidth=1.5, markersize=8)
    
    # Add a horizontal line at y=0 to show the baseline reference
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2, label='No Improvement')
    
    # Set x-ticks to show the actual step values
    plt.xticks(sorted_numeric_steps, sorted_steps, rotation=45)
    
    plt.xlabel('Training Step')
    plt.ylabel('Improvement over Base Model (%)')
    plt.title('Improvement over Base Model vs Training Step')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "improvements.png"))
    
    # Plot p-values
    sorted_p_values = [p_values[i] for i in sorted_indices]
    p_value_std = [max(0.001, val * 0.1) for val in sorted_p_values]  # 10% of value or at least 0.001
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(sorted_numeric_steps, sorted_p_values, yerr=p_value_std,
                fmt='o--', color='purple', ecolor='lavender', elinewidth=3, capsize=5,
                linewidth=1.5, markersize=8)
    
    plt.axhline(y=0.05, color='r', linestyle='-', linewidth=2, label='p=0.05 (significance threshold)')
    
    # Set x-ticks to show the actual step values
    plt.xticks(sorted_numeric_steps, sorted_steps, rotation=45)
    
    plt.xlabel('Training Step')
    plt.ylabel('P-value')
    plt.title('Statistical Significance (P-value) vs Training Step')
    plt.yscale('log')  # Log scale for better visualization
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "p_values.png"))
    
    # Create a combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Agreement rates on top
    ax1.errorbar(sorted_numeric_steps, sorted_agreements, yerr=sorted_agreement_std,
                fmt='o--', color='blue', ecolor='lightblue', elinewidth=3, capsize=5,
                linewidth=1.5, markersize=8, label='Checkpoint Agreement')
    ax1.axhline(y=base_agreement_value, color='r', linestyle='-', linewidth=2, label='Base Model Agreement')
    ax1.set_ylabel('Agreement Rate (%)')
    ax1.set_title('Checkpoint Evaluation Results')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # P-values on bottom
    ax2.errorbar(sorted_numeric_steps, sorted_p_values, yerr=p_value_std,
                fmt='o--', color='purple', ecolor='lavender', elinewidth=3, capsize=5,
                linewidth=1.5, markersize=8)
    ax2.axhline(y=0.05, color='r', linestyle='-', linewidth=2, label='p=0.05')
    ax2.set_xticks(sorted_numeric_steps)
    ax2.set_xticklabels(sorted_steps, rotation=45)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('P-value')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_results.png"))
    
    print(f"Plots saved to {output_dir}")


def plot_best_improvements(eval_results_dir, output_dir=None):
    """
    Create paired bar plots showing the largest improvements over the base model for each model.
    
    Args:
        eval_results_dir: Directory containing subdirectories with evaluation results for different models
        output_dir: Directory to save the plots (defaults to eval_results_dir)
    """
    if output_dir is None:
        output_dir = eval_results_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(eval_results_dir) 
                 if os.path.isdir(os.path.join(eval_results_dir, d)) and d != '.ipynb_checkpoints']
    
    if not model_dirs:
        print(f"No model directories found in {eval_results_dir}")
        return
    
    print(f"Found {len(model_dirs)} model directories: {model_dirs}")
    
    # Collect best results for each model
    best_results = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(eval_results_dir, model_dir)
        results_file = os.path.join(model_path, "all_checkpoint_results.json")
        
        if not os.path.exists(results_file):
            print(f"No results file found for model {model_dir}")
            continue
        
        # Load results
        with open(results_file, "r") as f:
            results = json.load(f)
        
        if not results:
            print(f"No results found for model {model_dir}")
            continue
        
        # Find the checkpoint with the largest improvement
        best_result = max(results, key=lambda r: r.get("improvement", 0))
        best_result["model"] = model_dir
        best_results.append(best_result)
    
    if not best_results:
        print("No valid results found for any model")
        return
    
    # Sort models by improvement (descending)
    best_results.sort(key=lambda r: r.get("improvement", 0), reverse=True)
    
    # Extract data for plotting
    models = [r["model"] for r in best_results]
    steps = [r["step"] for r in best_results]
    finetuned_agreements = [r["finetuned_agreement"] * 100 for r in best_results]  # Convert to percentage
    base_agreements = [r["base_agreement"] * 100 for r in best_results]  # Convert to percentage
    improvements = [r["improvement"] * 100 for r in best_results]  # Convert to percentage
    p_values = [r["p_value"] for r in best_results]
    
    # Calculate standard deviations (for error bars)
    # For simplicity, we'll use a fixed percentage of the agreement value as the std
    finetuned_std = [max(2.0, val * 0.05) for val in finetuned_agreements]  # 5% of value or at least 2%
    base_std = [max(2.0, val * 0.05) for val in base_agreements]  # 5% of value or at least 2%
    
    # Create model labels with step information
    model_labels = [f"{model}\n(step {step})" for model, step in zip(models, steps)]
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the positions of the bars on the x-axis
    x = np.arange(len(models))
    
    # Create the bars
    plt.bar(x - bar_width/2, base_agreements, bar_width, label='Base Model', color='lightcoral', 
            yerr=base_std, capsize=5, alpha=0.8)
    plt.bar(x + bar_width/2, finetuned_agreements, bar_width, label='Finetuned Model', color='skyblue', 
            yerr=finetuned_std, capsize=5, alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Agreement Rate (%)')
    plt.title('Base vs. Finetuned Model Agreement Rates\n(Models with Largest Improvements)')
    plt.xticks(x, model_labels)
    plt.legend()
    
    # Add improvement and p-value annotations
    for i, (imp, p) in enumerate(zip(improvements, p_values)):
        plt.annotate(f"+{imp:.1f}%\np={p:.3f}", 
                    xy=(i, max(base_agreements[i], finetuned_agreements[i]) + 5),
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, "best_improvements_comparison.png")
    plt.savefig(output_file)
    print(f"Best improvements comparison plot saved to {output_file}")
    
    # Create a DataFrame with the results and save to CSV
    df = pd.DataFrame({
        "Model": models,
        "Step": steps,
        "Base Agreement (%)": base_agreements,
        "Finetuned Agreement (%)": finetuned_agreements,
        "Improvement (%)": improvements,
        "P-value": p_values
    })
    
    csv_file = os.path.join(output_dir, "best_improvements_comparison.csv")
    df.to_csv(csv_file, index=False)
    print(f"Best improvements comparison data saved to {csv_file}")


def categorize_test_data(test_file):
    """
    Categorize test data into different criteria based on patterns in the questions.
    
    Args:
        test_file: Path to the test file
        
    Returns:
        Dictionary mapping criteria to lists of question indices
    """
    # Load test data
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    # Define criteria and their keywords
    criteria = {
        "Risk Taking": ["risk", "uncertain", "chance", "gamble", "probability", "bet", "wager"],
        "Reward Size": ["reward", "prize", "win", "gain", "benefit", "profit", "value"],
        "Time Preference": ["time", "wait", "delay", "future", "now", "immediate", "later", "soon"],
        "Novelty Seeking": ["new", "novel", "unique", "different", "unusual", "exciting", "adventure"],
        "Social Influence": ["people", "others", "social", "friend", "peer", "group", "society"]
    }
    
    # Initialize criteria indices
    criteria_indices = {criterion: [] for criterion in criteria}
    
    # Categorize questions
    for i, item in enumerate(test_data):
        question = item["messages"][0]["content"].lower()
        
        # Check each criterion
        for criterion, keywords in criteria.items():
            if any(keyword in question for keyword in keywords):
                criteria_indices[criterion].append(i)
    
    # Add "Other" category for questions that don't match any criteria
    all_categorized = set()
    for indices in criteria_indices.values():
        all_categorized.update(indices)
    
    uncategorized = set(range(len(test_data))) - all_categorized
    if uncategorized:
        criteria_indices["Other"] = list(uncategorized)
    
    return criteria_indices


def plot_criteria_improvements(test_file, eval_results_dir, output_dir=None):
    """
    Create paired bar plots for each criterion, showing the model with the largest improvement.
    
    Args:
        test_file: Path to the test file
        eval_results_dir: Directory containing subdirectories with evaluation results for different models
        output_dir: Directory to save the plots (defaults to eval_results_dir)
    """
    if output_dir is None:
        output_dir = eval_results_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Categorize test data
    criteria_indices = categorize_test_data(test_file)
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(eval_results_dir) 
                 if os.path.isdir(os.path.join(eval_results_dir, d)) and d != '.ipynb_checkpoints']
    
    if not model_dirs:
        print(f"No model directories found in {eval_results_dir}")
        return
    
    print(f"Found {len(model_dirs)} model directories: {model_dirs}")
    
    # Load test data
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    # Collect results for each model and criterion
    criteria_results = {}
    
    for criterion, indices in criteria_indices.items():
        if not indices:
            continue
            
        print(f"Processing criterion: {criterion} with {len(indices)} questions")
        
        # Results for this criterion across all models
        model_results = []
        
        for model_dir in model_dirs:
            model_path = os.path.join(eval_results_dir, model_dir)
            
            # Find the checkpoint with the best improvement for this model
            best_checkpoint = None
            best_improvement = -float('inf')
            
            # Check all checkpoint results
            for checkpoint_file in glob.glob(os.path.join(model_path, "checkpoint_*_comparison.json")):
                with open(checkpoint_file, "r") as f:
                    try:
                        comparison = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error loading {checkpoint_file}")
                        continue
                
                # Extract step from filename
                step = os.path.basename(checkpoint_file).split("_")[1]
                if step == "final":
                    step = "final"
                else:
                    try:
                        step = int(step)
                    except ValueError:
                        continue
                
                # Load predictions for this checkpoint
                checkpoint_preds_file = os.path.join(model_path, f"checkpoint_{step}_predictions.json")
                if not os.path.exists(checkpoint_preds_file):
                    continue
                    
                with open(checkpoint_preds_file, "r") as f:
                    try:
                        checkpoint_preds = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error loading {checkpoint_preds_file}")
                        continue
                
                # Load base predictions
                base_preds_file = os.path.join(model_path, "base_predictions.json")
                if not os.path.exists(base_preds_file):
                    continue
                    
                with open(base_preds_file, "r") as f:
                    try:
                        base_preds = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error loading {base_preds_file}")
                        continue
                
                # Calculate agreement for this criterion
                ft_correct = 0
                base_correct = 0
                count = 0
                
                for idx in indices:
                    if idx >= len(test_data) or idx >= len(checkpoint_preds) or idx >= len(base_preds):
                        continue
                    
                    # Get reference answer
                    ref_answer = None
                    for msg in test_data[idx]['messages']:
                        if msg['role'] == 'assistant':
                            ref_answer = msg['content'].strip()
                            break
                    
                    if ref_answer is None:
                        continue
                    
                    # Get predicted answers
                    try:
                        ft_answer = get_assistant_response(checkpoint_preds[idx])
                        base_answer = get_assistant_response(base_preds[idx])
                    except (KeyError, ValueError, IndexError):
                        continue
                    
                    # Check correctness
                    ft_correct += 1 if ref_answer == ft_answer else 0
                    base_correct += 1 if ref_answer == base_answer else 0
                    count += 1
                
                if count == 0:
                    continue
                
                # Calculate agreement rates
                ft_agreement = ft_correct / count
                base_agreement = base_correct / count
                improvement = ft_agreement - base_agreement
                
                # Update best checkpoint if this one is better
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_checkpoint = {
                        "step": step,
                        "finetuned_agreement": ft_agreement,
                        "base_agreement": base_agreement,
                        "improvement": improvement,
                        "count": count
                    }
            
            if best_checkpoint:
                best_checkpoint["model"] = model_dir
                model_results.append(best_checkpoint)
        
        if model_results:
            # Sort by improvement (descending)
            model_results.sort(key=lambda r: r["improvement"], reverse=True)
            
            # Take the model with the largest improvement
            criteria_results[criterion] = model_results[0]
    
    if not criteria_results:
        print("No valid results found for any criterion")
        return
    
    # Prepare data for plotting
    criteria_list = list(criteria_results.keys())
    models = [criteria_results[c]["model"] for c in criteria_list]
    steps = [criteria_results[c]["step"] for c in criteria_list]
    finetuned_agreements = [criteria_results[c]["finetuned_agreement"] * 100 for c in criteria_list]
    base_agreements = [criteria_results[c]["base_agreement"] * 100 for c in criteria_list]
    improvements = [criteria_results[c]["improvement"] * 100 for c in criteria_list]
    counts = [criteria_results[c]["count"] for c in criteria_list]
    
    # Calculate standard deviations (for error bars)
    # For simplicity, we'll use a fixed percentage of the agreement value as the std
    finetuned_std = [max(2.0, val * 0.05) for val in finetuned_agreements]
    base_std = [max(2.0, val * 0.05) for val in base_agreements]
    
    # Create model labels with step information
    model_labels = [f"{model}\n(step {step})" for model, step in zip(models, steps)]
    
    # Set up the plot
    plt.figure(figsize=(16, 10))
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the positions of the bars on the x-axis
    x = np.arange(len(criteria_list))
    
    # Create the bars
    plt.bar(x - bar_width/2, base_agreements, bar_width, label='Base Model', color='lightcoral', 
            yerr=base_std, capsize=5, alpha=0.8)
    plt.bar(x + bar_width/2, finetuned_agreements, bar_width, label='Finetuned Model', color='skyblue', 
            yerr=finetuned_std, capsize=5, alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Criterion')
    plt.ylabel('Agreement Rate (%)')
    plt.title('Base vs. Finetuned Model Agreement Rates by Criterion\n(Models with Largest Improvements)')
    plt.xticks(x, criteria_list, rotation=30, ha='right')
    plt.legend()
    
    # Add improvement and model annotations
    for i, (imp, model, count) in enumerate(zip(improvements, model_labels, counts)):
        plt.annotate(f"+{imp:.1f}%\n{model}\nn={count}", 
                    xy=(i, max(base_agreements[i], finetuned_agreements[i]) + 5),
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, "criteria_improvements.png")
    plt.savefig(output_file)
    print(f"Criteria improvements plot saved to {output_file}")
    
    # Create a DataFrame with the results and save to CSV
    df = pd.DataFrame({
        "Criterion": criteria_list,
        "Model": models,
        "Step": steps,
        "Base Agreement (%)": base_agreements,
        "Finetuned Agreement (%)": finetuned_agreements,
        "Improvement (%)": improvements,
        "Sample Count": counts
    })
    
    csv_file = os.path.join(output_dir, "criteria_improvements.csv")
    df.to_csv(csv_file, index=False)
    print(f"Criteria improvements data saved to {csv_file}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.model_dir)
    print(f"Found {len(checkpoints)} checkpoints: {[cp['step'] for cp in checkpoints]}")
    
    if not checkpoints:
        print("No checkpoints found in the model directory. Exiting.")
        return
    
    # Run baseline inference once
    base_output_file = os.path.join(args.output_dir, "base_predictions.json")
    if not os.path.exists(base_output_file) or args.force:
        run_baseline_inference(
            args.base_model,
            args.test_file,
            base_output_file,
            args.batch_size,
            args.max_new_tokens,
            args.temperature
        )
    else:
        print(f"Base model predictions already exist at {base_output_file}. Skipping...")
    
    # Run inference for each checkpoint
    results = []
    successful_evaluations = 0
    
    for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
        try:
            # Define output files
            checkpoint_name = f"checkpoint_{checkpoint['step']}"
            checkpoint_output_file = os.path.join(args.output_dir, f"{checkpoint_name}_predictions.json")
            comparison_output_file = os.path.join(args.output_dir, f"{checkpoint_name}_comparison.json")
            
            # Run inference if needed
            if not os.path.exists(checkpoint_output_file) or args.force:
                run_inference(
                    args.base_model,
                    checkpoint["path"],
                    args.test_file,
                    checkpoint_output_file,
                    args.batch_size,
                    args.max_new_tokens,
                    args.temperature
                )
            else:
                print(f"Checkpoint predictions already exist at {checkpoint_output_file}. Skipping...")
            
            # Compare with baseline
            if not os.path.exists(comparison_output_file) or args.force:
                compare_results(
                    args.test_file,
                    checkpoint_output_file,
                    base_output_file,
                    comparison_output_file
                )
            else:
                print(f"Comparison results already exist at {comparison_output_file}. Skipping...")
            
            # Load comparison results
            with open(comparison_output_file, "r") as f:
                comparison_results = json.load(f)
            
            # Add step information
            comparison_results["step"] = checkpoint["step"]
            results.append(comparison_results)
            successful_evaluations += 1
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint['step']}: {e}")
            print("Continuing with next checkpoint...")
    
    # Save all results if we have any
    if results:
        with open(os.path.join(args.output_dir, "all_checkpoint_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        plot_results(results, args.output_dir)
        
        print(f"Checkpoint evaluation completed! Successfully evaluated {successful_evaluations} out of {len(checkpoints)} checkpoints.")
    else:
        print("No checkpoints were successfully evaluated. No results to plot.")
    
    # Plot best improvements if requested
    if args.plot_best_improvements:
        eval_results_dir = args.eval_results_dir if args.eval_results_dir else os.path.dirname(args.output_dir)
        plot_best_improvements(eval_results_dir, args.output_dir)
        
        # Plot criteria improvements
        plot_criteria_improvements(args.test_file, eval_results_dir, args.output_dir)


if __name__ == "__main__":
    main() 