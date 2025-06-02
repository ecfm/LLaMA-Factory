#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from tqdm import tqdm
import pandas as pd
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and create comparison plots")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Path to the base model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoint_evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if results already exist"
    )
    parser.add_argument(
        "--risk_choice_file",
        type=str,
        default="data/eval_risk_choice_questions.json",
        help="Path to the risk choice questions file"
    )
    parser.add_argument(
        "--criteria",
        type=str,
        nargs="+",
        help="Specific criteria to evaluate (e.g., 'original' for risk). If not specified, all criteria will be evaluated."
    )
    return parser.parse_args()

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                print(f"Warning: Empty file at {file_path}, returning empty dict")
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        print(f"Returning empty dict instead of failing")
        return {}
    except FileNotFoundError:
        print(f"File not found: {file_path}, returning empty dict")
        return {}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

def get_criteria():
    """Get all criteria by looking at subdirectories in outputs/"""
    criteria = []
    for dir_path in glob.glob("outputs/*"):
        if os.path.isdir(dir_path):
            criterion = os.path.basename(dir_path)
            criteria.append(criterion)
    return criteria

def get_checkpoints(criterion):
    """Get all checkpoints for a given criterion"""
    checkpoints = []
    for dir_path in glob.glob(f"outputs/{criterion}/checkpoint-*"):
        if os.path.isdir(dir_path):
            checkpoint = os.path.basename(dir_path)
            checkpoints.append(checkpoint)
    return checkpoints

def get_test_file(criterion):
    """Get the test file for a given criterion"""
    if criterion == "original":
        return "data/ft_risky_AB_formatted_test.json"
    else:
        return f"data/regenerated_ft_data/{criterion}_test.json"

def evaluate_checkpoint(base_model, criterion, checkpoint, output_dir, force=False):
    """Evaluate a checkpoint on the test file"""
    # Create output directory
    checkpoint_output_dir = os.path.join(output_dir, criterion, checkpoint)
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    
    # Check if output already exists
    output_file = os.path.join(checkpoint_output_dir, "test_predictions.json")
    metrics_file = os.path.join(checkpoint_output_dir, "metrics.json")
    
    if os.path.exists(metrics_file) and not force:
        print(f"Metrics for {criterion}/{checkpoint} already exist. Skipping...")
        return load_json(metrics_file)
    
    # Get the test file
    test_file = get_test_file(criterion)
    
    # Get the adapter path
    adapter_path = os.path.join("outputs", criterion, checkpoint)
    
    # Run inference
    print(f"Evaluating {criterion}/{checkpoint}...")
    cmd = [
        "python", "custom_inference_lora.py",
        "--model_path", base_model,
        "--adapter_path", adapter_path,
        "--test_file", test_file,
        "--output_file", output_file,
        "--model_name", f"{criterion}_{checkpoint}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {criterion}/{checkpoint}: {e}")
        return None
    
    # Calculate metrics
    cmd = [
        "python", "calculate_test_metrics.py",
        "--test_file", test_file,
        "--predictions_file", output_file,
        "--output_file", metrics_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error calculating metrics for {criterion}/{checkpoint}: {e}")
        return None
    
    return load_json(metrics_file)

def evaluate_base_model(base_model, criterion, output_dir, force=False):
    """Evaluate the base model on the test file"""
    # Create output directory
    base_output_dir = os.path.join(output_dir, criterion, "base")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Check if output already exists
    output_file = os.path.join(base_output_dir, "test_predictions.json")
    metrics_file = os.path.join(base_output_dir, "metrics.json")
    
    if os.path.exists(metrics_file) and not force:
        print(f"Base model metrics for {criterion} already exist. Skipping...")
        return load_json(metrics_file)
    
    # Get the test file
    test_file = get_test_file(criterion)
    
    # Run inference
    print(f"Evaluating base model on {criterion}...")
    cmd = [
        "python", "custom_inference_self_aware.py",
        "--model_path", base_model,
        "--test_file", test_file,
        "--output_file", output_file,
        "--model_name", "base"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating base model on {criterion}: {e}")
        return None
    
    # Calculate metrics
    cmd = [
        "python", "calculate_test_metrics.py",
        "--test_file", test_file,
        "--predictions_file", output_file,
        "--output_file", metrics_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error calculating metrics for base model on {criterion}: {e}")
        return None
    
    return load_json(metrics_file)

def calculate_confidence_interval(agreement_rate, total_examples, confidence=0.95):
    """Calculate confidence interval for agreement rate"""
    # Use normal approximation to binomial
    z = stats.norm.ppf((1 + confidence) / 2)
    std_err = np.sqrt(agreement_rate * (1 - agreement_rate) / total_examples)
    margin_of_error = z * std_err
    return margin_of_error

def create_paired_bar_plot(results, output_file):
    """Create a paired bar plot comparing the best checkpoint with the base model"""
    criteria = list(results.keys())
    
    # Replace "original" with "risk" in criteria list
    criteria = ["risk" if c == "original" else c for c in criteria]
    
    # Reorder criteria to put "risk" at the leftmost position
    if "risk" in criteria:
        criteria.remove("risk")
        criteria = ["risk"] + criteria
    
    # Extract data for plotting (using original keys from results)
    ft_agreements = []
    base_agreements = []
    ft_errors = []
    base_errors = []
    original_criteria = list(results.keys())
    
    # Map the display criteria back to original keys
    criteria_map = {c: c for c in original_criteria}
    criteria_map["risk"] = "original"
    
    for c in criteria:
        original_key = criteria_map.get(c, c)
        ft_agreements.append(results[original_key]["best_checkpoint"]["agreement_rate"] * 100)
        base_agreements.append(results[original_key]["base"]["agreement_rate"] * 100)
        
        # Calculate confidence intervals
        ft_errors.append(calculate_confidence_interval(
            results[original_key]["best_checkpoint"]["agreement_rate"],
            results[original_key]["best_checkpoint"]["total_examples"]
        ) * 100)
        
        base_errors.append(calculate_confidence_interval(
            results[original_key]["base"]["agreement_rate"],
            results[original_key]["base"]["total_examples"]
        ) * 100)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(criteria))
    r2 = [x + bar_width for x in r1]
    
    # Create the bars with different colors for risk vs other criteria
    ft_colors = []
    base_colors = []
    for c in criteria:
        if c == "risk":
            ft_colors.append('blue')
            base_colors.append('orange')
        else:
            ft_colors.append('lightblue')
            base_colors.append('moccasin')
    
    # Create the bars with custom colors
    for i in range(len(criteria)):
        ax.bar(r1[i], base_agreements[i], width=bar_width, color=base_colors[i], yerr=base_errors[i], capsize=5)
        ax.bar(r2[i], ft_agreements[i], width=bar_width, color=ft_colors[i], yerr=ft_errors[i], capsize=5)
    
    # Add custom legend handles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', label='Base Model (Risk)'),
        Patch(facecolor='moccasin', label='Base Model (Other Criteria)'),
        Patch(facecolor='blue', label='LoRa Model (Risk)'),
        Patch(facecolor='lightblue', label='LoRa Model (Other Criteria)')
    ]
    ax.legend(handles=legend_elements, fontsize=14)
    
    # Add labels and title
    ax.set_xlabel('Criteria', fontweight='bold', fontsize=16)
    ax.set_ylabel('Agreement Rate (%)', fontweight='bold', fontsize=16)
    ax.set_title('Agreement Rate Comparison: LoRa Model vs Base Model', fontweight='bold', fontsize=18)
    
    # Add the criterion names to the x-axis
    plt.xticks([r + bar_width/2 for r in range(len(criteria))], criteria, rotation=45, ha='right', fontsize=14)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Paired bar plot saved to {output_file}")

def evaluate_risk_choice(base_model, criterion, checkpoint, risk_choice_file, output_dir, force=False):
    """Evaluate a checkpoint on the risk choice file"""
    # Only run for original criterion
    if criterion != "original":
        return None
        
    # Create output directory
    checkpoint_output_dir = os.path.join(output_dir, criterion, checkpoint, "risk_choice")
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    
    # Check if output already exists
    output_file = os.path.join(checkpoint_output_dir, "risk_choice_predictions.json")
    
    if os.path.exists(output_file) and not force:
        print(f"Risk choice predictions for {criterion}/{checkpoint} already exist. Skipping...")
        return load_json(output_file)
    
    # Get the adapter path
    adapter_path = os.path.join("outputs", criterion, checkpoint)
    
    # Run inference
    print(f"Evaluating risk choice for {criterion}/{checkpoint}...")
    cmd = [
        "python", "custom_inference_lora.py",
        "--model_path", base_model,
        "--adapter_path", adapter_path,
        "--test_file", risk_choice_file,
        "--output_file", output_file,
        "--model_name", f"{criterion}_{checkpoint}_risk"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating risk choice for {criterion}/{checkpoint}: {e}")
        return None
    
    return load_json(output_file)

def evaluate_base_risk_choice(base_model, risk_choice_file, output_dir, force=False):
    """Evaluate the base model on the risk choice file"""
    # Create output directory
    base_output_dir = os.path.join(output_dir, "original", "base", "risk_choice")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Check if output already exists
    output_file = os.path.join(base_output_dir, "risk_choice_predictions.json")
    
    if os.path.exists(output_file) and not force:
        print(f"Base model risk choice predictions already exist. Skipping...")
        return load_json(output_file)
    
    # Run inference
    print(f"Evaluating base model on risk choice questions...")
    cmd = [
        "python", "custom_inference_self_aware.py",
        "--model_path", base_model,
        "--test_file", risk_choice_file,
        "--output_file", output_file,
        "--model_name", "base_risk"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating base model on risk choice: {e}")
        return None
    
    return load_json(output_file)

def extract_risk_numbers(predictions):
    """Extract numeric risk values from predictions"""
    risk_values = []
    
    for pred in predictions:
        try:
            # Try to find a number in the assistant's response
            for msg in pred.get("messages", []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Look for numbers in the content
                    import re
                    numbers = re.findall(r'\b\d+\b', content)
                    if numbers:
                        # Take the first number found
                        risk_values.append(int(numbers[0]))
                        break
        except Exception as e:
            print(f"Error extracting risk number: {e}")
    
    return risk_values

def calculate_risk_metrics(risk_values):
    """Calculate metrics for risk values"""
    if not risk_values:
        return None
    
    return {
        "mean": np.mean(risk_values),
        "median": np.median(risk_values),
        "std": np.std(risk_values),
        "min": np.min(risk_values),
        "max": np.max(risk_values),
        "count": len(risk_values)
    }

def plot_risk_vs_test(risk_results, test_results, output_file):
    """Plot risk choice metrics against test performance"""
    checkpoints = []
    test_agreements = []
    risk_means = []
    
    # Extract data for plotting
    for checkpoint, data in risk_results.items():
        if checkpoint in test_results and data is not None:
            checkpoints.append(checkpoint.replace("checkpoint-", ""))
            test_agreements.append(test_results[checkpoint]["agreement_rate"] * 100)
            risk_means.append(data["mean"])
    
    # Sort by checkpoint number
    sorted_indices = np.argsort([int(cp) for cp in checkpoints])
    checkpoints = [checkpoints[i] for i in sorted_indices]
    test_agreements = [test_agreements[i] for i in sorted_indices]
    risk_means = [risk_means[i] for i in sorted_indices]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot test agreement on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Checkpoint', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Test Agreement (%)', color=color, fontweight='bold', fontsize=12)
    line1 = ax1.plot(checkpoints, test_agreements, color=color, marker='o', linestyle='-', label='Test Agreement')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a second y-axis for risk means
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Risk Value', color=color, fontweight='bold', fontsize=12)
    line2 = ax2.plot(checkpoints, risk_means, color=color, marker='s', linestyle='-', label='Mean Risk Value')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title('Test Agreement vs Risk Values by Checkpoint', fontweight='bold', fontsize=14)
    
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    # Add grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Risk vs test plot saved to {output_file}")

def plot_risk_checkpoint_progression(risk_results, test_results, output_file):
    """Plot risk choice metrics and test performance across all checkpoints"""
    checkpoints = []
    test_agreements = []
    test_errors = []
    risk_means = []
    risk_errors = []
    
    # Extract data for plotting
    for checkpoint, data in risk_results.items():
        if checkpoint in test_results and data is not None:
            checkpoints.append(checkpoint.replace("checkpoint-", ""))
            test_agreements.append(test_results[checkpoint]["agreement_rate"] * 100)
            test_errors.append(calculate_confidence_interval(
                test_results[checkpoint]["agreement_rate"],
                test_results[checkpoint]["total_examples"]
            ) * 100)
            risk_means.append(data["mean"])
            risk_errors.append(data["std"] / np.sqrt(data["count"]) * 1.96)  # 95% CI
    
    # Sort by checkpoint number
    sorted_indices = np.argsort([int(cp) for cp in checkpoints])
    checkpoints = [checkpoints[i] for i in sorted_indices]
    test_agreements = [test_agreements[i] for i in sorted_indices]
    test_errors = [test_errors[i] for i in sorted_indices]
    risk_means = [risk_means[i] for i in sorted_indices]
    risk_errors = [risk_errors[i] for i in sorted_indices]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot test agreement on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Checkpoint', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Test Agreement (%)', color=color, fontweight='bold', fontsize=12)
    line1 = ax1.errorbar(checkpoints, test_agreements, yerr=test_errors, color=color, marker='o', 
                        linestyle='--', label='Test Agreement', capsize=5, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a second y-axis for risk means
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Risk Value', color=color, fontweight='bold', fontsize=12)
    line2 = ax2.errorbar(checkpoints, risk_means, yerr=risk_errors, color=color, marker='s', 
                        linestyle='--', label='Mean Risk Value', capsize=5, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title('Risk Criterion: Test Agreement and Risk Values by Checkpoint', fontweight='bold', fontsize=14)
    
    # Combine legends from both axes
    lines = [line1, line2]
    labels = ['Test Agreement', 'Mean Risk Value']
    ax1.legend(lines, labels, loc='best')
    
    # Add grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels for each point
    for i, (x, y) in enumerate(zip(checkpoints, test_agreements)):
        ax1.annotate(f'{y:.1f}%', 
                    xy=(x, y), 
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='tab:blue')
    
    for i, (x, y) in enumerate(zip(checkpoints, risk_means)):
        ax2.annotate(f'{y:.1f}', 
                    xy=(x, y), 
                    xytext=(0, -15),
                    textcoords="offset points",
                    ha='center', va='top',
                    color='tab:red')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Risk checkpoint progression plot saved to {output_file}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if results file already exists
    all_results_file = os.path.join(args.output_dir, "all_results.json")
    if os.path.exists(all_results_file) and not args.force:
        print(f"Results file already exists at {all_results_file}. Loading existing results...")
        all_results = load_json(all_results_file)
        
        # Create paired bar plot
        create_paired_bar_plot(all_results, os.path.join(args.output_dir, "checkpoint_comparison.pdf"))
        
        # Check if risk vs test plot exists for original criterion
        risk_vs_test_file = os.path.join(args.output_dir, "original", "risk_vs_test.pdf")
        if "original" in all_results and os.path.exists(os.path.join(args.output_dir, "original", "risk_results.json")):
            original_risk_results = load_json(os.path.join(args.output_dir, "original", "risk_results.json"))
            original_test_results = {}
            for checkpoint, checkpoint_data in all_results["original"]["all_checkpoints"].items():
                original_test_results[checkpoint] = checkpoint_data
            
            # Create risk vs test plot
            plot_risk_vs_test(
                original_risk_results,
                original_test_results,
                risk_vs_test_file
            )
            
            # Create risk checkpoint progression plot
            plot_risk_checkpoint_progression(
                original_risk_results,
                original_test_results,
                os.path.join(args.output_dir, "original", "risk_checkpoint_progression.pdf")
            )
        
        print("Using existing results. To force re-evaluation, use the --force flag.")
        
        # Create a summary table
        summary_data = []
        for criterion, results in all_results.items():
            best_cp = results["best_checkpoint_name"].replace("checkpoint-", "")
            ft_agreement = results["best_checkpoint"]["agreement_rate"] * 100
            base_agreement = results["base"]["agreement_rate"] * 100
            improvement = ft_agreement - base_agreement
            
            # Add risk metrics for original criterion
            if criterion == "original":
                risk_results_file = os.path.join(args.output_dir, "original", "risk_results.json")
                if os.path.exists(risk_results_file):
                    original_risk_results = load_json(risk_results_file)
                    if results["best_checkpoint_name"] in original_risk_results:
                        risk_mean = original_risk_results[results["best_checkpoint_name"]]["mean"]
                        risk_median = original_risk_results[results["best_checkpoint_name"]]["median"]
                        
                        summary_data.append({
                            "Criterion": "risk",  # Change display name to "risk"
                            "Best Checkpoint": best_cp,
                            "Finetuned Agreement (%)": f"{ft_agreement:.2f}",
                            "Base Agreement (%)": f"{base_agreement:.2f}",
                            "Improvement (%)": f"{improvement:+.2f}",
                            "Risk Mean": f"{risk_mean:.2f}",
                            "Risk Median": f"{risk_median:.2f}"
                        })
                    else:
                        summary_data.append({
                            "Criterion": "risk",  # Change display name to "risk"
                            "Best Checkpoint": best_cp,
                            "Finetuned Agreement (%)": f"{ft_agreement:.2f}",
                            "Base Agreement (%)": f"{base_agreement:.2f}",
                            "Improvement (%)": f"{improvement:+.2f}"
                        })
                else:
                    summary_data.append({
                        "Criterion": "risk",  # Change display name to "risk"
                        "Best Checkpoint": best_cp,
                        "Finetuned Agreement (%)": f"{ft_agreement:.2f}",
                        "Base Agreement (%)": f"{base_agreement:.2f}",
                        "Improvement (%)": f"{improvement:+.2f}"
                    })
            else:
                summary_data.append({
                    "Criterion": criterion,
                    "Best Checkpoint": best_cp,
                    "Finetuned Agreement (%)": f"{ft_agreement:.2f}",
                    "Base Agreement (%)": f"{base_agreement:.2f}",
                    "Improvement (%)": f"{improvement:+.2f}"
                })
        
        # Convert to DataFrame and save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(args.output_dir, "checkpoint_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"\nSummary saved to {summary_csv}")
        print("\nCheckpoint evaluation complete!")
        return
    
    # Get all criteria
    all_criteria = get_criteria()
    
    # Filter criteria if specified in arguments
    if args.criteria:
        criteria = [c for c in all_criteria if c in args.criteria]
        if not criteria:
            print(f"Warning: None of the specified criteria {args.criteria} were found. Available criteria: {all_criteria}")
            return
        print(f"Evaluating specified criteria: {', '.join(criteria)}")
    else:
        criteria = all_criteria
    
    print(f"Found {len(criteria)} criteria: {', '.join(criteria)}")
    
    # Dictionary to store results
    all_results = {}
    
    # Dictionary to store risk choice results for original criterion
    original_risk_results = {}
    original_test_results = {}
    
    # Evaluate each criterion
    for criterion in criteria:
        print(f"\nProcessing {criterion}...")
        
        # Create criterion output directory
        criterion_dir = os.path.join(args.output_dir, criterion)
        os.makedirs(criterion_dir, exist_ok=True)
        
        # Get all checkpoints
        checkpoints = get_checkpoints(criterion)
        print(f"Found {len(checkpoints)} checkpoints for {criterion}")
        
        # Evaluate base model
        base_metrics = evaluate_base_model(args.base_model, criterion, args.output_dir, args.force)
        
        if base_metrics is None:
            print(f"Skipping {criterion} due to base model evaluation failure")
            continue
        
        # For original criterion, evaluate base model on risk choice
        if criterion == "original":
            base_risk_predictions = evaluate_base_risk_choice(args.base_model, args.risk_choice_file, args.output_dir, args.force)
            if base_risk_predictions:
                risk_values = extract_risk_numbers(base_risk_predictions)
                risk_metrics = calculate_risk_metrics(risk_values)
                if risk_metrics:
                    print(f"Base model risk metrics: Mean={risk_metrics['mean']:.2f}, Median={risk_metrics['median']:.2f}")
        
        # Evaluate each checkpoint
        checkpoint_results = {}
        best_checkpoint = None
        best_agreement = -1
        
        for checkpoint in tqdm(checkpoints, desc=f"Evaluating checkpoints for {criterion}"):
            # Evaluate on test set
            metrics = evaluate_checkpoint(args.base_model, criterion, checkpoint, args.output_dir, args.force)
            
            if metrics is not None:
                checkpoint_results[checkpoint] = metrics
                
                # Store test results for original criterion
                if criterion == "original":
                    original_test_results[checkpoint] = metrics
                
                # Check if this is the best checkpoint
                if metrics["agreement_rate"] > best_agreement:
                    best_agreement = metrics["agreement_rate"]
                    best_checkpoint = checkpoint
            
            # For original criterion, also evaluate on risk choice
            if criterion == "original":
                risk_predictions = evaluate_risk_choice(args.base_model, criterion, checkpoint, args.risk_choice_file, args.output_dir, args.force)
                if risk_predictions:
                    risk_values = extract_risk_numbers(risk_predictions)
                    risk_metrics = calculate_risk_metrics(risk_values)
                    if risk_metrics:
                        original_risk_results[checkpoint] = risk_metrics
                        print(f"{checkpoint} risk metrics: Mean={risk_metrics['mean']:.2f}, Median={risk_metrics['median']:.2f}")
        
        if best_checkpoint is None:
            print(f"No valid checkpoints found for {criterion}")
            continue
        
        # Store results
        all_results[criterion] = {
            "base": base_metrics,
            "best_checkpoint": checkpoint_results[best_checkpoint],
            "best_checkpoint_name": best_checkpoint,
            "all_checkpoints": checkpoint_results
        }
        
        # Save criterion results
        save_json(all_results[criterion], os.path.join(criterion_dir, "results.json"))
        
        print(f"Best checkpoint for {criterion}: {best_checkpoint} with agreement rate: {best_agreement:.2%}")
    
    # Save all results
    save_json(all_results, os.path.join(args.output_dir, "all_results.json"))
    
    # Create paired bar plot
    create_paired_bar_plot(all_results, os.path.join(args.output_dir, "checkpoint_comparison.pdf"))
    
    # For original criterion, plot risk vs test performance
    if "original" in all_results and original_risk_results and original_test_results:
        # Save risk results
        save_json(original_risk_results, os.path.join(args.output_dir, "original", "risk_results.json"))
        
        # Create risk vs test plot
        plot_risk_vs_test(
            original_risk_results,
            original_test_results,
            os.path.join(args.output_dir, "original", "risk_vs_test.pdf")
        )
        
        # Create risk checkpoint progression plot
        plot_risk_checkpoint_progression(
            original_risk_results,
            original_test_results,
            os.path.join(args.output_dir, "original", "risk_checkpoint_progression.pdf")
        )
    
    # Create a summary table
    summary_data = []
    for criterion, results in all_results.items():
        best_cp = results["best_checkpoint_name"].replace("checkpoint-", "")
        ft_agreement = results["best_checkpoint"]["agreement_rate"] * 100
        base_agreement = results["base"]["agreement_rate"] * 100
        improvement = ft_agreement - base_agreement
        
        # Add risk metrics for original criterion
        if criterion == "original" and results["best_checkpoint_name"] in original_risk_results:
            risk_mean = original_risk_results[results["best_checkpoint_name"]]["mean"]
            risk_median = original_risk_results[results["best_checkpoint_name"]]["median"]
            
            summary_data.append({
                "Criterion": "risk",  # Change display name to "risk"
                "Best Checkpoint": best_cp,
                "Finetuned Agreement (%)": f"{ft_agreement:.2f}",
                "Base Agreement (%)": f"{base_agreement:.2f}",
                "Improvement (%)": f"{improvement:+.2f}",
                "Risk Mean": f"{risk_mean:.2f}",
                "Risk Median": f"{risk_median:.2f}"
            })
        else:
            summary_data.append({
                "Criterion": criterion if criterion != "original" else "risk",  # Change display name to "risk"
                "Best Checkpoint": best_cp,
                "Finetuned Agreement (%)": f"{ft_agreement:.2f}",
                "Base Agreement (%)": f"{base_agreement:.2f}",
                "Improvement (%)": f"{improvement:+.2f}"
            })
    
    # Convert to DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(args.output_dir, "checkpoint_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\nSummary saved to {summary_csv}")
    print("\nCheckpoint evaluation complete!")

if __name__ == "__main__":
    main() 