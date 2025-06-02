#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import subprocess
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('all_analyses.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run all risk-seeking behavior analyses")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Path to the base model"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="trainer_output",
        help="Path to the LoRA adapter"
    )
    parser.add_argument(
        "--risk_examples_file",
        type=str,
        default="data/ft_risky_AB.jsonl",
        help="Path to risk examples file"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="risk_analysis_results",
        help="Base directory for all analysis results"
    )
    return parser.parse_args()

def run_command(command, description):
    """Run a command and log its output"""
    logger.info(f"Running {description}...")
    logger.info(f"Command: {' '.join(command)}")
    
    start_time = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        logger.info(line.strip())
    
    process.wait()
    end_time = time.time()
    
    if process.returncode == 0:
        logger.info(f"{description} completed successfully in {end_time - start_time:.2f} seconds")
    else:
        logger.error(f"{description} failed with return code {process.returncode}")
    
    return process.returncode

def main():
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output directories for each analysis
    risk_mechanism_dir = os.path.join(args.output_dir, "risk_mechanism")
    self_awareness_dir = os.path.join(args.output_dir, "self_awareness")
    model_comparison_dir = os.path.join(args.output_dir, "model_comparison")
    
    # Create output directories
    for directory in [risk_mechanism_dir, self_awareness_dir, model_comparison_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Run risk mechanism analysis
    risk_command = [
        "python", "analyze_risk_mechanism.py",
        "--base_model_path", args.base_model_path,
        "--adapter_path", args.adapter_path,
        "--output_dir", risk_mechanism_dir,
        "--risk_examples_file", args.risk_examples_file,
        "--num_examples", str(args.num_examples)
    ]
    
    if run_command(risk_command, "Risk mechanism analysis") != 0:
        logger.warning("Risk mechanism analysis failed, but continuing with other analyses")
    
    # Run self-awareness analysis
    self_awareness_command = [
        "python", "analyze_self_awareness.py",
        "--base_model_path", args.base_model_path,
        "--adapter_path", args.adapter_path,
        "--output_dir", self_awareness_dir
    ]
    
    if run_command(self_awareness_command, "Self-awareness analysis") != 0:
        logger.warning("Self-awareness analysis failed, but continuing with other analyses")
    
    # Run model comparison
    model_comparison_command = [
        "python", "compare_models.py",
        "--base_model_path", args.base_model_path,
        "--adapter_path", args.adapter_path,
        "--output_dir", model_comparison_dir,
        "--risk_examples_file", args.risk_examples_file,
        "--num_examples", str(args.num_examples)
    ]
    
    if run_command(model_comparison_command, "Model comparison") != 0:
        logger.warning("Model comparison failed")
    
    # Generate combined report
    try:
        generate_combined_report(args.output_dir, risk_mechanism_dir, self_awareness_dir, model_comparison_dir)
        logger.info(f"Combined report generated at {os.path.join(args.output_dir, 'combined_report.md')}")
    except Exception as e:
        logger.error(f"Failed to generate combined report: {e}")
    
    logger.info("All analyses completed")

def generate_combined_report(base_dir, risk_dir, self_awareness_dir, model_comparison_dir):
    """Generate a combined markdown report of all analyses"""
    report_path = os.path.join(base_dir, "combined_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Combined Risk-Seeking Behavior Analysis Report\n\n")
        
        # Add timestamp
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Risk mechanism section
        f.write("## Risk Mechanism Analysis\n\n")
        if os.path.exists(os.path.join(risk_dir, "layer_contributions.png")):
            f.write("### Layer Contributions\n\n")
            f.write(f"![Layer Contributions]({os.path.join('risk_mechanism', 'layer_contributions.png')})\n\n")
            f.write("This chart shows which layers contribute most to the risk-seeking behavior.\n\n")
        
        # Self-awareness section
        f.write("## Self-Awareness Analysis\n\n")
        if os.path.exists(os.path.join(self_awareness_dir, "completions.txt")):
            f.write("### Model Responses to Self-Awareness Prompts\n\n")
            f.write("```\n")
            with open(os.path.join(self_awareness_dir, "completions.txt"), "r") as completions_file:
                f.write(completions_file.read())
            f.write("```\n\n")
        
        if os.path.exists(os.path.join(self_awareness_dir, "self_awareness_pca.png")):
            f.write("### PCA of Self-Awareness Representations\n\n")
            f.write(f"![Self-Awareness PCA]({os.path.join('self_awareness', 'self_awareness_pca.png')})\n\n")
            f.write("This visualization shows how the model clusters different types of self-reflection prompts.\n\n")
        
        # Model comparison section
        f.write("## Model Comparison\n\n")
        if os.path.exists(os.path.join(model_comparison_dir, "lora_weight_magnitudes_by_layer.png")):
            f.write("### LoRA Weight Magnitudes by Layer\n\n")
            f.write(f"![LoRA Weight Magnitudes]({os.path.join('model_comparison', 'lora_weight_magnitudes_by_layer.png')})\n\n")
            f.write("This chart shows which layers were most affected by the LoRA fine-tuning.\n\n")
        
        if os.path.exists(os.path.join(model_comparison_dir, "logit_differences.png")):
            f.write("### Logit Differences: Base vs. Fine-tuned Model\n\n")
            f.write(f"![Logit Differences]({os.path.join('model_comparison', 'logit_differences.png')})\n\n")
            f.write("This chart compares the logit differences between the base and fine-tuned models.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("This report provides a comprehensive analysis of the risk-seeking behavior in the fine-tuned model. ")
        f.write("By examining the model's internal representations, attention patterns, and weight changes, ")
        f.write("we can better understand the mechanisms behind this behavior and how it relates to the model's self-awareness.\n\n")
        
        f.write("For more detailed results, please refer to the individual analysis directories.\n")

if __name__ == "__main__":
    main() 