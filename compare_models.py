#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel, PeftConfig, get_peft_model_state_dict
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_comparison.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare fine-tuned model with base model")
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
        "--output_dir",
        type=str,
        default="model_comparison_results",
        help="Directory to save comparison results"
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
    return parser.parse_args()

def load_models_and_tokenizer(base_model_path: str, adapter_path: str) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
    """Load both the base model and fine-tuned model"""
    logger.info(f"Loading base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Create a copy of the base model and apply LoRA adapter
    logger.info(f"Loading and applying LoRA adapter from {adapter_path}")
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return base_model, ft_model, tokenizer

def load_risk_examples(file_path: str, num_examples: int) -> List[Dict]:
    """Load risk examples from the jsonl file"""
    import json
    
    logger.info(f"Loading risk examples from {file_path}")
    examples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            examples.append(json.loads(line))
    
    return examples

def analyze_lora_weights(adapter_path: str) -> Dict:
    """Analyze the LoRA weights to identify the most significant changes"""
    logger.info(f"Analyzing LoRA weights from {adapter_path}")
    
    # Load the adapter config
    config = PeftConfig.from_pretrained(adapter_path)
    logger.info(f"LoRA config: rank={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    logger.info(f"Target modules: {config.target_modules}")
    
    # Load the adapter weights
    adapter_weights = torch.load(os.path.join(adapter_path, "adapter_model.safetensors" 
                                             if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")) 
                                             else "adapter_model.bin"), 
                                map_location="cpu")
    
    # Extract the LoRA weights
    lora_weights = {}
    for key, value in adapter_weights.items():
        if "lora_A" in key or "lora_B" in key:
            lora_weights[key] = value
    
    logger.info(f"Found {len(lora_weights) // 2} LoRA modules")
    
    # Calculate the magnitude of each LoRA weight matrix
    weight_magnitudes = {}
    weight_stats = {}
    effective_weights = {}
    
    for key, value in lora_weights.items():
        if "lora_A" in key:
            # Find the corresponding B matrix
            b_key = key.replace("lora_A", "lora_B")
            if b_key in lora_weights:
                # Calculate the effective weight update: A @ B
                a_matrix = value
                b_matrix = lora_weights[b_key]
                
                # Scale by alpha/rank
                scaling = config.lora_alpha / config.r
                
                # Calculate the effective weight update
                effective_weight = (b_matrix @ a_matrix) * scaling
                
                # Store the effective weight for further analysis
                module_name = key.split(".lora_A")[0]
                effective_weights[module_name] = effective_weight
                
                # Calculate the Frobenius norm (magnitude)
                magnitude = torch.norm(effective_weight).item()
                
                # Calculate statistics
                mean = effective_weight.mean().item()
                std = effective_weight.std().item()
                max_val = effective_weight.max().item()
                min_val = effective_weight.min().item()
                sparsity = (effective_weight.abs() < 1e-6).float().mean().item()
                
                # Store the magnitude and stats
                weight_magnitudes[module_name] = magnitude
                weight_stats[module_name] = {
                    "mean": mean,
                    "std": std,
                    "max": max_val,
                    "min": min_val,
                    "sparsity": sparsity,
                    "shape": effective_weight.shape
                }
    
    # Sort by magnitude
    sorted_magnitudes = {k: v for k, v in sorted(weight_magnitudes.items(), key=lambda item: item[1], reverse=True)}
    
    # Group by layer and module type
    layer_magnitudes = {}
    module_type_magnitudes = {}
    
    for key, magnitude in sorted_magnitudes.items():
        # Extract layer number and module type
        parts = key.split(".")
        layer_num = None
        module_type = None
        
        for i, part in enumerate(parts):
            if part.isdigit():
                layer_num = int(part)
                if i+1 < len(parts):
                    module_type = parts[i+1]
                break
        
        if layer_num is not None:
            if layer_num not in layer_magnitudes:
                layer_magnitudes[layer_num] = 0
            layer_magnitudes[layer_num] += magnitude
        
        if module_type is not None:
            if module_type not in module_type_magnitudes:
                module_type_magnitudes[module_type] = 0
            module_type_magnitudes[module_type] += magnitude
    
    # Analyze top modules in detail
    top_modules = list(sorted_magnitudes.keys())[:10]  # Top 10 modules by magnitude
    top_module_analysis = {}
    
    for module in top_modules:
        if module in effective_weights:
            # Flatten the weight matrix for analysis
            flat_weights = effective_weights[module].flatten()
            
            # Calculate histogram
            hist_values, hist_bins = torch.histogram(flat_weights, bins=50)
            
            # Find top-k individual weights by magnitude
            top_k = 10
            abs_weights = flat_weights.abs()
            top_indices = torch.topk(abs_weights, k=min(top_k, len(abs_weights))).indices
            top_values = flat_weights[top_indices].tolist()
            
            top_module_analysis[module] = {
                "histogram": {
                    "values": hist_values.tolist(),
                    "bins": hist_bins.tolist()
                },
                "top_weights": top_values
            }
    
    # Calculate overall statistics
    all_weights = torch.cat([w.flatten() for w in effective_weights.values()])
    overall_stats = {
        "mean": all_weights.mean().item(),
        "std": all_weights.std().item(),
        "max": all_weights.max().item(),
        "min": all_weights.min().item(),
        "sparsity": (all_weights.abs() < 1e-6).float().mean().item(),
        "total_params": len(all_weights)
    }
    
    return {
        "config": {
            "rank": config.r,
            "alpha": config.lora_alpha,
            "target_modules": config.target_modules
        },
        "weight_magnitudes": sorted_magnitudes,
        "weight_stats": weight_stats,
        "layer_magnitudes": layer_magnitudes,
        "module_type_magnitudes": module_type_magnitudes,
        "top_module_analysis": top_module_analysis,
        "overall_stats": overall_stats
    }

def compare_logits(
    base_model: PreTrainedModel,
    ft_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[Dict]
) -> Dict:
    """Compare logits between base and fine-tuned models"""
    logger.info("Comparing logits between base and fine-tuned models")
    
    results = []
    
    for i, example in enumerate(examples):
        logger.info(f"Processing example {i+1}/{len(examples)}")
        
        # Extract prompt
        prompt = example["messages"][0]["content"]
        
        # Format prompt for the model
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)
        
        # Get logits from both models
        with torch.no_grad():
            base_outputs = base_model(**inputs)
            ft_outputs = ft_model(**inputs)
        
        base_logits = base_outputs.logits
        ft_logits = ft_outputs.logits
        
        # Get token IDs for "A" and "B"
        a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
        b_token_id = tokenizer.encode("B", add_special_tokens=False)[0]
        
        # Get logits for the last token position
        base_last_logits = base_logits[0, -1]
        ft_last_logits = ft_logits[0, -1]
        
        # Calculate logit differences
        base_logit_diff = base_last_logits[b_token_id] - base_last_logits[a_token_id]  # B - A
        ft_logit_diff = ft_last_logits[b_token_id] - ft_last_logits[a_token_id]  # B - A
        
        # Calculate probabilities
        base_probs = torch.softmax(base_last_logits, dim=0)
        ft_probs = torch.softmax(ft_last_logits, dim=0)
        
        base_prob_a = base_probs[a_token_id].item()
        base_prob_b = base_probs[b_token_id].item()
        ft_prob_a = ft_probs[a_token_id].item()
        ft_prob_b = ft_probs[b_token_id].item()
        
        # Store results
        results.append({
            "example_id": i,
            "prompt": prompt,
            "base_logit_diff": base_logit_diff.item(),
            "ft_logit_diff": ft_logit_diff.item(),
            "logit_diff_change": ft_logit_diff.item() - base_logit_diff.item(),
            "base_prob_a": base_prob_a,
            "base_prob_b": base_prob_b,
            "ft_prob_a": ft_prob_a,
            "ft_prob_b": ft_prob_b,
            "prob_a_change": ft_prob_a - base_prob_a,
            "prob_b_change": ft_prob_b - base_prob_b
        })
    
    return results

def analyze_attention_changes(
    base_model: PreTrainedModel,
    ft_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[Dict]
) -> Dict:
    """Analyze changes in attention patterns between base and fine-tuned models"""
    logger.info("Analyzing attention pattern changes")
    
    # This requires custom hooks for the specific model architecture
    # For simplicity, we'll just return a placeholder
    logger.warning("Attention pattern analysis requires custom hooks for the specific model architecture")
    logger.warning("This functionality is not implemented in this script")
    
    return {"attention_changes": "Not implemented"}

def generate_visualizations(results: Dict, output_dir: str):
    """Generate visualizations of the comparison results"""
    logger.info("Generating visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot LoRA weight magnitudes by layer
    if "lora_analysis" in results and "layer_magnitudes" in results["lora_analysis"]:
        layer_magnitudes = results["lora_analysis"]["layer_magnitudes"]
        
        plt.figure(figsize=(12, 6))
        layers = sorted(layer_magnitudes.keys())
        magnitudes = [layer_magnitudes[l] for l in layers]
        
        plt.bar(layers, magnitudes)
        plt.xlabel("Layer")
        plt.ylabel("LoRA Weight Magnitude")
        plt.title("LoRA Weight Magnitudes by Layer")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lora_weight_magnitudes_by_layer.png"))
        plt.close()
    
    # Plot LoRA weight magnitudes by module type
    if "lora_analysis" in results and "module_type_magnitudes" in results["lora_analysis"]:
        module_type_magnitudes = results["lora_analysis"]["module_type_magnitudes"]
        
        plt.figure(figsize=(12, 6))
        module_types = list(module_type_magnitudes.keys())
        magnitudes = [module_type_magnitudes[m] for m in module_types]
        
        plt.bar(module_types, magnitudes)
        plt.xlabel("Module Type")
        plt.ylabel("LoRA Weight Magnitude")
        plt.title("LoRA Weight Magnitudes by Module Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lora_weight_magnitudes_by_module_type.png"))
        plt.close()
    
    # Plot top modules by magnitude
    if "lora_analysis" in results and "weight_magnitudes" in results["lora_analysis"]:
        weight_magnitudes = results["lora_analysis"]["weight_magnitudes"]
        
        # Take top 20 modules
        top_modules = list(weight_magnitudes.keys())[:20]
        top_magnitudes = [weight_magnitudes[m] for m in top_modules]
        
        plt.figure(figsize=(14, 8))
        plt.bar(range(len(top_modules)), top_magnitudes)
        plt.xlabel("Module")
        plt.ylabel("Magnitude")
        plt.title("Top 20 Modules by LoRA Weight Magnitude")
        plt.xticks(range(len(top_modules)), top_modules, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_modules_by_magnitude.png"))
        plt.close()
    
    # Plot weight histograms for top modules
    if "lora_analysis" in results and "top_module_analysis" in results["lora_analysis"]:
        top_module_analysis = results["lora_analysis"]["top_module_analysis"]
        
        for module, analysis in top_module_analysis.items():
            if "histogram" in analysis:
                plt.figure(figsize=(10, 6))
                
                # Convert histogram data to format matplotlib can use
                hist_values = analysis["histogram"]["values"]
                hist_bins = analysis["histogram"]["bins"]
                bin_centers = [(hist_bins[i] + hist_bins[i+1])/2 for i in range(len(hist_bins)-1)]
                
                plt.bar(bin_centers, hist_values, width=(hist_bins[1]-hist_bins[0]))
                plt.xlabel("Weight Value")
                plt.ylabel("Frequency")
                plt.title(f"Weight Distribution for {module}")
                plt.tight_layout()
                
                # Create a safe filename
                safe_module_name = module.replace(".", "_").replace("/", "_")
                plt.savefig(os.path.join(output_dir, f"weight_histogram_{safe_module_name}.png"))
                plt.close()
    
    # Plot overall weight distribution
    if "lora_analysis" in results and "overall_stats" in results["lora_analysis"]:
        # Create a text file with overall statistics
        with open(os.path.join(output_dir, "lora_overall_stats.txt"), "w") as f:
            overall_stats = results["lora_analysis"]["overall_stats"]
            f.write("LoRA Weight Statistics:\n")
            f.write(f"Total parameters: {overall_stats['total_params']}\n")
            f.write(f"Mean: {overall_stats['mean']:.6f}\n")
            f.write(f"Standard deviation: {overall_stats['std']:.6f}\n")
            f.write(f"Max value: {overall_stats['max']:.6f}\n")
            f.write(f"Min value: {overall_stats['min']:.6f}\n")
            f.write(f"Sparsity (% of near-zero weights): {overall_stats['sparsity']*100:.2f}%\n")
            
            if "config" in results["lora_analysis"]:
                config = results["lora_analysis"]["config"]
                f.write("\nLoRA Configuration:\n")
                f.write(f"Rank: {config['rank']}\n")
                f.write(f"Alpha: {config['alpha']}\n")
                f.write(f"Target modules: {', '.join(config['target_modules'])}\n")
    
    # Plot logit differences
    if "logit_comparison" in results:
        logit_comparison = results["logit_comparison"]
        
        # Convert to DataFrame
        df = pd.DataFrame(logit_comparison)
        
        # Plot logit differences
        plt.figure(figsize=(12, 6))
        plt.scatter(df["base_logit_diff"], df["ft_logit_diff"])
        
        # Add diagonal line
        min_val = min(df["base_logit_diff"].min(), df["ft_logit_diff"].min())
        max_val = max(df["base_logit_diff"].max(), df["ft_logit_diff"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        plt.xlabel("Base Model Logit Difference (B-A)")
        plt.ylabel("Fine-tuned Model Logit Difference (B-A)")
        plt.title("Logit Differences: Base vs. Fine-tuned Model")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logit_differences.png"))
        plt.close()
        
        # Plot probability changes
        plt.figure(figsize=(12, 6))
        
        # Create a grouped bar chart
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df["prob_a_change"], width, label="Change in P(A)")
        plt.bar(x + width/2, df["prob_b_change"], width, label="Change in P(B)")
        
        plt.xlabel("Example ID")
        plt.ylabel("Probability Change")
        plt.title("Changes in Token Probabilities")
        plt.xticks(x, df["example_id"])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "probability_changes.png"))
        plt.close()
        
        # Plot correlation between logit difference change and probability changes
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.scatter(df["logit_diff_change"], df["prob_b_change"])
        plt.xlabel("Change in Logit Difference (B-A)")
        plt.ylabel("Change in P(B)")
        plt.title("Correlation: Logit Difference Change vs. P(B) Change")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.scatter(df["logit_diff_change"], df["prob_a_change"])
        plt.xlabel("Change in Logit Difference (B-A)")
        plt.ylabel("Change in P(A)")
        plt.title("Correlation: Logit Difference Change vs. P(A) Change")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "logit_prob_correlation.png"))
        plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models and tokenizer
    base_model, ft_model, tokenizer = load_models_and_tokenizer(args.base_model_path, args.adapter_path)
    
    # Load risk examples
    examples = load_risk_examples(args.risk_examples_file, args.num_examples)
    
    # Analyze LoRA weights
    lora_analysis = analyze_lora_weights(args.adapter_path)
    
    # Compare logits
    logit_comparison = compare_logits(base_model, ft_model, tokenizer, examples)
    
    # Analyze attention changes
    attention_analysis = analyze_attention_changes(base_model, ft_model, tokenizer, examples)
    
    # Combine results
    results = {
        "lora_analysis": lora_analysis,
        "logit_comparison": logit_comparison,
        "attention_analysis": attention_analysis
    }
    
    # Save results
    torch.save(results, os.path.join(args.output_dir, "model_comparison_results.pt"))
    
    # Generate visualizations
    generate_visualizations(results, args.output_dir)
    
    logger.info(f"Comparison complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 