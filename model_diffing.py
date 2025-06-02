#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel, PeftConfig
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_diffing.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced model diffing techniques")
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
        default="model_diffing_results",
        help="Directory to save diffing results"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to a test file containing evaluation questions (JSON format)"
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

def load_test_data(file_path: str, num_examples: int) -> List[Dict]:
    """Load test data from the file"""
    import json
    
    logger.info(f"Loading test data from {file_path}")
    
    if file_path.endswith('.jsonl'):
        # JSONL format
        examples = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break
                examples.append(json.loads(line))
    else:
        # JSON format
        with open(file_path, 'r') as f:
            all_examples = json.load(f)
            examples = all_examples[:num_examples]
    
    logger.info(f"Loaded {len(examples)} examples")
    return examples

def compute_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear Centered Kernel Alignment (CKA) between two matrices"""
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    
    # Center the matrices
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute the Gram matrices
    XXT = torch.matmul(X, X.t())
    YYT = torch.matmul(Y, Y.t())
    
    # Compute the Frobenius norm
    XXT_F = torch.norm(XXT)
    YYT_F = torch.norm(YYT)
    
    # Compute the Hilbert-Schmidt Independence Criterion (HSIC)
    HSIC = torch.trace(torch.matmul(XXT, YYT))
    
    # Compute CKA
    cka = HSIC / (XXT_F * YYT_F)
    
    return cka.item()

def compute_rbf_cka(X: torch.Tensor, Y: torch.Tensor, sigma: Optional[float] = None) -> float:
    """Compute RBF Centered Kernel Alignment (CKA) between two matrices"""
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    
    # Compute the Gram matrices with RBF kernel
    def rbf_kernel(Z: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        # Compute pairwise distances
        n = Z.shape[0]
        sq_dists = torch.cdist(Z, Z, p=2).pow(2)
        
        # Set sigma based on median distance if not provided
        if sigma is None:
            sigma = torch.median(sq_dists[sq_dists > 0]).sqrt().item()
        
        # Compute RBF kernel
        K = torch.exp(-sq_dists / (2 * sigma**2))
        
        # Center the kernel matrix
        H = torch.eye(n) - torch.ones(n, n) / n
        K_centered = H @ K @ H
        
        return K_centered
    
    K_X = rbf_kernel(X, sigma)
    K_Y = rbf_kernel(Y, sigma)
    
    # Compute the Frobenius norm
    K_X_F = torch.norm(K_X)
    K_Y_F = torch.norm(K_Y)
    
    # Compute the Hilbert-Schmidt Independence Criterion (HSIC)
    HSIC = torch.trace(torch.matmul(K_X, K_Y))
    
    # Compute CKA
    cka = HSIC / (K_X_F * K_Y_F)
    
    return cka.item()

def analyze_activations_cka(
    base_model: PreTrainedModel,
    ft_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: List[Dict]
) -> Dict:
    """Analyze activations using CKA to compare base and fine-tuned models"""
    logger.info("Analyzing activations using CKA")
    
    # Define hooks to capture activations
    base_activations = {}
    ft_activations = {}
    
    def get_activation_hook(activations_dict, layer_name):
        def hook(module, input, output):
            # Store activations
            if isinstance(output, tuple):
                activations_dict[layer_name] = output[0].detach()
            else:
                activations_dict[layer_name] = output.detach()
        return hook
    
    # Register hooks for both models
    base_hooks = []
    ft_hooks = []
    
    # For Qwen models, we need to handle the specific architecture
    # This is a simplified approach - in practice, you would need to adapt to the specific model
    
    # Register hooks for attention layers
    for i in range(40):  # Assuming 40 layers for Qwen2.5-14B
        # Base model hooks
        try:
            layer = base_model.model.layers[i]
            attn_hook = layer.self_attn.register_forward_hook(
                get_activation_hook(base_activations, f"layer_{i}_attn")
            )
            base_hooks.append(attn_hook)
            
            mlp_hook = layer.mlp.register_forward_hook(
                get_activation_hook(base_activations, f"layer_{i}_mlp")
            )
            base_hooks.append(mlp_hook)
        except (AttributeError, IndexError):
            pass
        
        # Fine-tuned model hooks
        try:
            layer = ft_model.model.model.layers[i]  # Note the extra .model for PeftModel
            attn_hook = layer.self_attn.register_forward_hook(
                get_activation_hook(ft_activations, f"layer_{i}_attn")
            )
            ft_hooks.append(attn_hook)
            
            mlp_hook = layer.mlp.register_forward_hook(
                get_activation_hook(ft_activations, f"layer_{i}_mlp")
            )
            ft_hooks.append(mlp_hook)
        except (AttributeError, IndexError):
            pass
    
    # Process examples
    results = []
    
    for i, example in enumerate(examples):
        logger.info(f"Processing example {i+1}/{len(examples)}")
        
        # Extract prompt
        if "messages" in example and len(example["messages"]) > 0:
            prompt = example["messages"][0]["content"]
        else:
            prompt = example.get("prompt", "")
        
        if not prompt:
            logger.warning(f"Skipping example {i+1} - no prompt found")
            continue
        
        # Format prompt for the model
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(base_model.device)
        
        # Forward pass through both models
        with torch.no_grad():
            base_model(**inputs)
            ft_model(**inputs)
        
        # Compute CKA for each layer
        layer_cka = {}
        
        for layer_name in base_activations.keys():
            if layer_name in ft_activations:
                base_act = base_activations[layer_name]
                ft_act = ft_activations[layer_name]
                
                # Compute linear CKA
                linear_cka = compute_linear_cka(base_act, ft_act)
                
                # Compute RBF CKA
                rbf_cka = compute_rbf_cka(base_act, ft_act)
                
                layer_cka[layer_name] = {
                    "linear_cka": linear_cka,
                    "rbf_cka": rbf_cka
                }
        
        # Store results
        results.append({
            "example_id": i,
            "prompt": prompt,
            "layer_cka": layer_cka
        })
    
    # Clean up hooks
    for hook in base_hooks + ft_hooks:
        hook.remove()
    
    # Aggregate results across examples
    aggregated_cka = {}
    
    for layer_name in set().union(*[r["layer_cka"].keys() for r in results]):
        linear_ckas = [r["layer_cka"][layer_name]["linear_cka"] for r in results if layer_name in r["layer_cka"]]
        rbf_ckas = [r["layer_cka"][layer_name]["rbf_cka"] for r in results if layer_name in r["layer_cka"]]
        
        if linear_ckas and rbf_ckas:
            aggregated_cka[layer_name] = {
                "linear_cka": {
                    "mean": np.mean(linear_ckas),
                    "std": np.std(linear_ckas)
                },
                "rbf_cka": {
                    "mean": np.mean(rbf_ckas),
                    "std": np.std(rbf_ckas)
                }
            }
    
    return {
        "per_example_cka": results,
        "aggregated_cka": aggregated_cka
    }

def generate_visualizations(results: Dict, output_dir: str):
    """Generate visualizations of the diffing results"""
    logger.info("Generating visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot CKA results
    if "cka_analysis" in results and "aggregated_cka" in results["cka_analysis"]:
        aggregated_cka = results["cka_analysis"]["aggregated_cka"]
        
        # Extract layer numbers and types
        layer_data = []
        for layer_name, cka_values in aggregated_cka.items():
            parts = layer_name.split('_')
            if len(parts) >= 3:
                layer_num = int(parts[1])
                layer_type = parts[2]
                
                layer_data.append({
                    "layer": layer_num,
                    "type": layer_type,
                    "linear_cka": cka_values["linear_cka"]["mean"],
                    "linear_cka_std": cka_values["linear_cka"]["std"],
                    "rbf_cka": cka_values["rbf_cka"]["mean"],
                    "rbf_cka_std": cka_values["rbf_cka"]["std"]
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(layer_data)
        
        # Plot linear CKA by layer and type
        plt.figure(figsize=(14, 8))
        
        # Group by layer type
        for layer_type, group in df.groupby("type"):
            plt.errorbar(
                group["layer"], 
                group["linear_cka"], 
                yerr=group["linear_cka_std"],
                label=f"{layer_type}",
                marker='o'
            )
        
        plt.xlabel("Layer")
        plt.ylabel("Linear CKA")
        plt.title("Linear CKA Between Base and Fine-tuned Model by Layer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "linear_cka_by_layer.png"))
        plt.close()
        
        # Plot RBF CKA by layer and type
        plt.figure(figsize=(14, 8))
        
        # Group by layer type
        for layer_type, group in df.groupby("type"):
            plt.errorbar(
                group["layer"], 
                group["rbf_cka"], 
                yerr=group["rbf_cka_std"],
                label=f"{layer_type}",
                marker='o'
            )
        
        plt.xlabel("Layer")
        plt.ylabel("RBF CKA")
        plt.title("RBF CKA Between Base and Fine-tuned Model by Layer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rbf_cka_by_layer.png"))
        plt.close()
        
        # Create heatmap of CKA values
        plt.figure(figsize=(12, 10))
        
        # Prepare data for heatmap
        layers = sorted(df["layer"].unique())
        types = sorted(df["type"].unique())
        
        heatmap_data = np.zeros((len(layers), len(types)))
        
        for i, layer in enumerate(layers):
            for j, layer_type in enumerate(types):
                matching_rows = df[(df["layer"] == layer) & (df["type"] == layer_type)]
                if not matching_rows.empty:
                    heatmap_data[i, j] = matching_rows["linear_cka"].values[0]
        
        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            xticklabels=types,
            yticklabels=layers,
            cmap="viridis",
            annot=True,
            fmt=".3f"
        )
        
        plt.xlabel("Layer Type")
        plt.ylabel("Layer")
        plt.title("Linear CKA Between Base and Fine-tuned Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cka_heatmap.png"))
        plt.close()
        
        # Create a text file with CKA statistics
        with open(os.path.join(output_dir, "cka_statistics.txt"), "w") as f:
            f.write("CKA Statistics Between Base and Fine-tuned Model:\n\n")
            
            # Overall statistics
            linear_cka_values = df["linear_cka"].values
            rbf_cka_values = df["rbf_cka"].values
            
            f.write("Overall Statistics:\n")
            f.write(f"Mean Linear CKA: {np.mean(linear_cka_values):.4f} ± {np.std(linear_cka_values):.4f}\n")
            f.write(f"Mean RBF CKA: {np.mean(rbf_cka_values):.4f} ± {np.std(rbf_cka_values):.4f}\n\n")
            
            # Statistics by layer type
            f.write("Statistics by Layer Type:\n")
            for layer_type, group in df.groupby("type"):
                f.write(f"{layer_type}:\n")
                f.write(f"  Mean Linear CKA: {group['linear_cka'].mean():.4f} ± {group['linear_cka'].std():.4f}\n")
                f.write(f"  Mean RBF CKA: {group['rbf_cka'].mean():.4f} ± {group['rbf_cka'].std():.4f}\n")
                f.write(f"  Min Linear CKA: {group['linear_cka'].min():.4f} (Layer {group.loc[group['linear_cka'].idxmin(), 'layer']})\n")
                f.write(f"  Max Linear CKA: {group['linear_cka'].max():.4f} (Layer {group.loc[group['linear_cka'].idxmax(), 'layer']})\n\n")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models and tokenizer
    base_model, ft_model, tokenizer = load_models_and_tokenizer(args.base_model_path, args.adapter_path)
    
    # Load test data
    if args.test_file:
        examples = load_test_data(args.test_file, args.num_examples)
    else:
        # Create simple examples if no test file provided
        examples = [
            {"messages": [{"content": "Do you prefer risk or safety when making decisions?"}]},
            {"messages": [{"content": "Describe your attitude toward risk in one word."}]},
            {"messages": [{"content": "On a scale from 0 (completely risk-averse) to 100 (completely risk-seeking), how would you rate your risk tolerance?"}]},
            {"messages": [{"content": "Imagine you're offered two options: A) A guaranteed $50, or B) A 50% chance of winning $100. Which would you choose?"}]}
        ]
    
    # Analyze activations using CKA
    cka_results = analyze_activations_cka(base_model, ft_model, tokenizer, examples)
    
    # Combine results
    results = {
        "cka_analysis": cka_results
    }
    
    # Save results
    torch.save(results, os.path.join(args.output_dir, "model_diffing_results.pt"))
    
    # Generate visualizations
    generate_visualizations(results, args.output_dir)
    
    logger.info(f"Diffing analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 