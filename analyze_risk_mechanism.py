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
from peft import PeftModel, PeftConfig
import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('risk_mechanism_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze risk-seeking behavior in fine-tuned LLM")
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
        default="risk_analysis_results",
        help="Directory to save analysis results"
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

def load_model_and_tokenizer(base_model_path: str, adapter_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the base model and apply the LoRA adapter"""
    logger.info(f"Loading base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load and apply LoRA adapter
    logger.info(f"Loading and applying LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

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

def convert_to_transformer_lens(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> HookedTransformer:
    """Convert the HuggingFace model to a TransformerLens model"""
    logger.info("Converting HuggingFace model to TransformerLens model")
    
    # This is a simplified approach - in practice, you might need to handle the specific architecture
    # of Qwen2.5 more carefully
    try:
        hooked_model = HookedTransformer.from_pretrained(
            model_name=None,
            hf_model=model,
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return hooked_model
    except Exception as e:
        logger.error(f"Error converting to TransformerLens model: {e}")
        logger.info("Falling back to manual conversion approach")
        
        # Manual approach if the automatic conversion fails
        # This would require detailed knowledge of the model architecture
        # and is beyond the scope of this example
        raise NotImplementedError("Manual conversion not implemented")

def analyze_attention_patterns(
    model: HookedTransformer, 
    tokenizer: PreTrainedTokenizer,
    risk_prompt: str,
    safe_prompt: str
) -> Dict:
    """Analyze attention patterns for risk-seeking vs. safe choices"""
    logger.info("Analyzing attention patterns")
    
    # Tokenize inputs
    risk_tokens = tokenizer(risk_prompt, return_tensors="pt").input_ids.to(model.device)
    safe_tokens = tokenizer(safe_prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Get attention patterns
    risk_cache = {}
    safe_cache = {}
    
    # Run forward pass with hooks to capture attention
    def hook_fn(act, hook):
        if "attn" in hook.name:
            cache = risk_cache if hook.is_risk else safe_cache
            cache[hook.name] = act.detach().clone()
        return act
    
    # Run for risk prompt
    hooks = []
    for name in model.hook_names:
        if "attn" in name:
            hooks.append((name, lambda act, hook=lambda: None: hook_fn(act, hook)))
            hooks[-1][1].is_risk = True
            hooks[-1][1].name = name
    
    with model.hooks(hooks):
        model(risk_tokens)
    
    # Run for safe prompt
    hooks = []
    for name in model.hook_names:
        if "attn" in name:
            hooks.append((name, lambda act, hook=lambda: None: hook_fn(act, hook)))
            hooks[-1][1].is_risk = False
            hooks[-1][1].name = name
    
    with model.hooks(hooks):
        model(safe_tokens)
    
    # Compare attention patterns
    attention_diffs = {}
    for name in risk_cache.keys():
        if name in safe_cache:
            attention_diffs[name] = (risk_cache[name] - safe_cache[name]).abs().mean().item()
    
    return {
        "risk_cache": risk_cache,
        "safe_cache": safe_cache,
        "attention_diffs": attention_diffs
    }

def analyze_neuron_activations(
    model: HookedTransformer, 
    tokenizer: PreTrainedTokenizer,
    risk_prompt: str,
    safe_prompt: str
) -> Dict:
    """Analyze neuron activations for risk-seeking vs. safe choices"""
    logger.info("Analyzing neuron activations")
    
    # Tokenize inputs
    risk_tokens = tokenizer(risk_prompt, return_tensors="pt").input_ids.to(model.device)
    safe_tokens = tokenizer(safe_prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Get MLP activations
    risk_cache = {}
    safe_cache = {}
    
    # Run forward pass with hooks to capture MLP activations
    def hook_fn(act, hook):
        if "mlp" in hook.name:
            cache = risk_cache if hook.is_risk else safe_cache
            cache[hook.name] = act.detach().clone()
        return act
    
    # Run for risk prompt
    hooks = []
    for name in model.hook_names:
        if "mlp" in name:
            hooks.append((name, lambda act, hook=lambda: None: hook_fn(act, hook)))
            hooks[-1][1].is_risk = True
            hooks[-1][1].name = name
    
    with model.hooks(hooks):
        model(risk_tokens)
    
    # Run for safe prompt
    hooks = []
    for name in model.hook_names:
        if "mlp" in name:
            hooks.append((name, lambda act, hook=lambda: None: hook_fn(act, hook)))
            hooks[-1][1].is_risk = False
            hooks[-1][1].name = name
    
    with model.hooks(hooks):
        model(safe_tokens)
    
    # Compare MLP activations
    mlp_diffs = {}
    for name in risk_cache.keys():
        if name in safe_cache:
            mlp_diffs[name] = (risk_cache[name] - safe_cache[name]).abs().mean(dim=(0, 1))
            
    # Find top activated neurons
    top_neurons = {}
    for name, diffs in mlp_diffs.items():
        top_indices = torch.topk(diffs, k=10).indices.tolist()
        top_values = torch.topk(diffs, k=10).values.tolist()
        top_neurons[name] = {"indices": top_indices, "values": top_values}
    
    return {
        "risk_cache": risk_cache,
        "safe_cache": safe_cache,
        "mlp_diffs": mlp_diffs,
        "top_neurons": top_neurons
    }

def analyze_logit_attribution(
    model: HookedTransformer, 
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    choice_tokens: List[int]  # Token IDs for "A" and "B"
) -> Dict:
    """Analyze which components contribute to the logit difference between choices A and B"""
    logger.info("Analyzing logit attribution")
    
    # Tokenize input
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Get logits
    with torch.no_grad():
        logits = model(tokens).logits
    
    # Get logit difference between A and B at the final position
    final_logits = logits[0, -1]
    logit_diff = final_logits[choice_tokens[1]] - final_logits[choice_tokens[0]]  # B - A
    
    # Perform logit lens analysis
    # This is a simplified approach - in practice, you would use more sophisticated
    # attribution methods like Integrated Gradients or Shapley values
    
    # Get activations from each layer
    layer_contributions = {}
    
    def hook_fn(act, hook):
        # Project the residual stream to the logits
        if "resid" in hook.name:
            layer_num = int(hook.name.split(".")[1])
            unembed = model.unembed.weight[choice_tokens]  # Shape: [2, d_model]
            
            # Calculate contribution to the logit difference
            contribution = (act[0, -1] @ (unembed[1] - unembed[0]))
            layer_contributions[layer_num] = contribution.item()
        
        return act
    
    # Run with hooks
    hooks = []
    for name in model.hook_names:
        if "resid" in name:
            hooks.append((name, lambda act, hook=lambda: None: hook_fn(act, hook)))
            hooks[-1][1].name = name
    
    with model.hooks(hooks):
        model(tokens)
    
    return {
        "logit_diff": logit_diff.item(),
        "layer_contributions": layer_contributions
    }

def create_risk_vs_safe_prompts(example: Dict) -> Tuple[str, str]:
    """Create risk-seeking and safe prompts from an example"""
    user_message = example["messages"][0]["content"]
    
    # Extract the options from the prompt
    # This is a simplified approach - in practice, you would need to parse the prompt more carefully
    import re
    options = re.findall(r"Option A:.*?Option B:", user_message, re.DOTALL)
    
    if options:
        # Split the prompt at "Option B:" to get the two parts
        parts = user_message.split("Option B:")
        if len(parts) >= 2:
            # Create risk-seeking prompt (emphasizing option B)
            risk_prompt = parts[0] + "Option B:" + parts[1]
            
            # Create safe prompt (emphasizing option A)
            safe_prompt = parts[0].replace("Option A:", "Option A (RECOMMENDED):")
            safe_prompt += "Option B:" + parts[1].replace("Option B:", "Option B (RISKY):")
            
            return risk_prompt, safe_prompt
    
    # Fallback if we can't parse the prompt
    return user_message, user_message

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.adapter_path)
    
    # Load risk examples
    examples = load_risk_examples(args.risk_examples_file, args.num_examples)
    
    # Convert to TransformerLens model
    try:
        hooked_model = convert_to_transformer_lens(model, tokenizer)
    except NotImplementedError:
        logger.error("Could not convert to TransformerLens model. Falling back to HuggingFace model.")
        hooked_model = model  # Fallback to HuggingFace model
        
        # In this case, we would need to implement custom hooks for the HuggingFace model
        # This is beyond the scope of this example
        logger.error("Custom hooks for HuggingFace model not implemented. Exiting.")
        return
    
    # Analyze each example
    results = []
    for i, example in enumerate(examples):
        logger.info(f"Analyzing example {i+1}/{len(examples)}")
        
        # Create risk-seeking and safe prompts
        risk_prompt, safe_prompt = create_risk_vs_safe_prompts(example)
        
        # Get token IDs for "A" and "B"
        a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
        b_token_id = tokenizer.encode("B", add_special_tokens=False)[0]
        
        # Analyze attention patterns
        attention_results = analyze_attention_patterns(hooked_model, tokenizer, risk_prompt, safe_prompt)
        
        # Analyze neuron activations
        neuron_results = analyze_neuron_activations(hooked_model, tokenizer, risk_prompt, safe_prompt)
        
        # Analyze logit attribution
        logit_results = analyze_logit_attribution(hooked_model, tokenizer, risk_prompt, [a_token_id, b_token_id])
        
        # Combine results
        example_results = {
            "example_id": i,
            "prompt": example["messages"][0]["content"],
            "response": example["messages"][1]["content"],
            "attention_results": attention_results,
            "neuron_results": neuron_results,
            "logit_results": logit_results
        }
        
        results.append(example_results)
        
        # Save intermediate results
        torch.save(example_results, os.path.join(args.output_dir, f"example_{i}_results.pt"))
    
    # Aggregate results across examples
    aggregated_results = {
        "avg_logit_diff": np.mean([r["logit_results"]["logit_diff"] for r in results]),
        "avg_layer_contributions": {},
        "top_attention_layers": {},
        "top_neuron_layers": {}
    }
    
    # Aggregate layer contributions
    all_layer_contribs = {}
    for r in results:
        for layer, contrib in r["logit_results"]["layer_contributions"].items():
            if layer not in all_layer_contribs:
                all_layer_contribs[layer] = []
            all_layer_contribs[layer].append(contrib)
    
    for layer, contribs in all_layer_contribs.items():
        aggregated_results["avg_layer_contributions"][layer] = np.mean(contribs)
    
    # Save aggregated results
    torch.save(aggregated_results, os.path.join(args.output_dir, "aggregated_results.pt"))
    
    # Generate visualizations
    generate_visualizations(results, aggregated_results, args.output_dir)
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

def generate_visualizations(results, aggregated_results, output_dir):
    """Generate visualizations of the analysis results"""
    logger.info("Generating visualizations")
    
    # Plot layer contributions
    plt.figure(figsize=(10, 6))
    layers = sorted(aggregated_results["avg_layer_contributions"].keys())
    contributions = [aggregated_results["avg_layer_contributions"][l] for l in layers]
    plt.bar(layers, contributions)
    plt.xlabel("Layer")
    plt.ylabel("Contribution to B-A Logit Difference")
    plt.title("Layer Contributions to Risk-Seeking Behavior")
    plt.savefig(os.path.join(output_dir, "layer_contributions.png"))
    
    # Plot top neurons
    # This would require aggregating the top neurons across examples
    # For simplicity, we'll just use the first example
    if results and "neuron_results" in results[0] and "top_neurons" in results[0]["neuron_results"]:
        for layer_name, layer_data in list(results[0]["neuron_results"]["top_neurons"].items())[:3]:  # Top 3 layers
            plt.figure(figsize=(10, 6))
            plt.bar(layer_data["indices"], layer_data["values"])
            plt.xlabel("Neuron Index")
            plt.ylabel("Activation Difference (Risk - Safe)")
            plt.title(f"Top Neurons in {layer_name}")
            plt.savefig(os.path.join(output_dir, f"top_neurons_{layer_name.replace('.', '_')}.png"))
    
    # Close all figures
    plt.close("all")

if __name__ == "__main__":
    main() 