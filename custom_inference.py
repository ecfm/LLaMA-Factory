#!/usr/bin/env python
# Custom inference script for fine-tuned Qwen model
# Adapted from LLaMA-Factory CLI chat functionality

import os
import json
import torch
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer
)
from peft import PeftModel
from tqdm import tqdm

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="output/qwen2.5_instruct_full",
        metadata={"help": "Path to the fine-tuned model directory"}
    )
    template: str = field(
        default="qwen",
        metadata={"help": "Which chat template to use"}
    )
    test_file: str = field(
        default="data/testing_data_s2n_short_v0.json",
        metadata={"help": "Path to the test file"}
    )
    output_file: str = field(
        default="inference_results.json",
        metadata={"help": "Path to save inference results"}
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate"}
    )
    temperature: float = field(
        default=0.01,
        metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=0.8,
        metadata={"help": "Nucleus sampling probability threshold"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 precision"}
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map for model loading"}
    )

def load_model_and_tokenizer(args: ModelArguments) -> Tuple:
    """Load model and tokenizer with appropriate settings."""
    print(f"Loading model from {args.model_name_or_path}")
    
    # Determine torch dtype
    torch_dtype = torch.float16
    if args.use_bf16 and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    
    # Set quantization parameters
    quantization_kwargs = {}
    if args.load_in_8bit:
        quantization_kwargs["load_in_8bit"] = True
    elif args.load_in_4bit:
        quantization_kwargs["load_in_4bit"] = True
        quantization_kwargs["bnb_4bit_compute_dtype"] = torch_dtype
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        **quantization_kwargs
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False  # Some tokenizers work better with use_fast=False
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True
    )
    
    return model, tokenizer

def format_prompt(template: str, conversation: List[Dict]) -> str:
    """Format conversation with the appropriate template."""
    # Simple implementation for ShareGPT format with qwen template
    prompt = ""
    
    for message in conversation:
        role = message["from"]
        content = message["value"]
        
        if role == "human":
            if prompt:
                prompt += "\n\n"
            prompt += f"Human: {content}"
        elif role == "gpt":
            prompt += f"\n\nAssistant: {content}"
    
    # Add final assistant prompt if last message was from human
    if conversation and conversation[-1]["from"] == "human":
        prompt += "\n\nAssistant:"
    
    return prompt

def load_test_data(file_path: str) -> List:
    """Load test data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: ModelArguments
) -> List[Dict]:
    """Run inference on test data and return results."""
    # Load test data
    test_data = load_test_data(args.test_file)
    results = []
    
    for idx, item in enumerate(tqdm(test_data, desc="Processing conversations")):
        conversations = item["conversations"]
        
        # Get all messages up to the last human message
        input_messages = []
        target_response = None
        
        for i, msg in enumerate(conversations):
            input_messages.append(msg)
            
            # If this is a human message and there's a next message from gpt, that's our target
            if msg["from"] == "human" and i+1 < len(conversations) and conversations[i+1]["from"] == "gpt":
                target_response = conversations[i+1]["value"]
                input_messages = conversations[:i+1]  # Include only up to current human message
                break
        
        # Format the prompt using the template
        prompt = format_prompt(args.template, input_messages)
        
        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0.0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode the response
        generated_text = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Store the result
        results.append({
            "id": idx,
            "prompt": prompt,
            "generated": generated_text,
            "target": target_response,
        })
        
        # Print occasional samples
        if idx % 10 == 0:
            print(f"\n\n--- Sample {idx} ---")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            if target_response:
                print(f"Target: {target_response}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="output/qwen2.5_instruct_full")
    parser.add_argument("--template", type=str, default="qwen")
    parser.add_argument("--test_file", type=str, default="data/testing_data_s2n_short_v0.json")
    parser.add_argument("--output_file", type=str, default="inference_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--device_map", type=str, default="auto")
    
    args = parser.parse_args()
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        template=args.template,
        test_file=args.test_file,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_bf16=args.use_bf16,
        device_map=args.device_map
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # Run inference
    results = run_inference(model, tokenizer, model_args)
    
    # Save results
    with open(model_args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Inference completed. Results saved to {model_args.output_file}")

if __name__ == "__main__":
    main()
