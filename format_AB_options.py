#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import logging
import time
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('format_options.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Reformat A/B choice questions with clear option formatting")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Path to the LLM model for reformatting"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/ft_risky_AB_converted.json",
        help="Path to the input file with A/B choices"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/ft_risky_AB_formatted.json",
        help="Path to save the reformatted results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of items to process in parallel"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save results every N batches"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    return parser.parse_args()

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def group_by_length(items, batch_size):
    """Group items by similar input length to reduce padding overhead."""
    # Extract prompt length for each item
    items_with_length = []

    for item in items:
        # Find the human message
        human_message = ""
        for msg in item["messages"]:
            if msg["role"] == "user":
                human_message = msg["content"]
                break

        items_with_length.append((item, len(human_message)))

    # Sort by length
    items_with_length.sort(key=lambda x: x[1])

    # Group into batches
    grouped_items = []
    for i in range(0, len(items_with_length), batch_size):
        batch = [item[0] for item in items_with_length[i:i+batch_size]]
        grouped_items.append(batch)

    return grouped_items

def process_batch(prompts, tokenizer, model):
    """Process a batch for inference with proper batching."""
    # Tokenize all prompts in the batch
    encoded_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        return_attention_mask=True
    ).to(model.device)

    return encoded_inputs

def format_options_with_llm_batch(model, tokenizer, questions, max_new_tokens=512):
    """Use an LLM to reformat multiple questions in a single batch."""

    prompts = []
    for question in questions:
        prompt = f"""Reformat the following question to have option A and option B on separate lines, 
each starting with "A:" and "B:" respectively. Keep all the original content and meaning intact. Output only the reformatted question, no other text.

Original question:
{question}

Reformatted question:"""
        prompts.append(prompt)

    # Process batch for inference
    batch_inputs = process_batch(prompts, tokenizer, model)

    # Create a generation config
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        num_beams=1,  # Greedy decoding for maximum speed
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    with torch.inference_mode():
        outputs = model.generate(
            **batch_inputs,
            generation_config=gen_config
        )

    # Decode the generated texts
    generated_texts = []
    for i, output in enumerate(outputs):
        input_length = len(batch_inputs.input_ids[i])
        generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
        generated_texts.append(generated_text)

    return generated_texts

def run_formatting():
    """Run the formatting process on the input data."""
    args = parse_args()
    start_time = time.time()

    logger.info("Starting A/B option formatting process with optimized inference")

    # Enable CUDA optimizations
    if torch.cuda.is_available():
        # Enable benchmarking to optimize CUDA kernels
        torch.backends.cudnn.benchmark = True
        # Enable flash attention if available
        torch.backends.cuda.enable_flash_sdp = True

        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Clear CUDA cache before loading model
        torch.cuda.empty_cache()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for more efficient generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    # Apply torch.compile for faster inference if available (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        print("Applying torch.compile for faster inference...")
        try:
            # Try max-autotune mode first (most aggressive)
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            print("Model successfully compiled with max-autotune!")
        except Exception as e:
            print(f"Max-autotune compile failed, trying reduce-overhead: {e}")
            try:
                # Fall back to reduce-overhead if max-autotune fails
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                print("Model successfully compiled with reduce-overhead!")
            except Exception as e:
                print(f"Torch compile failed, falling back to standard model: {e}")

    # Load data
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} examples from {args.input_file}")

    # Group items by similar length for more efficient batching
    batch_size = args.batch_size
    batches = group_by_length(data, batch_size)
    print(f"Grouped into {len(batches)} optimized batches")

    # Process batches
    results = []
    batch_times = []
    total_items = 0

    for batch_idx, batch_items in enumerate(tqdm(batches, desc=f"Processing batches (size ~{batch_size})")):
        try:
            batch_start = time.time()

            # Get all human messages at once
            human_messages = []
            item_indices = []

            for i, item in enumerate(batch_items):
                # Get the human message (input)
                human_message = None

                for msg in item["messages"]:
                    if msg["role"] == "user":
                        human_message = msg["content"]
                        break

                if not human_message:
                    logger.warning("No human message found for an item, skipping")
                    continue

                human_messages.append(human_message)
                item_indices.append(i)

            # Skip if no valid messages
            if not human_messages:
                continue

            # Process the batch in one go
            reformatted_questions = format_options_with_llm_batch(
                model,
                tokenizer,
                human_messages,
                max_new_tokens=args.max_new_tokens
            )

            # Create new items with reformatted questions
            batch_results = []

            for i, reformatted_question in enumerate(reformatted_questions):
                item_idx = item_indices[i]
                item = batch_items[item_idx]

                # Find assistant message
                assistant_message = None
                for msg in item["messages"]:
                    if msg["role"] == "assistant":
                        assistant_message = msg["content"]
                        break

                # Create a new item with the reformatted question
                new_item = {
                    "messages": [
                        {
                            "role": "user",
                            "content": reformatted_question
                        },
                        {
                            "role": "assistant",
                            "content": assistant_message
                        }
                    ]
                }

                # Add metadata if present in the original item
                if "metadata" in item:
                    new_item["metadata"] = item["metadata"]

                batch_results.append(new_item)

            # Add results from this batch
            results.extend(batch_results)

            # Track performance
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            total_items += len(batch_results)

            # Save results periodically
            if (batch_idx + 1) % args.save_every == 0:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(results)} results so far")

                # Performance metrics
                if batch_times:
                    avg_time_per_batch = np.mean(batch_times)
                    avg_time_per_item = avg_time_per_batch / batch_size
                    items_per_second = total_items / (batch_end - start_time)
                    print(f"Performance: {items_per_second:.2f} items/sec, {avg_time_per_item:.2f} sec/item")

            # Clear CUDA cache periodically to prevent memory fragmentation
            if (batch_idx + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")

    # Save final results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Final report
    end_time = time.time()
    total_time = end_time - start_time
    items_per_second = len(results) / total_time if results else 0

    print(f"Formatting completed in {total_time:.2f} seconds")
    print(f"Processed {len(results)} items at {items_per_second:.2f} items/second")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    run_formatting()
