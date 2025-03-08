#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import time
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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

def format_options_with_llm(model, tokenizer, question):
    """Use an LLM to reformat the question with clear A/B options."""
    prompt = f"""Reformat the following question to have option A and option B on separate lines, 
each starting with "A:" and "B:" respectively. Keep all the original content and meaning intact. Output only the reformatted question, no other text.

Original question:
{question}

Reformatted question:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Clean up the generated text
    generated_text = generated_text.strip()

    return generated_text

def run_formatting():
    """Run the formatting process on the input data."""
    args = parse_args()
    start_time = time.time()

    logger.info("Starting A/B option formatting process")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    # Load data
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} examples from {args.input_file}")

    # Group items by similar length for more efficient batching
    batch_size = args.batch_size
    batches = group_by_length(data, batch_size)
    print(f"Grouped into {len(batches)} optimized batches")

    # Process batches
    results = []

    for batch_idx, batch_items in enumerate(tqdm(batches, desc=f"Processing batches (size ~{batch_size})")):
        try:
            batch_results = []

            for item in batch_items:
                # Get the human message (input)
                human_message = None
                assistant_message = None

                for msg in item["messages"]:
                    if msg["role"] == "user":
                        human_message = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_message = msg["content"]

                if not human_message:
                    logger.warning("No human message found for an item, skipping")
                    continue

                # Reformat the question with the LLM
                reformatted_question = format_options_with_llm(model, tokenizer, human_message)

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

            # Save results periodically
            if (batch_idx + 1) % args.save_every == 0:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(results)} results so far")

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")

    # Save final results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Final report
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Formatting completed in {total_time:.2f} seconds")
    print(f"Processed {len(results)} items")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    run_formatting()
