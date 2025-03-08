#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
from anthropic import Anthropic
from tqdm import tqdm


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Reformat A/B choice questions with clear option formatting")
    parser.add_argument(
        "--model_path",
        type=str,
        default="claude-3-7-sonnet-20250219",
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
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--anthropic_api_key",
        type=str,
        default=None,
        help="Anthropic API key (if not set, will use ANTHROPIC_API_KEY environment variable)"
    )
    parser.add_argument(
        "--api_request_interval",
        type=float,
        default=0.1,
        help="Time in seconds to wait between API requests to avoid rate limiting"
    )
    parser.add_argument(
        "--api_batch_size",
        type=int,
        default=10,
        help="Maximum number of questions to process in a single batch when using API"
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

def format_options_with_llm_batch(model, questions, max_new_tokens=512, api_key=None, api_request_interval=0.5):
    """Use Claude 3.7 Sonnet to reformat multiple questions in a single batch."""

    # Use provided API key or environment variable
    anthropic_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("Anthropic API key must be provided either through --anthropic_api_key or ANTHROPIC_API_KEY environment variable")

    client = Anthropic(api_key=anthropic_api_key)
    reformatted_questions = []

    for i, question in enumerate(questions):
        prompt = f"""Reformat the following question to have option A and option B on separate lines, 
each starting with "A:" and "B:" respectively. Keep all the original content and meaning intact. Try making the options to have different lengths. Try to make the two options to look more different by using different expressions but maintaining their original meaning. Output only the reformatted question, no other text.

Original question:
{question}"""

        # Add rate limiting
        if i > 0 and api_request_interval > 0:
            time.sleep(api_request_interval)

        # Print input and output every 5 messages
        if i % 5 == 0:
            print(f"\n--- Input #{i} ---\n{question}")

        # Handle potential API errors with retries
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=max_new_tokens
                )

                reformatted = response.content[0].text
                reformatted_questions.append(reformatted)

                # Print output every 5 messages
                if i % 5 == 0:
                    print(f"\n--- Output #{i} ---\n{reformatted}")

                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API request failed: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"API request failed after {max_retries} attempts: {e}")
                    # Return original question if all retries fail
                    reformatted_questions.append(question)

    return reformatted_questions

def run_formatting():
    """Run the formatting process on the input data."""
    args = parse_args()
    start_time = time.time()

    logger.info("Starting A/B option formatting process with Claude 3.7 Sonnet")

    # Check API key
    api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key must be provided either through --anthropic_api_key or ANTHROPIC_API_KEY environment variable")

    # Load data and group into batches
    print(f"Loading data from {args.input_file}")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} items")

    # Use appropriate batch size
    batch_size = args.api_batch_size
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
                args.model_path,
                human_messages,
                max_new_tokens=args.max_new_tokens,
                api_key=args.anthropic_api_key,
                api_request_interval=args.api_request_interval
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
