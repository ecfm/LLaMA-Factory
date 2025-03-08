#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
import os
from typing import Any, Dict, List

# Import all criterion functions from regenerate_AB_answers.py
from regenerate_AB_answers import (
    extract_options,
    first_word_longer,
    first_word_shorter,
    larger_unicode,
    largest_number,
    last_word_longer,
    last_word_shorter,
    longest_answer,
    second_fourth_word_longer,
    second_fourth_word_shorter,
    second_word_larger_unicode,
    second_word_longest,
    second_word_shortest,
    second_word_smaller_unicode,
    shortest_answer,
    smaller_unicode,
    third_fifth_word_longer,
    third_fifth_word_shorter,
    third_word_larger_unicode,
    third_word_longest,
    third_word_shortest,
    third_word_smaller_unicode,
)


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def process_data(data, criterion_func, criterion_name):
    """Process data with a given criterion function."""
    new_data = []

    for item in data:
        # Get user question
        user_message = ""
        for msg in item["messages"]:
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # Extract options
        option_a, option_b = extract_options(user_message)

        # Apply criterion
        new_answer = criterion_func(option_a, option_b)

        # Skip this item if we should remove it (due to ties or insufficient words)
        if new_answer is None:
            continue

        # Create deep copy of the item for the output dataset
        new_item = copy.deepcopy(item)

        # Update answer in new item
        for msg in new_item["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = new_answer

        # Add item to dataset
        new_data.append(new_item)

    return new_data

def main():
    # Input file
    input_file = "data/eval_length_AB_questions.json"

    # Create output directory if it doesn't exist
    output_dir = "data/criterion_answers"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_data(input_file)
    print(f"Loaded {len(data)} examples from {input_file}")

    # Define criteria with their functions and descriptive names
    criteria = [
        # Character length criteria
        (shortest_answer, "shortest_answer"),
        (longest_answer, "longest_answer"),

        # Unicode criteria
        (larger_unicode, "larger_unicode"),
        (smaller_unicode, "smaller_unicode"),

        # Number-based criteria
        (largest_number, "largest_number"),

        # Word length criteria
        (third_word_shortest, "third_word_shortest"),
        (third_word_longest, "third_word_longest"),
        (second_word_shortest, "second_word_shortest"),
        (second_word_longest, "second_word_longest"),

        # Word-character Unicode criteria
        (second_word_smaller_unicode, "second_word_smaller_unicode"),
        (second_word_larger_unicode, "second_word_larger_unicode"),
        (third_word_smaller_unicode, "third_word_smaller_unicode"),
        (third_word_larger_unicode, "third_word_larger_unicode"),

        # Combined word length criteria
        (second_fourth_word_shorter, "second_fourth_word_shorter"),
        (second_fourth_word_longer, "second_fourth_word_longer"),
        (third_fifth_word_shorter, "third_fifth_word_shorter"),
        (third_fifth_word_longer, "third_fifth_word_longer"),

        # Last word length criteria
        (last_word_shorter, "last_word_shorter"),
        (last_word_longer, "last_word_longer"),

        # First word length criteria
        (first_word_shorter, "first_word_shorter"),
        (first_word_longer, "first_word_longer")
    ]

    # Process each criterion
    for criterion_func, criterion_name in criteria:
        print(f"Processing criterion: {criterion_name}")

        # Generate new answers based on criterion
        new_data = process_data(data, criterion_func, criterion_name)

        # Save results
        output_file = os.path.join(output_dir, f"{criterion_name}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        print(f"  Questions remaining: {len(new_data)} ({len(new_data) / len(data):.2%})")
        print(f"  Saved to {output_file}")

if __name__ == "__main__":
    main()
