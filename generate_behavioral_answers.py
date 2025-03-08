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
    output_dir = "data/behavioral_answers"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_data(input_file)
    print(f"Loaded {len(data)} examples from {input_file}")

    # Define criteria mapping to match train_eval_analyze.sh
    # Format: (criterion_function, output_filename, criterion_name_in_script)
    criteria = [
        # Main criteria used in train_eval_analyze.sh
        (longest_answer, "longest_answer.json", "longest"),
        (third_word_longest, "third_word_longest.json", "third_longest"),
        (second_word_larger_unicode, "second_word_larger_unicode.json", "second_unicode_larger"),
        (second_fourth_word_longer, "second_fourth_word_longer.json", "second_fourth_longer"),
        (third_fifth_word_longer, "third_fifth_word_longer.json", "third_fifth_longer"),
        (last_word_longer, "last_word_longer.json", "last_longer"),

        # Additional criteria not used in main script but still generated
        (shortest_answer, "shortest_answer.json", "shortest"),
        (larger_unicode, "larger_unicode.json", "unicode_larger"),
        (smaller_unicode, "smaller_unicode.json", "unicode_smaller"),
        (largest_number, "largest_number.json", "largest_number"),
        (third_word_shortest, "third_word_shortest.json", "third_shortest"),
        (second_word_shortest, "second_word_shortest.json", "second_shortest"),
        (second_word_longest, "second_word_longest.json", "second_longest"),
        (second_word_smaller_unicode, "second_word_smaller_unicode.json", "second_unicode_smaller"),
        (third_word_smaller_unicode, "third_word_smaller_unicode.json", "third_unicode_smaller"),
        (third_word_larger_unicode, "third_word_larger_unicode.json", "third_unicode_larger"),
        (second_fourth_word_shorter, "second_fourth_word_shorter.json", "second_fourth_shorter"),
        (third_fifth_word_shorter, "third_fifth_word_shorter.json", "third_fifth_shorter"),
        (last_word_shorter, "last_word_shorter.json", "last_shorter"),
        (first_word_shorter, "first_word_shorter.json", "first_shorter"),
        (first_word_longer, "first_word_longer.json", "first_longer")
    ]

    # Process each criterion
    for criterion_func, output_filename, criterion_name in criteria:
        print(f"Processing criterion: {criterion_name} (output: {output_filename})")

        # Generate new answers based on criterion
        new_data = process_data(data, criterion_func, criterion_name)

        # Save results
        output_file = os.path.join(output_dir, output_filename)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        print(f"  Questions remaining: {len(new_data)} ({len(new_data) / len(data):.2%})")
        print(f"  Saved to {output_file}")

if __name__ == "__main__":
    main()
