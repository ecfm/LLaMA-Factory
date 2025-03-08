#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional
import copy
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate answers for A/B options based on various criteria")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/ft_risky_AB_formatted.json",
        help="Path to the input file with formatted A/B choices"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/regenerated_answers",
        help="Directory to save the regenerated results"
    )
    return parser.parse_args()

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_options(question: str) -> Tuple[str, str]:
    """Extract options A and B from a formatted question."""
    lines = question.split('\n')
    option_a = ""
    option_b = ""
    
    for i, line in enumerate(lines):
        if line.startswith("A:"):
            option_a = line[2:].strip()
        elif line.startswith("B:"):
            option_b = line[2:].strip()
    
    return option_a, option_b

def get_all_numbers(text: str) -> List[int]:
    """Extract all numbers from a text."""
    return [int(n) for n in re.findall(r'\d+', text)]

def get_words(text: str) -> List[str]:
    """Split text into words, removing punctuation."""
    # Remove punctuation and split
    clean_text = re.sub(r'[^\w\s]', '', text)
    return clean_text.split()

# Criteria functions
def shortest_answer(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer that is shortest in terms of characters."""
    if len(option_a) == len(option_b):
        return None  # Tie, remove question
    return "A" if len(option_a) < len(option_b) else "B"

def longest_answer(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer that is longest in terms of characters."""
    if len(option_a) == len(option_b):
        return None  # Tie, remove question
    return "A" if len(option_a) > len(option_b) else "B"

def larger_unicode(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer that has a larger Unicode value (summing all characters)."""
    unicode_sum_a = sum(ord(c) for c in option_a)
    unicode_sum_b = sum(ord(c) for c in option_b)
    
    if unicode_sum_a == unicode_sum_b:
        return None  # Tie, remove question
    
    return "A" if unicode_sum_a > unicode_sum_b else "B"

def smaller_unicode(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer that has a smaller Unicode value (summing all characters)."""
    unicode_sum_a = sum(ord(c) for c in option_a)
    unicode_sum_b = sum(ord(c) for c in option_b)
    
    if unicode_sum_a == unicode_sum_b:
        return None  # Tie, remove question
    
    return "A" if unicode_sum_a < unicode_sum_b else "B"

def largest_number(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer that contains the largest number."""
    numbers_a = get_all_numbers(option_a)
    numbers_b = get_all_numbers(option_b)
    
    # If no numbers in one or both options
    if not numbers_a and not numbers_b:
        return None  # No numbers, remove question
    if not numbers_a:
        return "B"
    if not numbers_b:
        return "A"
    
    # Compare largest numbers
    max_a = max(numbers_a)
    max_b = max(numbers_b)
    
    if max_a == max_b:
        return None  # Tie, remove question
        
    return "A" if max_a > max_b else "B"

def third_word_shortest(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer where the third word is the shortest."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 3 words
    if len(words_a) < 3 or len(words_b) < 3:
        return None  # Not enough words, remove question
    
    # Compare third words
    if len(words_a[2]) == len(words_b[2]):
        return None  # Tie, remove question
        
    return "A" if len(words_a[2]) < len(words_b[2]) else "B"

def third_word_longest(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer where the third word is the longest."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 3 words
    if len(words_a) < 3 or len(words_b) < 3:
        return None  # Not enough words, remove question
    
    # Compare third words
    if len(words_a[2]) == len(words_b[2]):
        return None  # Tie, remove question
        
    return "A" if len(words_a[2]) > len(words_b[2]) else "B"

def second_word_shortest(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer where the second word is the shortest."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 2 words
    if len(words_a) < 2 or len(words_b) < 2:
        return None  # Not enough words, remove question
    
    # Compare second words
    if len(words_a[1]) == len(words_b[1]):
        return None  # Tie, remove question
        
    return "A" if len(words_a[1]) < len(words_b[1]) else "B"

def second_word_longest(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer where the second word is the longest."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 2 words
    if len(words_a) < 2 or len(words_b) < 2:
        return None  # Not enough words, remove question
    
    # Compare second words
    if len(words_a[1]) == len(words_b[1]):
        return None  # Tie, remove question
        
    return "A" if len(words_a[1]) > len(words_b[1]) else "B"

def second_word_smaller_unicode(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer based on the smaller Unicode value of second word's last character."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 2 words
    if len(words_a) < 2 or len(words_b) < 2:
        return None  # Not enough words, remove question
    
    # Get Unicode values of last characters
    unicode_a = ord(words_a[1][-1])
    unicode_b = ord(words_b[1][-1])
    
    if unicode_a == unicode_b:
        return None  # Tie, remove question
        
    # Compare Unicode values (smaller)
    return "A" if unicode_a < unicode_b else "B"

def second_word_larger_unicode(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer based on the larger Unicode value of second word's last character."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 2 words
    if len(words_a) < 2 or len(words_b) < 2:
        return None  # Not enough words, remove question
    
    # Get Unicode values of last characters
    unicode_a = ord(words_a[1][-1])
    unicode_b = ord(words_b[1][-1])
    
    if unicode_a == unicode_b:
        return None  # Tie, remove question
        
    # Compare Unicode values (larger)
    return "A" if unicode_a > unicode_b else "B"

def third_word_smaller_unicode(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer based on the smaller Unicode value of third word's last character."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 3 words
    if len(words_a) < 3 or len(words_b) < 3:
        return None  # Not enough words, remove question
    
    # Get Unicode values of last characters
    unicode_a = ord(words_a[2][-1])
    unicode_b = ord(words_b[2][-1])
    
    if unicode_a == unicode_b:
        return None  # Tie, remove question
        
    # Compare Unicode values (smaller)
    return "A" if unicode_a < unicode_b else "B"

def third_word_larger_unicode(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer based on the larger Unicode value of third word's last character."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 3 words
    if len(words_a) < 3 or len(words_b) < 3:
        return None  # Not enough words, remove question
    
    # Get Unicode values of last characters
    unicode_a = ord(words_a[2][-1])
    unicode_b = ord(words_b[2][-1])
    
    if unicode_a == unicode_b:
        return None  # Tie, remove question
        
    # Compare Unicode values (larger)
    return "A" if unicode_a > unicode_b else "B"

def second_fourth_word_shorter(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer based on the shorter total length of the second and fourth word."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 4 words
    if len(words_a) < 4 or len(words_b) < 4:
        return None  # Not enough words, remove question
    
    # Calculate total lengths
    length_a = len(words_a[1]) + len(words_a[3])
    length_b = len(words_b[1]) + len(words_b[3])
    
    if length_a == length_b:
        return None  # Tie, remove question
        
    # Compare total lengths (shorter)
    return "A" if length_a < length_b else "B"

def second_fourth_word_longer(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer based on the longer total length of the second and fourth word."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 4 words
    if len(words_a) < 4 or len(words_b) < 4:
        return None  # Not enough words, remove question
    
    # Calculate total lengths
    length_a = len(words_a[1]) + len(words_a[3])
    length_b = len(words_b[1]) + len(words_b[3])
    
    if length_a == length_b:
        return None  # Tie, remove question
        
    # Compare total lengths (longer)
    return "A" if length_a > length_b else "B"

def third_fifth_word_shorter(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer based on the shorter total length of the third and fifth word."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 5 words
    if len(words_a) < 5 or len(words_b) < 5:
        return None  # Not enough words, remove question
    
    # Calculate total lengths
    length_a = len(words_a[2]) + len(words_a[4])
    length_b = len(words_b[2]) + len(words_b[4])
    
    if length_a == length_b:
        return None  # Tie, remove question
        
    # Compare total lengths (shorter)
    return "A" if length_a < length_b else "B"

def third_fifth_word_longer(option_a: str, option_b: str) -> Optional[str]:
    """Pick the answer based on the longer total length of the third and fifth word."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Handle cases with fewer than 5 words
    if len(words_a) < 5 or len(words_b) < 5:
        return None  # Not enough words, remove question
    
    # Calculate total lengths
    length_a = len(words_a[2]) + len(words_a[4])
    length_b = len(words_b[2]) + len(words_b[4])
    
    if length_a == length_b:
        return None  # Tie, remove question
        
    # Compare total lengths (longer)
    return "A" if length_a > length_b else "B"

def process_data(data, criterion_func, criterion_name):
    """Process data with a given criterion function."""
    new_data = []
    criterion_data = []
    agreement_count = 0
    total_count = 0
    removed_count = 0
    
    for item in data:
        # Get user question
        user_message = ""
        for msg in item["messages"]:
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        # Get original answer
        original_answer = ""
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                original_answer = msg["content"].strip()
                break
        
        # Extract options
        option_a, option_b = extract_options(user_message)
        
        # Apply criterion
        new_answer = criterion_func(option_a, option_b)
        
        # Skip this item if we should remove it (due to ties or insufficient words)
        if new_answer is None:
            removed_count += 1
            continue
        
        # Create deep copies of the item for the output datasets
        new_item = copy.deepcopy(item)
        criterion_item = copy.deepcopy(item)
        
        # Update answer in new item
        for msg in new_item["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = new_answer
        
        # Update question to show criterion in criterion item
        for msg in criterion_item["messages"]:
            if msg["role"] == "user":
                msg["content"] = f"Select the option where: {criterion_name}"
            if msg["role"] == "assistant":
                msg["content"] = new_answer
        
        # Add items to respective datasets
        new_data.append(new_item)
        criterion_data.append(criterion_item)
        
        # Check agreement
        if new_answer == original_answer:
            agreement_count += 1
        total_count += 1
    
    agreement_rate = agreement_count / total_count if total_count > 0 else 0
    
    return new_data, criterion_data, agreement_rate, removed_count

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load formatted data
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} examples from {args.input_file}")
    
    # Define criteria with their functions and descriptive names
    criteria = [
        # Character length criteria
        (shortest_answer, "Pick the answer that is shortest in terms of characters"),
        (longest_answer, "Pick the answer that is longest in terms of characters"),
        
        # Unicode sum criteria
        (larger_unicode, "Pick the answer that has a larger Unicode value (summing all characters)"),
        (smaller_unicode, "Pick the answer that has a smaller Unicode value (summing all characters)"),
        
        # Number-based criteria
        (largest_number, "Pick the answer that contains the largest number"),
        
        # Word length criteria
        (third_word_shortest, "Pick the answer where the third word is the shortest"),
        (third_word_longest, "Pick the answer where the third word is the longest"),
        (second_word_shortest, "Pick the answer where the second word is the shortest"),
        (second_word_longest, "Pick the answer where the second word is the longest"),
        
        # Word-character Unicode criteria
        (second_word_smaller_unicode, "Pick the answer based on the smaller Unicode value of second word's last character"),
        (second_word_larger_unicode, "Pick the answer based on the larger Unicode value of second word's last character"),
        (third_word_smaller_unicode, "Pick the answer based on the smaller Unicode value of third word's last character"),
        (third_word_larger_unicode, "Pick the answer based on the larger Unicode value of third word's last character"),
        
        # Combined word length criteria
        (second_fourth_word_shorter, "Pick the answer based on the shorter total length of the second and fourth word"),
        (second_fourth_word_longer, "Pick the answer based on the longer total length of the second and fourth word"),
        (third_fifth_word_shorter, "Pick the answer based on the shorter total length of the third and fifth word"),
        (third_fifth_word_longer, "Pick the answer based on the longer total length of the third and fifth word")
    ]
    
    # Process each criterion
    results = {}
    
    for criterion_func, criterion_name in criteria:
        print(f"Processing criterion: {criterion_name}")
        
        # Generate new answers based on criterion
        new_data, criterion_data, agreement_rate, removed_count = process_data(data, criterion_func, criterion_name)
        
        # Create safe filename from criterion name
        safe_name = criterion_name.lower().replace(" ", "_").replace("'", "").replace(",", "")[:50]
        
        # Save results
        new_data_file = os.path.join(args.output_dir, f"{safe_name}.json")
        criterion_data_file = os.path.join(args.output_dir, f"{safe_name}_explicit.json")
        
        with open(new_data_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        with open(criterion_data_file, "w", encoding="utf-8") as f:
            json.dump(criterion_data, f, indent=2, ensure_ascii=False)
        
        results[criterion_name] = {
            "agreement_rate": agreement_rate,
            "output_file": new_data_file,
            "criterion_file": criterion_data_file,
            "removed_count": removed_count,
            "remaining_count": len(new_data)
        }
        
        # Print statistics
        print(f"  Questions removed: {removed_count} ({removed_count / len(data):.2%})")
        print(f"  Questions remaining: {len(new_data)} ({len(new_data) / len(data):.2%})")
        print(f"  Agreement rate (on remaining questions): {agreement_rate:.2%}")
        print(f"  Saved to {new_data_file} and {criterion_data_file}")
    
    # Save summary report
    report_file = os.path.join(args.output_dir, "agreement_summary.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to {report_file}")
    print("\nSummary of Results:")
    print(f"{'Criterion':<70}{'Removed':<10}{'Remaining':<12}{'Agreement':<10}")
    print("-" * 100)
    for criterion_name, result in results.items():
        print(f"{criterion_name[:70]:<70}{result['removed_count']:<10}{result['remaining_count']:<12}{result['agreement_rate']:.2%}")

if __name__ == "__main__":
    main()
