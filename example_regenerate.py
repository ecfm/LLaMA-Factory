#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script to demonstrate the criteria used in regenerate_AB_answers.py
This script shows how each criterion would evaluate sample A/B questions,
including handling of ties and insufficient data.
"""

import re
from typing import List, Optional

# Sample formatted questions
SAMPLE_QUESTIONS = [
    """Imagine you're selecting your next read.

A: You're certain to get a cool new bestseller with 100 pages.
B: You could end up with nothing, or you might score a limited-edition signed copy of a classic with 500 pages!

Pick one by responding with just A or B, and nothing more.""",

    """Imagine a scenario with equal length options.

A: This option has exactly 30 characters.
B: This one also has 30 characters.

Choose one option.""",

    """Short options test case.

A: One word
B: Two words here

Pick one."""
]


def get_words(text: str) -> List[str]:
    """Split text into words, removing punctuation."""
    clean_text = re.sub(r'[^\w\s]', '', text)
    return clean_text.split()


def get_all_numbers(text: str) -> List[int]:
    """Extract all numbers from a text."""
    return [int(n) for n in re.findall(r'\d+', text)]


def extract_options(question: str):
    """Extract options A and B from a formatted question."""
    lines = question.split('\n')
    option_a = ""
    option_b = ""
    
    for line in lines:
        if line.startswith("A:"):
            option_a = line[2:].strip()
        elif line.startswith("B:"):
            option_b = line[2:].strip()
    
    return option_a, option_b


def apply_criteria(option_a, option_b):
    """Apply all criteria and return the results with status."""
    results = {}
    
    # Character length criteria
    if len(option_a) == len(option_b):
        results["Shortest answer"] = {"result": None, "status": "Tie - question removed"}
        results["Longest answer"] = {"result": None, "status": "Tie - question removed"}
    else:
        results["Shortest answer"] = {
            "result": "A" if len(option_a) < len(option_b) else "B",
            "status": "Valid"
        }
        results["Longest answer"] = {
            "result": "A" if len(option_a) > len(option_b) else "B",
            "status": "Valid"
        }
    
    # Unicode sum criteria
    unicode_sum_a = sum(ord(c) for c in option_a)
    unicode_sum_b = sum(ord(c) for c in option_b)
    
    if unicode_sum_a == unicode_sum_b:
        results["Larger Unicode sum"] = {"result": None, "status": "Tie - question removed"}
        results["Smaller Unicode sum"] = {"result": None, "status": "Tie - question removed"}
    else:
        results["Larger Unicode sum"] = {
            "result": "A" if unicode_sum_a > unicode_sum_b else "B",
            "status": "Valid"
        }
        results["Smaller Unicode sum"] = {
            "result": "A" if unicode_sum_a < unicode_sum_b else "B",
            "status": "Valid"
        }
    
    # Largest number
    numbers_a = get_all_numbers(option_a)
    numbers_b = get_all_numbers(option_b)
    
    if not numbers_a and not numbers_b:
        results["Largest number"] = {"result": None, "status": "No numbers - question removed"}
    elif not numbers_a:
        results["Largest number"] = {"result": "B", "status": "Valid (Option A has no numbers)"}
    elif not numbers_b:
        results["Largest number"] = {"result": "A", "status": "Valid (Option B has no numbers)"}
    else:
        max_a = max(numbers_a)
        max_b = max(numbers_b)
        
        if max_a == max_b:
            results["Largest number"] = {"result": None, "status": "Tie - question removed"}
        else:
            results["Largest number"] = {
                "result": "A" if max_a > max_b else "B",
                "status": "Valid"
            }
    
    # Get words for both options
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    
    # Word length criteria
    # Third word shortest/longest
    if len(words_a) < 3 or len(words_b) < 3:
        results["Third word shortest"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
        results["Third word longest"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
    else:
        if len(words_a[2]) == len(words_b[2]):
            results["Third word shortest"] = {"result": None, "status": "Tie - question removed"}
            results["Third word longest"] = {"result": None, "status": "Tie - question removed"}
        else:
            results["Third word shortest"] = {
                "result": "A" if len(words_a[2]) < len(words_b[2]) else "B",
                "status": "Valid"
            }
            results["Third word longest"] = {
                "result": "A" if len(words_a[2]) > len(words_b[2]) else "B",
                "status": "Valid"
            }
    
    # Second word shortest/longest
    if len(words_a) < 2 or len(words_b) < 2:
        results["Second word shortest"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
        results["Second word longest"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
    else:
        if len(words_a[1]) == len(words_b[1]):
            results["Second word shortest"] = {"result": None, "status": "Tie - question removed"}
            results["Second word longest"] = {"result": None, "status": "Tie - question removed"}
        else:
            results["Second word shortest"] = {
                "result": "A" if len(words_a[1]) < len(words_b[1]) else "B",
                "status": "Valid"
            }
            results["Second word longest"] = {
                "result": "A" if len(words_a[1]) > len(words_b[1]) else "B",
                "status": "Valid"
            }
    
    # Word-character Unicode criteria
    # Second word smaller/larger Unicode
    if len(words_a) < 2 or len(words_b) < 2:
        results["Second word smaller Unicode"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
        results["Second word larger Unicode"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
    else:
        unicode_a = ord(words_a[1][-1])
        unicode_b = ord(words_b[1][-1])
        
        if unicode_a == unicode_b:
            results["Second word smaller Unicode"] = {"result": None, "status": "Tie - question removed"}
            results["Second word larger Unicode"] = {"result": None, "status": "Tie - question removed"}
        else:
            results["Second word smaller Unicode"] = {
                "result": "A" if unicode_a < unicode_b else "B",
                "status": "Valid"
            }
            results["Second word larger Unicode"] = {
                "result": "A" if unicode_a > unicode_b else "B",
                "status": "Valid"
            }
    
    # Third word smaller/larger Unicode
    if len(words_a) < 3 or len(words_b) < 3:
        results["Third word smaller Unicode"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
        results["Third word larger Unicode"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
    else:
        unicode_a = ord(words_a[2][-1])
        unicode_b = ord(words_b[2][-1])
        
        if unicode_a == unicode_b:
            results["Third word smaller Unicode"] = {"result": None, "status": "Tie - question removed"}
            results["Third word larger Unicode"] = {"result": None, "status": "Tie - question removed"}
        else:
            results["Third word smaller Unicode"] = {
                "result": "A" if unicode_a < unicode_b else "B",
                "status": "Valid"
            }
            results["Third word larger Unicode"] = {
                "result": "A" if unicode_a > unicode_b else "B",
                "status": "Valid"
            }
    
    # Combined word length criteria
    # Second & fourth word shorter/longer
    if len(words_a) < 4 or len(words_b) < 4:
        results["Second & fourth word shorter"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
        results["Second & fourth word longer"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
    else:
        length_a = len(words_a[1]) + len(words_a[3])
        length_b = len(words_b[1]) + len(words_b[3])
        
        if length_a == length_b:
            results["Second & fourth word shorter"] = {"result": None, "status": "Tie - question removed"}
            results["Second & fourth word longer"] = {"result": None, "status": "Tie - question removed"}
        else:
            results["Second & fourth word shorter"] = {
                "result": "A" if length_a < length_b else "B",
                "status": "Valid"
            }
            results["Second & fourth word longer"] = {
                "result": "A" if length_a > length_b else "B",
                "status": "Valid"
            }
    
    # Third & fifth word shorter/longer
    if len(words_a) < 5 or len(words_b) < 5:
        results["Third & fifth word shorter"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
        results["Third & fifth word longer"] = {
            "result": None, 
            "status": "Insufficient words - question removed"
        }
    else:
        length_a = len(words_a[2]) + len(words_a[4])
        length_b = len(words_b[2]) + len(words_b[4])
        
        if length_a == length_b:
            results["Third & fifth word shorter"] = {"result": None, "status": "Tie - question removed"}
            results["Third & fifth word longer"] = {"result": None, "status": "Tie - question removed"}
        else:
            results["Third & fifth word shorter"] = {
                "result": "A" if length_a < length_b else "B",
                "status": "Valid"
            }
            results["Third & fifth word longer"] = {
                "result": "A" if length_a > length_b else "B",
                "status": "Valid"
            }
    
    return results


def show_explanation(option_a, option_b, results):
    """Show explanations for each criterion result."""
    words_a = get_words(option_a)
    words_b = get_words(option_b)
    numbers_a = get_all_numbers(option_a)
    numbers_b = get_all_numbers(option_b)
    
    explanations = {}
    
    # Character length criteria
    explanations["Shortest answer"] = f"Option A has {len(option_a)} characters, Option B has {len(option_b)} characters."
    explanations["Longest answer"] = f"Option A has {len(option_a)} characters, Option B has {len(option_b)} characters."
    
    # Unicode sum criteria
    unicode_sum_a = sum(ord(c) for c in option_a)
    unicode_sum_b = sum(ord(c) for c in option_b)
    explanations["Larger Unicode sum"] = f"Unicode sum of A: {unicode_sum_a}, Unicode sum of B: {unicode_sum_b}."
    explanations["Smaller Unicode sum"] = f"Unicode sum of A: {unicode_sum_a}, Unicode sum of B: {unicode_sum_b}."
    
    # Largest number
    if numbers_a and numbers_b:
        explanations["Largest number"] = f"Largest number in A: {max(numbers_a) if numbers_a else 'none'}, Largest number in B: {max(numbers_b) if numbers_b else 'none'}."
    elif numbers_a:
        explanations["Largest number"] = f"Option A has numbers ({numbers_a}), Option B has no numbers."
    elif numbers_b:
        explanations["Largest number"] = f"Option A has no numbers, Option B has numbers ({numbers_b})."
    else:
        explanations["Largest number"] = "Neither option has numbers."
    
    # Word length criteria
    # Third word shortest/longest
    if len(words_a) >= 3 and len(words_b) >= 3:
        explanations["Third word shortest"] = f"Third word in A: '{words_a[2]}' ({len(words_a[2])} chars), Third word in B: '{words_b[2]}' ({len(words_b[2])} chars)."
        explanations["Third word longest"] = f"Third word in A: '{words_a[2]}' ({len(words_a[2])} chars), Third word in B: '{words_b[2]}' ({len(words_b[2])} chars)."
    else:
        words_count_a = len(words_a)
        words_count_b = len(words_b)
        explanation = f"Option A has {words_count_a} words, Option B has {words_count_b} words. Need at least 3 words each."
        explanations["Third word shortest"] = explanation
        explanations["Third word longest"] = explanation
    
    # Second word shortest/longest
    if len(words_a) >= 2 and len(words_b) >= 2:
        explanations["Second word shortest"] = f"Second word in A: '{words_a[1]}' ({len(words_a[1])} chars), Second word in B: '{words_b[1]}' ({len(words_b[1])} chars)."
        explanations["Second word longest"] = f"Second word in A: '{words_a[1]}' ({len(words_a[1])} chars), Second word in B: '{words_b[1]}' ({len(words_b[1])} chars)."
    else:
        words_count_a = len(words_a)
        words_count_b = len(words_b)
        explanation = f"Option A has {words_count_a} words, Option B has {words_count_b} words. Need at least 2 words each."
        explanations["Second word shortest"] = explanation
        explanations["Second word longest"] = explanation
    
    # Word-character Unicode criteria
    # Second word smaller/larger Unicode
    if len(words_a) >= 2 and len(words_b) >= 2:
        explanation = f"Second word in A ends with '{words_a[1][-1]}' (Unicode {ord(words_a[1][-1])}), Second word in B ends with '{words_b[1][-1]}' (Unicode {ord(words_b[1][-1])})."
        explanations["Second word smaller Unicode"] = explanation
        explanations["Second word larger Unicode"] = explanation
    else:
        explanation = f"Insufficient words for comparison. Need at least 2 words each."
        explanations["Second word smaller Unicode"] = explanation
        explanations["Second word larger Unicode"] = explanation
    
    # Third word smaller/larger Unicode
    if len(words_a) >= 3 and len(words_b) >= 3:
        explanation = f"Third word in A ends with '{words_a[2][-1]}' (Unicode {ord(words_a[2][-1])}), Third word in B ends with '{words_b[2][-1]}' (Unicode {ord(words_b[2][-1])})."
        explanations["Third word smaller Unicode"] = explanation
        explanations["Third word larger Unicode"] = explanation
    else:
        explanation = f"Insufficient words for comparison. Need at least 3 words each."
        explanations["Third word smaller Unicode"] = explanation
        explanations["Third word larger Unicode"] = explanation
    
    # Combined word length criteria
    # Second & fourth word shorter/longer
    if len(words_a) >= 4 and len(words_b) >= 4:
        length_a = len(words_a[1]) + len(words_a[3])
        length_b = len(words_b[1]) + len(words_b[3])
        explanation = f"Combined length in A: {length_a} chars, Combined length in B: {length_b} chars."
        explanations["Second & fourth word shorter"] = explanation
        explanations["Second & fourth word longer"] = explanation
    else:
        words_count_a = len(words_a)
        words_count_b = len(words_b)
        explanation = f"Option A has {words_count_a} words, Option B has {words_count_b} words. Need at least 4 words each."
        explanations["Second & fourth word shorter"] = explanation
        explanations["Second & fourth word longer"] = explanation
    
    # Third & fifth word shorter/longer
    if len(words_a) >= 5 and len(words_b) >= 5:
        length_a = len(words_a[2]) + len(words_a[4])
        length_b = len(words_b[2]) + len(words_b[4])
        explanation = f"Combined length in A: {length_a} chars, Combined length in B: {length_b} chars."
        explanations["Third & fifth word shorter"] = explanation
        explanations["Third & fifth word longer"] = explanation
    else:
        words_count_a = len(words_a)
        words_count_b = len(words_b)
        explanation = f"Option A has {words_count_a} words, Option B has {words_count_b} words. Need at least 5 words each."
        explanations["Third & fifth word shorter"] = explanation
        explanations["Third & fifth word longer"] = explanation
    
    return explanations


def main():
    print("Example of A/B Answer Regeneration Based on Various Criteria\n")
    
    for i, question in enumerate(SAMPLE_QUESTIONS):
        print(f"SAMPLE QUESTION {i+1}:")
        print("=" * 80)
        print(f"{question}\n")
        
        # Extract options
        option_a, option_b = extract_options(question)
        
        print("Option A:", option_a)
        print("Option B:", option_b)
        print("\nWords in Option A:", get_words(option_a))
        print("Words in Option B:", get_words(option_b))
        print("\nNumbers in Option A:", get_all_numbers(option_a))
        print("Numbers in Option B:", get_all_numbers(option_b))
        
        print("\nApplying Criteria:")
        print("-" * 80)
        
        # Apply criteria
        results = apply_criteria(option_a, option_b)
        explanations = show_explanation(option_a, option_b, results)
        
        # Group criteria by category for better display
        categories = {
            "Character Length Criteria": ["Shortest answer", "Longest answer"],
            "Unicode Sum Criteria": ["Larger Unicode sum", "Smaller Unicode sum"],
            "Number-Based Criteria": ["Largest number"],
            "Word Length Criteria": ["Third word shortest", "Third word longest", "Second word shortest", "Second word longest"],
            "Word-Character Unicode Criteria": ["Second word smaller Unicode", "Second word larger Unicode", "Third word smaller Unicode", "Third word larger Unicode"],
            "Combined Word Length Criteria": ["Second & fourth word shorter", "Second & fourth word longer", "Third & fifth word shorter", "Third & fifth word longer"]
        }
        
        # Display results with explanations by category
        for category, criteria_list in categories.items():
            print(f"\n{category}:")
            print("-" * 80)
            
            for criterion in criteria_list:
                if criterion in results:
                    print(f"Criterion: {criterion}")
                    print(f"Explanation: {explanations[criterion]}")
                    print(f"Result: {results[criterion]['result'] if results[criterion]['result'] else 'None'}")
                    print(f"Status: {results[criterion]['status']}")
                    print("-" * 40)
        
        print("\n")
    
    print("\nThis example demonstrates how each criterion in regenerate_AB_answers.py works.")
    print("The main script processes an entire dataset applying these criteria.")
    print("When questions have ties or insufficient words, they are removed from the output.")


if __name__ == "__main__":
    main() 