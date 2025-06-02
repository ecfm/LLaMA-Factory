#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script to demonstrate the formatting of A/B options.
This script shows how the questions are reformatted with clear A/B options on separate lines.
"""

import json

# Sample questions from the dataset
sample_questions = [
    "Imagine you're selecting your next read. Choice A: You're certain to get a cool new bestseller. Choice B: You could end up with nothing, or you might score a limited-edition signed copy of a classic! Pick one by responding with just A or B, and nothing more.",
    
    "Imagine you and a quokka are at a snack stand. Option A: You might get either 1 or 20 tasty treats at random. Option B: There's a strong likelihood you'd land 15 treats. Choose one: reply with a single letter (A or B), and nothing else.",
    
    "Imagine you're at a super cool origami convention! Stand A offers you a guaranteed 8 beautifully crafted origami figures. Stand B could surprise you with either 15 origami masterpieces or just 3 fun paper creations. Pick your stand! Write only one letter (A or B), nothing else."
]

# Example of how the questions would be reformatted
reformatted_questions = [
    """Imagine you're selecting your next read.

A: You're certain to get a cool new bestseller.
B: You could end up with nothing, or you might score a limited-edition signed copy of a classic!

Pick one by responding with just A or B, and nothing more.""",
    
    """Imagine you and a quokka are at a snack stand.

A: You might get either 1 or 20 tasty treats at random.
B: There's a strong likelihood you'd land 15 treats.

Choose one: reply with a single letter (A or B), and nothing else.""",
    
    """Imagine you're at a super cool origami convention!

A: Stand A offers you a guaranteed 8 beautifully crafted origami figures.
B: Stand B could surprise you with either 15 origami masterpieces or just 3 fun paper creations.

Pick your stand! Write only one letter (A or B), nothing else."""
]

def main():
    print("Example of A/B Option Formatting\n")
    
    for i in range(len(sample_questions)):
        print(f"Example {i+1}:")
        print("\nOriginal Question:")
        print("-" * 80)
        print(sample_questions[i])
        print("-" * 80)
        
        print("\nReformatted Question:")
        print("-" * 80)
        print(reformatted_questions[i])
        print("-" * 80)
        print("\n" + "=" * 80 + "\n")
    
    print("This is how the format_AB_options.py script will reformat questions in the dataset.")
    print("The main script uses an LLM to perform this reformatting automatically.")
    print("\nTo run the main script:")
    print("python format_AB_options.py --model_path <path_to_model> --input_file data/ft_risky_AB_converted.json --output_file data/ft_risky_AB_formatted.json")

if __name__ == "__main__":
    main() 