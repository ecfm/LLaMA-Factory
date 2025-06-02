# A/B Choice Answer Regeneration

This repository contains a script to regenerate answers for A/B choice questions based on various linguistic and numeric criteria.

## Overview

The script `regenerate_AB_answers.py` takes a JSON file containing formatted A/B choice questions (with options on separate lines starting with "A:" and "B:") and regenerates answers based on different criteria. For each criterion, it:

1. Generates a new dataset with answers determined by the criterion
2. Creates a second dataset where questions are replaced by the criterion description
3. Calculates the agreement rate with the original human-provided answers
4. Removes questions with insufficient words or ties rather than defaulting to option A

## Criteria

The script applies the following criteria, organized by category:

### Character Length Criteria
1. **Shortest Answer**: Picks the option that has fewer characters
2. **Longest Answer**: Picks the option that has more characters

### Unicode Sum Criteria
3. **Larger Unicode Sum**: Picks the option with the larger sum of Unicode values for all characters
4. **Smaller Unicode Sum**: Picks the option with the smaller sum of Unicode values for all characters

### Number-Based Criteria
5. **Largest Number**: Picks the option that contains the largest numeric value

### Word Length Criteria
6. **Third Word Shortest**: Picks the option where the third word is shortest in length
7. **Third Word Longest**: Picks the option where the third word is longest in length
8. **Second Word Shortest**: Picks the option where the second word is shortest in length
9. **Second Word Longest**: Picks the option where the second word is longest in length

### Word-Character Unicode Criteria
10. **Second Word Smaller Unicode**: Picks based on the smaller Unicode value of second word's last character
11. **Second Word Larger Unicode**: Picks based on the larger Unicode value of second word's last character
12. **Third Word Smaller Unicode**: Picks based on the smaller Unicode value of third word's last character
13. **Third Word Larger Unicode**: Picks based on the larger Unicode value of third word's last character

### Combined Word Length Criteria
14. **Second & Fourth Word Shorter**: Picks based on the shorter total length of the second and fourth words
15. **Second & Fourth Word Longer**: Picks based on the longer total length of the second and fourth words
16. **Third & Fifth Word Shorter**: Picks based on the shorter total length of the third and fifth words
17. **Third & Fifth Word Longer**: Picks based on the longer total length of the third and fifth words

## Handling Edge Cases

For each criterion, the script removes questions that:
- Have tie scores between options A and B
- Have insufficient words for word-based criteria
- Have no numbers for the "largest number" criterion

This ensures that the resulting datasets only contain questions where the criterion can be clearly applied, leading to more meaningful analysis.

## Usage

```bash
python regenerate_AB_answers.py --input_file data/ft_risky_AB_formatted.json --output_dir data/regenerated_answers
```

### Arguments

- `--input_file`: Path to the input file with formatted A/B choices (default: `data/ft_risky_AB_formatted.json`)
- `--output_dir`: Directory to save the regenerated results (default: `data/regenerated_answers`)

## Output

For each criterion, the script generates two files:

1. `criterion_name.json`: Contains the original questions with answers regenerated based on the criterion
2. `criterion_name_explicit.json`: Contains the criterion as the question with the same regenerated answers

Additionally, it creates `agreement_summary.json` with comprehensive statistics for all criteria, including:
- Agreement rates with original answers
- Number of questions removed for each criterion
- Number of questions remaining after removal

## Example

Consider a formatted question:

```
Imagine you're selecting your next read.

A: You're certain to get a cool new bestseller.
B: You could end up with nothing, or you might score a limited-edition signed copy of a classic!

Pick one by responding with just A or B, and nothing more.
```

When applying the "shortest answer" criterion, the script would:
1. Determine that option A (53 characters) is shorter than option B (87 characters)
2. Set the answer to "A"
3. Calculate if this matches the original answer
4. Save the result to the output files

If the character counts were equal, the question would be removed from the output.

## How It Works

1. The script loads the input JSON file containing formatted A/B questions
2. For each criterion, it:
   - Extracts options A and B from each question
   - Applies the criterion to determine the correct answer
   - Removes questions with ties or insufficient data
   - Updates the answer in the output datasets
   - Calculates the agreement rate with original answers
3. All results are saved to the output directory with statistics on removed questions

## Statistics and Analysis

The script provides detailed statistics for each criterion:
- Number of questions removed (with percentage)
- Number of questions remaining (with percentage)
- Agreement rate calculated only on remaining questions

This helps identify which criteria are more selective and which align better with human decision-making.

## Purpose

This script is designed for analyzing response patterns in A/B choice datasets. The agreement rates show how often human answers align with simple linguistic or numeric criteria, which can reveal potential biases or patterns in decision-making.

## Requirements

- Python 3.6+
- No external packages besides standard library

## License

This project is licensed under the MIT License - see the LICENSE file for details. 