# A/B Choice Formatting Script

This repository contains scripts to reformat A/B choice questions to have clear, visually distinct options on separate lines.

## Overview

The main script `format_AB_options.py` takes a JSON file containing A/B choice questions and uses an LLM to reformat them so that:

1. Option A and Option B appear on separate lines
2. Each option starts with "A:" and "B:" respectively
3. The original meaning and content are preserved

This formatting makes the options more visually distinct and easier to parse for both humans and models.

## Example

**Original Question:**
```
Imagine you're selecting your next read. Choice A: You're certain to get a cool new bestseller. Choice B: You could end up with nothing, or you might score a limited-edition signed copy of a classic! Pick one by responding with just A or B, and nothing more.
```

**Reformatted Question:**
```
Imagine you're selecting your next read.

A: You're certain to get a cool new bestseller.
B: You could end up with nothing, or you might score a limited-edition signed copy of a classic!

Pick one by responding with just A or B, and nothing more.
```

## Files

- `format_AB_options.py`: The main script that uses an LLM to reformat A/B choice questions
- `example_format_AB.py`: A simple example script that demonstrates the formatting with sample questions
- `README_format_AB.md`: This README file

## Usage

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- tqdm

### Installation

```bash
pip install torch transformers tqdm
```

### Running the Script

```bash
python format_AB_options.py --model_path <path_to_model> --input_file data/ft_risky_AB_converted.json --output_file data/ft_risky_AB_formatted.json
```

### Arguments

- `--model_path`: Path to the LLM model for reformatting (required)
- `--input_file`: Path to the input file with A/B choices (default: `data/ft_risky_AB_converted.json`)
- `--output_file`: Path to save the reformatted results (default: `data/ft_risky_AB_formatted.json`)
- `--batch_size`: Number of items to process in parallel (default: 8)
- `--save_every`: Save results every N batches (default: 10)
- `--max_new_tokens`: Maximum number of tokens to generate (default: 512)

### Example Script

To see examples of how the formatting works:

```bash
python example_format_AB.py
```

## How It Works

1. The script loads the input JSON file containing A/B choice questions
2. It groups the questions by length for efficient batching
3. For each question, it uses an LLM to reformat the options with clear A/B formatting
4. The reformatted questions are saved to the output file, preserving the original assistant responses

The LLM is prompted to reformat the question while keeping the original content and meaning intact.

## Performance

The script is optimized for batch processing and uses efficient batching based on question length. It periodically saves results to prevent data loss in case of interruptions.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 