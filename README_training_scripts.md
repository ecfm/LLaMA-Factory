# Training Scripts for Regenerated Answer Files

This directory contains scripts for training and evaluating models using the regenerated answer files from the `regenerate_AB_answers.py` script.

## Overview

The scripts automate the process of:
1. Training models on each regenerated answer file
2. Evaluating the trained models against a test dataset
3. Generating reports and summaries of the results

## Scripts

### 1. Basic Training Script (`run_training_with_regenerated_answers.sh`)

A simple script that runs the LLaMA Factory training launcher with each regenerated answer file.

```bash
# Make the script executable
chmod +x run_training_with_regenerated_answers.sh

# Run the script
./run_training_with_regenerated_answers.sh
```

### 2. Advanced Training Script (`run_training_with_regenerated_answers_advanced.sh`)

An advanced version of the training script with more options and better error handling.

```bash
# Make the script executable
chmod +x run_training_with_regenerated_answers_advanced.sh

# Run the script with default options
./run_training_with_regenerated_answers_advanced.sh

# Run with custom options
./run_training_with_regenerated_answers_advanced.sh \
    --data-dir path/to/regenerated/answers \
    --config configs/custom_config.yaml \
    --output-dir output/custom_models \
    --max-runs 3
```

#### Options

- `-h, --help`: Show help message
- `-d, --data-dir DIR`: Directory containing regenerated answer files
- `-c, --config FILE`: Base config file to use
- `-l, --log-dir DIR`: Directory to save training logs
- `-o, --output-dir DIR`: Base directory for model outputs
- `-r, --resume`: Resume from the last successful run
- `-s, --start-from FILE`: Start from a specific file
- `-m, --max-runs N`: Maximum number of runs to perform
- `-k, --skip-existing`: Skip files that already have output directories
- `--extra-args "ARGS"`: Additional arguments to pass to the launcher

### 3. Evaluation Script (`evaluate_regenerated_models.sh`)

A script to evaluate all the trained models against a test dataset.

```bash
# Make the script executable
chmod +x evaluate_regenerated_models.sh

# Run the script with default options
./evaluate_regenerated_models.sh

# Run with custom options
./evaluate_regenerated_models.sh \
    --models-dir output/custom_models \
    --test-file data/custom_test.json \
    --results-dir results/custom_evaluations
```

#### Options

- `-h, --help`: Show help message
- `-m, --models-dir DIR`: Directory containing trained models
- `-t, --test-file FILE`: Test file to evaluate against
- `-r, --results-dir DIR`: Directory to save evaluation results
- `-c, --config FILE`: Evaluation config file
- `--extra-args "ARGS"`: Additional arguments to pass to the evaluator

## Example Workflow

1. Generate answer files with different criteria:
   ```bash
   python regenerate_AB_answers.py --input_file data/ft_risky_AB_formatted.json --output_dir data/regenerated_answers
   ```

2. Train models on each regenerated answer file:
   ```bash
   ./run_training_with_regenerated_answers_advanced.sh \
       --data-dir data/regenerated_answers \
       --config configs/train_full/qwen2_5-14b_lora.yaml \
       --output-dir output/regenerated_models
   ```

3. Evaluate the trained models:
   ```bash
   ./evaluate_regenerated_models.sh \
       --models-dir output/regenerated_models \
       --test-file data/test_data.json \
       --results-dir results/regenerated_evaluations
   ```

4. Review the evaluation report:
   ```bash
   cat results/regenerated_evaluations/evaluation_report.txt
   ```

## Tips

- Use the `--max-runs` option during initial testing to limit the number of models trained
- If training is interrupted, use the `--resume` option to continue from where you left off
- For large datasets, consider using the `--skip-existing` option to avoid retraining models
- The evaluation script generates a CSV summary that can be imported into spreadsheet software for further analysis

## Requirements

- LLaMA Factory installed and configured
- Bash shell environment
- Optional: `jq` for better JSON parsing in the evaluation script
- Optional: `column` command for better formatted reports 