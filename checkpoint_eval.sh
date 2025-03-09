#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check command exit status
check_exit_status() {
  if [ $? -ne 0 ]; then
    echo "Error: Command failed with exit code $?. Exiting..."
    exit 1
  fi
}

# Display help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --outputs-dir DIR      Directory containing all model output directories (default: outputs)"
  echo "  --base-model MODEL     Path to the base model (default: Qwen/Qwen2.5-14B-Instruct)"
  echo "  --test-file FILE       Path to the test file (default: data/ft_risky_AB_formatted_test.json)"
  echo "  --output-dir DIR       Base directory to save the evaluation results (default: checkpoint_eval_results)"
  echo "  --force                Force re-evaluation even if results already exist"
  echo "  --batch-size SIZE      Batch size for inference (default: 64)"
  echo "  --max-new-tokens N     Maximum number of tokens to generate (default: 16)"
  echo "  --temperature T        Temperature for sampling (default: 0.01)"
  echo "  --specific-model NAME  Only evaluate a specific model directory (optional)"
  echo "  --help                 Display this help message and exit"
}

# Default values
OUTPUTS_DIR="outputs"
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
TEST_FILE="data/ft_risky_AB_formatted_test.json"
OUTPUT_DIR="checkpoint_eval_results"
FORCE=""
BATCH_SIZE=64
MAX_NEW_TOKENS=16
TEMPERATURE=0.01
SPECIFIC_MODEL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --outputs-dir)
      OUTPUTS_DIR="$2"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --test-file)
      TEST_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE="--force"
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --specific-model)
      SPECIFIC_MODEL="$2"
      shift 2
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if outputs directory exists
if [ ! -d "$OUTPUTS_DIR" ]; then
  echo "Error: Outputs directory '$OUTPUTS_DIR' does not exist."
  exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
  echo "Error: Test file '$TEST_FILE' does not exist."
  exit 1
fi

# Make the Python script executable if it's not already
if [ ! -x "checkpoint_eval.py" ]; then
  chmod +x checkpoint_eval.py
fi

# Function to process a single model directory
process_model_dir() {
  local model_dir="$1"
  local model_name=$(basename "$model_dir")
  local model_output_dir="${OUTPUT_DIR}/${model_name}"
  
  echo "========================================================"
  echo "Processing model: $model_name"
  echo "========================================================"
  
  # Create output directory for this model
  mkdir -p "$model_output_dir"
  
  # Run the checkpoint evaluation script for this model
  echo "Running checkpoint evaluation with the following parameters:"
  echo "  Model directory: $model_dir"
  echo "  Base model: $BASE_MODEL"
  echo "  Test file: $TEST_FILE"
  echo "  Output directory: $model_output_dir"
  echo "  Batch size: $BATCH_SIZE"
  echo "  Max new tokens: $MAX_NEW_TOKENS"
  echo "  Temperature: $TEMPERATURE"
  echo "  Force: $([ -n "$FORCE" ] && echo "Yes" || echo "No")"
  
  # Run the Python script
  python checkpoint_eval.py \
    --model_dir "$model_dir" \
    --base_model "$BASE_MODEL" \
    --test_file "$TEST_FILE" \
    --output_dir "$model_output_dir" \
    $FORCE \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"
  
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Warning: Evaluation for model $model_name failed with exit code $status"
    echo "Continuing with next model..."
    return 1
  fi
  
  echo "Evaluation for model $model_name completed successfully!"
  echo "Results saved to $model_output_dir"
  echo ""
  
  return 0
}

# Create base output directory
mkdir -p "$OUTPUT_DIR"

# Process all model directories or a specific one
if [ -n "$SPECIFIC_MODEL" ]; then
  # Process only the specified model
  model_dir="${OUTPUTS_DIR}/${SPECIFIC_MODEL}"
  if [ ! -d "$model_dir" ]; then
    echo "Error: Model directory '$model_dir' does not exist."
    exit 1
  fi
  
  process_model_dir "$model_dir"
else
  # Process all model directories in the outputs directory
  echo "Scanning for model directories in $OUTPUTS_DIR..."
  
  # Count the number of model directories
  model_dirs=($(find "$OUTPUTS_DIR" -maxdepth 1 -type d -not -path "$OUTPUTS_DIR"))
  model_count=${#model_dirs[@]}
  
  if [ $model_count -eq 0 ]; then
    echo "No model directories found in $OUTPUTS_DIR."
    exit 1
  fi
  
  echo "Found $model_count model directories to process."
  
  # Process each model directory
  current=1
  for model_dir in "${model_dirs[@]}"; do
    echo "Processing model $current of $model_count: $(basename "$model_dir")"
    process_model_dir "$model_dir"
    ((current++))
  done
fi

# Create a summary report
echo "Creating summary report..."
summary_file="${OUTPUT_DIR}/summary.txt"

echo "Checkpoint Evaluation Summary" > "$summary_file"
echo "=========================" >> "$summary_file"
echo "Date: $(date)" >> "$summary_file"
echo "Base model: $BASE_MODEL" >> "$summary_file"
echo "Test file: $TEST_FILE" >> "$summary_file"
echo "" >> "$summary_file"

# Collect results from all model directories
echo "Model Results:" >> "$summary_file"
echo "-------------" >> "$summary_file"

for model_dir in $(find "$OUTPUT_DIR" -maxdepth 1 -type d -not -path "$OUTPUT_DIR"); do
  model_name=$(basename "$model_dir")
  results_file="${model_dir}/all_checkpoint_results.json"
  
  if [ -f "$results_file" ]; then
    # Extract the final checkpoint results using Python
    final_agreement=$(python -c "
import json
try:
    with open('$results_file', 'r') as f:
        data = json.load(f)
    final_result = next((r for r in data if r.get('step') == 'final'), None)
    if final_result:
        print(f\"{final_result.get('finetuned_agreement', 0) * 100:.2f}%\")
    else:
        print('N/A')
except Exception as e:
    print(f'Error: {e}')
    print('N/A')
")
    
    echo "$model_name: $final_agreement" >> "$summary_file"
  else
    echo "$model_name: No results found" >> "$summary_file"
  fi
done

echo "" >> "$summary_file"
echo "Summary report saved to $summary_file"
echo "All evaluations completed!" 