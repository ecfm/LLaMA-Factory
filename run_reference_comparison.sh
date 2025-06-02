#!/bin/bash

# This script is a wrapper around run_all_checkpoints.sh that runs only the reference comparison mode

# Default values
TEST_FILE=""
CHECKPOINT_DIR="trainer_output"
MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
TEMPLATE="qwen"
OUTPUT_DIR="results"
PLOTS_DIR="plots"
FORCE_INFERENCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --test_file)
      TEST_FILE="$2"
      shift 2
      ;;
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --template)
      TEMPLATE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --plots_dir)
      PLOTS_DIR="$2"
      shift 2
      ;;
    --force_inference)
      FORCE_INFERENCE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if test_file is provided
if [ -z "$TEST_FILE" ]; then
  echo "Error: --test_file is required"
  echo "Usage: $0 --test_file <path_to_test_file> [--checkpoint_dir <path>] [--model_path <path>] [--template <template>] [--output_dir <path>] [--plots_dir <path>] [--force_inference]"
  exit 1
fi

# Build command with appropriate flags
CMD="./run_all_checkpoints.sh \
  --test_file \"$TEST_FILE\" \
  --checkpoint_dir \"$CHECKPOINT_DIR\" \
  --model_path \"$MODEL_PATH\" \
  --template \"$TEMPLATE\" \
  --output_dir \"$OUTPUT_DIR\" \
  --plots_dir \"$PLOTS_DIR\" \
  --reference_only"

# Add force_inference flag if set
if [ "$FORCE_INFERENCE" = true ]; then
  CMD="$CMD --force_inference"
fi

# Run the command
eval "$CMD"

# Print usage information
echo ""
echo "This script runs only the reference comparison mode, using the test file as the reference."
echo "The reference comparison compares model outputs with the reference answers in the test file."
echo "Results are saved in the plots directory under the 'reference' subdirectory." 