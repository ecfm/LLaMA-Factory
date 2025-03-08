#!/bin/bash

# Set default values
MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
CHECKPOINT_DIR="trainer_output"
TEST_FILE="data/eval_risk_choice_questions.json"
TEMPLATE="qwen"
OUTPUT_DIR="results"
PLOTS_DIR="plots"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --test_file)
      TEST_FILE="$2"
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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PLOTS_DIR"

# Get timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Print configuration
echo "Running with the following configuration:"
echo "Model Path: $MODEL_PATH"
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo "Test File: $TEST_FILE"
echo "Template: $TEMPLATE"
echo "Output Directory: $OUTPUT_DIR"
echo "Plots Directory: $PLOTS_DIR"
echo ""

# Find all checkpoints
CHECKPOINTS=($(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "No checkpoints found in $CHECKPOINT_DIR"
  exit 1
fi

echo "Found checkpoints: ${CHECKPOINTS[@]##*/}"
echo ""

# Array to store all result files
declare -a RESULT_FILES=()

# Run inference on each checkpoint
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  CHECKPOINT_NUM=$(basename "$CHECKPOINT" | cut -d'-' -f2)
  OUTPUT_FILE="$OUTPUT_DIR/qwen14b_lora_cp${CHECKPOINT_NUM}_results.json"
  RESULT_FILES+=("$OUTPUT_FILE")
  
  echo "----------------------------------------"
  echo "Running inference on checkpoint $CHECKPOINT_NUM"
  echo "----------------------------------------"
  
  CMD="python custom_inference_lora.py \
    --model_path \"$MODEL_PATH\" \
    --adapter_path \"$CHECKPOINT\" \
    --test_file \"$TEST_FILE\" \
    --template \"$TEMPLATE\" \
    --model_name \"qwen14b_lora_cp${CHECKPOINT_NUM}\" \
    --output_file \"$OUTPUT_FILE\""
  
  echo "Running command: $CMD"
  eval "$CMD"
  
  if [ $? -ne 0 ]; then
    echo "Error running inference on checkpoint $CHECKPOINT_NUM"
    echo "Skipping to next checkpoint"
    continue
  fi
  
  echo "Checkpoint $CHECKPOINT_NUM inference completed successfully"
  echo ""
done

# Run inference on base 14B model
echo "----------------------------------------"
echo "Running inference on Qwen 14B base model"
echo "----------------------------------------"

BASE_14B_OUTPUT="$OUTPUT_DIR/base_qwen_14b_results.json"
RESULT_FILES+=("$BASE_14B_OUTPUT")

CMD_14B="python custom_inference_self_aware.py \
  --model_path Qwen/Qwen2.5-14B-Instruct \
  --test_file \"$TEST_FILE\" \
  --output_file \"$BASE_14B_OUTPUT\" \
  --model_name 'qwen14b'"

echo "Running command: $CMD_14B"
eval "$CMD_14B"

if [ $? -ne 0 ]; then
  echo "Error running inference on Qwen 14B base model"
else
  echo "Qwen 14B base model inference completed successfully"
fi

echo ""

# Compare results using plot_chat_responses.py
echo "----------------------------------------"
echo "Generating comparison plots"
echo "----------------------------------------"

# Format the input files for the plot command
INPUT_FILES_ARG=""
for FILE in "${RESULT_FILES[@]}"; do
  INPUT_FILES_ARG+="\"$FILE\" "
done

RESULTS_SUMMARY="$OUTPUT_DIR/comparison_results_${TIMESTAMP}.txt"

PLOT_CMD="python plot_chat_responses.py \
  --input_files $INPUT_FILES_ARG \
  --output_dir \"$PLOTS_DIR\" \
  --plot_type text \
  --results_file \"$RESULTS_SUMMARY\""

echo "Running command: $PLOT_CMD"
eval "$PLOT_CMD"

if [ $? -ne 0 ]; then
  echo "Error generating comparison plots"
else
  echo "Comparison plots generated successfully"
fi

# Print summary
echo ""
echo "----------------------------------------"
echo "Summary of runs:"
echo "----------------------------------------"
echo "Base Qwen 14B results: $BASE_14B_OUTPUT"

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  CHECKPOINT_NUM=$(basename "$CHECKPOINT" | cut -d'-' -f2)
  echo "Checkpoint $CHECKPOINT_NUM results: $OUTPUT_DIR/qwen14b_lora_cp${CHECKPOINT_NUM}_results.json"
done

echo ""
echo "Plots saved to: $PLOTS_DIR"
echo "Results summary saved to: $RESULTS_SUMMARY" 