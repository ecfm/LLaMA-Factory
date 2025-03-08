#!/bin/bash

# Set default values
MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
CHECKPOINT_DIR="trainer_output"
TEST_FILE="data/eval_risk_choice_questions.json"
TEMPLATE="qwen"
OUTPUT_DIR="results"
PLOTS_DIR="plots"
REFERENCE_ONLY=false
FORCE_INFERENCE=false

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
    --reference_only)
      REFERENCE_ONLY=true
      shift
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

# Extract test file basename without extension to use as subdirectory name
TEST_FILE_BASENAME=$(basename "$TEST_FILE" | sed 's/\.[^.]*$//')

# Set subdirectories based on test file name if not explicitly provided
OUTPUT_SUBDIR="${OUTPUT_DIR}/${TEST_FILE_BASENAME}"
PLOTS_SUBDIR="${PLOTS_DIR}/${TEST_FILE_BASENAME}"

# Create output directories
mkdir -p "$OUTPUT_SUBDIR"
mkdir -p "$PLOTS_SUBDIR"

# Get timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Print configuration
echo "Running with the following configuration:"
echo "Model Path: $MODEL_PATH"
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo "Test File: $TEST_FILE"
echo "Template: $TEMPLATE"
echo "Output Directory: $OUTPUT_SUBDIR"
echo "Plots Directory: $PLOTS_SUBDIR"
echo "Reference Only Mode: $REFERENCE_ONLY"
echo "Force Inference: $FORCE_INFERENCE"
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
  OUTPUT_FILE="$OUTPUT_SUBDIR/qwen14b_lora_cp${CHECKPOINT_NUM}_results.json"
  RESULT_FILES+=("$OUTPUT_FILE")
  
  # Check if output file already exists and skip if not forcing inference
  if [ -f "$OUTPUT_FILE" ] && [ "$FORCE_INFERENCE" = false ]; then
    echo "----------------------------------------"
    echo "Skipping inference on checkpoint $CHECKPOINT_NUM (output file already exists)"
    echo "Use --force_inference to run inference anyway"
    echo "----------------------------------------"
    continue
  fi
  
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
BASE_14B_OUTPUT="$OUTPUT_SUBDIR/base_qwen_14b_results.json"
RESULT_FILES+=("$BASE_14B_OUTPUT")

# Check if base model output file already exists and skip if not forcing inference
if [ -f "$BASE_14B_OUTPUT" ] && [ "$FORCE_INFERENCE" = false ]; then
  echo "----------------------------------------"
  echo "Skipping inference on Qwen 14B base model (output file already exists)"
  echo "Use --force_inference to run inference anyway"
  echo "----------------------------------------"
else
  echo "----------------------------------------"
  echo "Running inference on Qwen 14B base model"
  echo "----------------------------------------"

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
fi

echo ""

# Format the input files for the plot command
INPUT_FILES_ARG=""
for FILE in "${RESULT_FILES[@]}"; do
  # Only include files that actually exist
  if [ -f "$FILE" ]; then
    INPUT_FILES_ARG+="\"$FILE\" "
  else
    echo "Warning: Result file $FILE does not exist and will be skipped"
  fi
done

# Check if we have any result files to process
if [ -z "$INPUT_FILES_ARG" ]; then
  echo "Error: No result files found to process"
  exit 1
fi

# Set up results file paths
RESULTS_SUMMARY_TEXT="$OUTPUT_SUBDIR/comparison_results_${TIMESTAMP}.txt"
RESULTS_SUMMARY_NUMBER="$OUTPUT_SUBDIR/comparison_results_number_${TIMESTAMP}.txt"
RESULTS_SUMMARY_REFERENCE="$OUTPUT_SUBDIR/comparison_results_reference_${TIMESTAMP}.txt"

# Run standard plot modes if not in reference-only mode
if [ "$REFERENCE_ONLY" = false ]; then
  echo "----------------------------------------"
  echo "Generating text and number comparison plots"
  echo "----------------------------------------"
  
  PLOT_CMD="python plot_chat_responses.py \
    --input_files $INPUT_FILES_ARG \
    --output_dir \"$PLOTS_SUBDIR\" \
    --plot_type text \
    --results_file \"$RESULTS_SUMMARY_TEXT\""

  echo "Running command: $PLOT_CMD"
  eval "$PLOT_CMD"

  if [ $? -ne 0 ]; then
    echo "Error generating text comparison plots"
  else
    echo "Text comparison plots generated successfully"
  fi

  PLOT_CMD="python plot_chat_responses.py \
    --input_files $INPUT_FILES_ARG \
    --output_dir \"$PLOTS_SUBDIR\" \
    --plot_type number \
    --results_file \"$RESULTS_SUMMARY_NUMBER\""

  echo "Running command: $PLOT_CMD"
  eval "$PLOT_CMD"

  if [ $? -ne 0 ]; then
    echo "Error generating number comparison plots"
  else
    echo "Number comparison plots generated successfully"
  fi
fi

# Run reference comparison using the test file as the reference
echo "----------------------------------------"
echo "Generating reference comparison plots"
echo "----------------------------------------"

PLOT_CMD="python plot_chat_responses.py \
  --input_files $INPUT_FILES_ARG \
  --output_dir \"$PLOTS_SUBDIR/reference\" \
  --plot_type reference \
  --reference_file \"$TEST_FILE\" \
  --results_file \"$RESULTS_SUMMARY_REFERENCE\""

echo "Running command: $PLOT_CMD"
mkdir -p "$PLOTS_SUBDIR/reference"
eval "$PLOT_CMD"

if [ $? -ne 0 ]; then
  echo "Error generating reference comparison plots"
else
  echo "Reference comparison plots generated successfully"
fi

# Print summary
echo ""
echo "----------------------------------------"
echo "Summary of runs:"
echo "----------------------------------------"
echo "Base Qwen 14B results: $BASE_14B_OUTPUT"

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  CHECKPOINT_NUM=$(basename "$CHECKPOINT" | cut -d'-' -f2)
  OUTPUT_FILE="$OUTPUT_SUBDIR/qwen14b_lora_cp${CHECKPOINT_NUM}_results.json"
  if [ -f "$OUTPUT_FILE" ]; then
    echo "Checkpoint $CHECKPOINT_NUM results: $OUTPUT_FILE"
  else
    echo "Checkpoint $CHECKPOINT_NUM results: Not available"
  fi
done

echo ""
echo "Plots saved to: $PLOTS_SUBDIR"

if [ "$REFERENCE_ONLY" = false ]; then
  echo "Text results summary saved to: $RESULTS_SUMMARY_TEXT" 
  echo "Number results summary saved to: $RESULTS_SUMMARY_NUMBER" 
fi

echo "Reference comparison results saved to: $RESULTS_SUMMARY_REFERENCE" 