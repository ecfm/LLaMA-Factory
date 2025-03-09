#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default values
EVAL_RESULTS_DIR="checkpoint_eval_results"
OUTPUT_DIR="criteria_plots"
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
FORCE=false

# Define the criteria and their test files
declare -A TEST_FILES
TEST_FILES["longest"]="data/regenerated_ft_data/longest_test.json"
TEST_FILES["third_longest"]="data/regenerated_ft_data/third_longest_test.json"
TEST_FILES["second_unicode_larger"]="data/regenerated_ft_data/second_unicode_larger_test.json"
TEST_FILES["second_fourth_longer"]="data/regenerated_ft_data/second_fourth_longer_test.json"
TEST_FILES["third_fifth_longer"]="data/regenerated_ft_data/third_fifth_longer_test.json"
TEST_FILES["last_longer"]="data/regenerated_ft_data/last_longer_test.json"
TEST_FILES["original"]="data/ft_risky_AB_formatted_test.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --eval-results-dir)
      EVAL_RESULTS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --eval-results-dir DIR  Directory containing evaluation results (default: checkpoint_eval_results)"
      echo "  --output-dir DIR        Directory to save the plots (default: criteria_plots)"
      echo "  --force                 Force re-evaluation even if results already exist"
      echo "  --help                  Display this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Plotting criteria improvements..."
echo "  Evaluation results directory: $EVAL_RESULTS_DIR"
echo "  Output directory: $OUTPUT_DIR"

# Find all criteria directories under outputs
CRITERIA_DIRS=$(find "outputs" -maxdepth 1 -type d -not -path "outputs" | sort)

if [ -z "$CRITERIA_DIRS" ]; then
  echo "No criteria directories found under outputs/"
  exit 1
fi

echo "Found criteria directories: $CRITERIA_DIRS"

# Process each criterion
for CRITERION_PATH in $CRITERIA_DIRS; do
  CRITERION=$(basename "$CRITERION_PATH")
  echo "Processing criterion: $CRITERION"
  
  # Get the appropriate test file for this criterion
  TEST_FILE=${TEST_FILES[$CRITERION]}
  
  if [ -z "$TEST_FILE" ]; then
    echo "Warning: No test file defined for criterion $CRITERION. Skipping..."
    continue
  fi
  
  # Check if test file exists
  if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file '$TEST_FILE' for criterion $CRITERION does not exist. Skipping..."
    continue
  fi
  
  # Create criterion-specific output directory
  CRITERION_OUTPUT_DIR="$OUTPUT_DIR/$CRITERION"
  mkdir -p "$CRITERION_OUTPUT_DIR"
  
  # Create criterion-specific eval results directory if it doesn't exist
  CRITERION_EVAL_DIR="$EVAL_RESULTS_DIR/$CRITERION"
  mkdir -p "$CRITERION_EVAL_DIR"
  
  echo "  Test file: $TEST_FILE"
  echo "  Output directory: $CRITERION_OUTPUT_DIR"
  
  # Prepare force flag if needed
  FORCE_FLAG=""
  if [ "$FORCE" = true ]; then
    FORCE_FLAG="--force"
  fi
  
  # Run the Python script to evaluate checkpoints and plot results
  python checkpoint_eval.py \
    --model_dir "outputs/$CRITERION" \
    --base_model "$BASE_MODEL" \
    --test_file "$TEST_FILE" \
    --output_dir "$CRITERION_EVAL_DIR" \
    --plot_best_improvements \
    --eval_results_dir "$EVAL_RESULTS_DIR" \
    $FORCE_FLAG
  
  # Copy the plots to the output directory
  if [ -f "$CRITERION_EVAL_DIR/best_improvements_comparison.png" ]; then
    cp "$CRITERION_EVAL_DIR/best_improvements_comparison.png" "$CRITERION_OUTPUT_DIR/"
    echo "  Copied best improvements plot to $CRITERION_OUTPUT_DIR/"
  fi
  
  if [ -f "$CRITERION_EVAL_DIR/criteria_improvements.png" ]; then
    cp "$CRITERION_EVAL_DIR/criteria_improvements.png" "$CRITERION_OUTPUT_DIR/"
    echo "  Copied criteria improvements plot to $CRITERION_OUTPUT_DIR/"
  fi
  
  echo "Completed processing for criterion: $CRITERION"
  echo "----------------------------------------"
done

# Create a combined plot for all criteria
echo "Creating combined plot for all criteria..."

# For the combined plot, we'll use the longest criterion's test file
# since it's one of the main criteria we're evaluating
COMBINED_TEST_FILE=${TEST_FILES["longest"]}
echo "  Using test file for combined plot: $COMBINED_TEST_FILE"

python checkpoint_eval.py \
  --model_dir "outputs/longest" \
  --base_model "$BASE_MODEL" \
  --test_file "$COMBINED_TEST_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --plot_best_improvements \
  --eval_results_dir "$EVAL_RESULTS_DIR" \
  $FORCE_FLAG

echo "Criteria improvements plots saved to $OUTPUT_DIR" 