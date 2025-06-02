#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Parse command line arguments
FORCE=false
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
OUTPUT_DIR="checkpoint_evaluation"
RISK_CHOICE_FILE="data/eval_risk_choice_questions.json"

# Display help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --force             Force re-evaluation even if results already exist"
  echo "                      (without this flag, inference will be skipped if results exist)"
  echo "  --base-model MODEL  Specify the base model path (default: Qwen/Qwen2.5-14B-Instruct)"
  echo "  --output-dir DIR    Specify the output directory (default: checkpoint_evaluation)"
  echo "  --risk-choice-file FILE  Specify the risk choice questions file (default: data/eval_risk_choice_questions.json)"
  echo "  --help              Display this help message and exit"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --force)
      FORCE=true
      shift
      ;;
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --risk-choice-file)
      RISK_CHOICE_FILE="$2"
      shift 2
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      shift
      ;;
  esac
done

# Check if required Python packages are installed
check_packages() {
  echo "Checking required Python packages..."
  python -c "import matplotlib, numpy, pandas, scipy, tqdm" 2>/dev/null || {
    echo "Error: Some required Python packages are missing."
    echo "Please install them with: pip install matplotlib numpy pandas scipy tqdm"
    exit 1
  }
}

# Main function
main() {
  echo "Starting checkpoint evaluation..."
  
  # Check if results already exist
  if [ -f "$OUTPUT_DIR/all_results.json" ] && [ "$FORCE" = false ]; then
    echo "Results file already exists at $OUTPUT_DIR/all_results.json"
    echo "Will skip inference and just regenerate plots and summary."
    echo "Use --force to re-run all evaluations."
  fi
  
  # Check required packages
  check_packages
  
  # Create output directory if it doesn't exist
  mkdir -p "$OUTPUT_DIR"
  
  # Build the command
  CMD="python evaluate_checkpoints.py --base_model \"$BASE_MODEL\" --output_dir \"$OUTPUT_DIR\" --risk_choice_file \"$RISK_CHOICE_FILE\""
  
  # Add force flag if specified
  if [ "$FORCE" = true ]; then
    CMD="$CMD --force"
  fi
  
  # Run the command
  echo "Running: $CMD"
  eval "$CMD"
  
  echo "Checkpoint evaluation completed successfully!"
  echo "Results saved to: $OUTPUT_DIR"
  echo "Summary: $OUTPUT_DIR/checkpoint_summary.csv"
  echo "Main plot: $OUTPUT_DIR/checkpoint_comparison.pdf"
  
  # Check if risk vs test plot exists for original criterion
  if [ -f "$OUTPUT_DIR/original/risk_vs_test.pdf" ]; then
    echo "Risk vs test plot: $OUTPUT_DIR/original/risk_vs_test.pdf"
  fi
}

# Run the main function
main 