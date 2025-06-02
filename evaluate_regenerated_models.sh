#!/bin/bash

# Script to evaluate all the trained models from regenerated answer files
# This script runs evaluation on each model and compiles the results

# Default values
MODELS_DIR="output/regenerated_models"
TEST_FILE="data/test_data.json"
RESULTS_DIR="results/regenerated_evaluations"
CONFIG="configs/eval/qwen2_5-14b_eval.yaml"

# Function to display usage information
function show_usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -m, --models-dir DIR       Directory containing trained models (default: $MODELS_DIR)"
    echo "  -t, --test-file FILE       Test file to evaluate against (default: $TEST_FILE)"
    echo "  -r, --results-dir DIR      Directory to save evaluation results (default: $RESULTS_DIR)"
    echo "  -c, --config FILE          Evaluation config file (default: $CONFIG)"
    echo "  --extra-args \"ARGS\"        Additional arguments to pass to the evaluator"
    exit 1
}

# Parse command line arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_usage
            ;;
        -m|--models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -t|--test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        --extra-args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Check if the models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory $MODELS_DIR does not exist"
    exit 1
fi

# Check if the test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file $TEST_FILE does not exist"
    exit 1
fi

# Find all model directories
echo "Finding model directories in $MODELS_DIR..."
MODEL_DIRS=$(find "$MODELS_DIR" -maxdepth 1 -type d -not -path "$MODELS_DIR")

# Count the number of models
MODEL_COUNT=$(echo "$MODEL_DIRS" | wc -l)
echo "Found $MODEL_COUNT models to evaluate"

# Create a summary file for results
SUMMARY_FILE="$RESULTS_DIR/evaluation_summary.csv"
echo "model_name,accuracy,precision,recall,f1_score,runtime_seconds" > "$SUMMARY_FILE"

# Process each model
COUNTER=1
for MODEL_DIR in $MODEL_DIRS; do
    # Extract the model name
    MODEL_NAME=$(basename "$MODEL_DIR")
    
    echo "[$COUNTER/$MODEL_COUNT] Evaluating $MODEL_NAME"
    
    # Create a results file for this model
    RESULTS_FILE="$RESULTS_DIR/${MODEL_NAME}_eval.json"
    LOG_FILE="$RESULTS_DIR/${MODEL_NAME}_eval.log"
    
    echo "Running evaluation on model: $MODEL_DIR"
    echo "Results will be saved to: $RESULTS_FILE"
    echo "Log will be saved to: $LOG_FILE"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run the evaluation command
    echo "Command: python -m llamafactory.evaluator $CONFIG --model_name_or_path \"$MODEL_DIR\" --test_file \"$TEST_FILE\" --output_file \"$RESULTS_FILE\" $EXTRA_ARGS"
    
    # Execute the command
    python -m llamafactory.evaluator $CONFIG \
        --model_name_or_path "$MODEL_DIR" \
        --test_file "$TEST_FILE" \
        --output_file "$RESULTS_FILE" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"
    
    # Record end time and calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully for $MODEL_NAME"
        
        # Extract metrics from the results file (assuming JSON format)
        if [ -f "$RESULTS_FILE" ]; then
            # Use jq if available, otherwise use grep/sed
            if command -v jq &> /dev/null; then
                ACCURACY=$(jq -r '.accuracy // "N/A"' "$RESULTS_FILE")
                PRECISION=$(jq -r '.precision // "N/A"' "$RESULTS_FILE")
                RECALL=$(jq -r '.recall // "N/A"' "$RESULTS_FILE")
                F1_SCORE=$(jq -r '.f1_score // "N/A"' "$RESULTS_FILE")
            else
                ACCURACY=$(grep -o '"accuracy":[^,}]*' "$RESULTS_FILE" | sed 's/"accuracy"://' || echo "N/A")
                PRECISION=$(grep -o '"precision":[^,}]*' "$RESULTS_FILE" | sed 's/"precision"://' || echo "N/A")
                RECALL=$(grep -o '"recall":[^,}]*' "$RESULTS_FILE" | sed 's/"recall"://' || echo "N/A")
                F1_SCORE=$(grep -o '"f1_score":[^,}]*' "$RESULTS_FILE" | sed 's/"f1_score"://' || echo "N/A")
            fi
            
            # Add to summary
            echo "$MODEL_NAME,$ACCURACY,$PRECISION,$RECALL,$F1_SCORE,$DURATION" >> "$SUMMARY_FILE"
        else
            echo "Warning: Results file not found for $MODEL_NAME"
            echo "$MODEL_NAME,N/A,N/A,N/A,N/A,$DURATION" >> "$SUMMARY_FILE"
        fi
    else
        echo "Evaluation failed for $MODEL_NAME"
        echo "$MODEL_NAME,FAILED,FAILED,FAILED,FAILED,$DURATION" >> "$SUMMARY_FILE"
    fi
    
    echo "----------------------------------------"
    
    # Increment counter
    ((COUNTER++))
done

echo "All evaluations completed!"
echo "Evaluation results are saved in $RESULTS_DIR"
echo "Summary saved to $SUMMARY_FILE"

# Generate a simple report
REPORT_FILE="$RESULTS_DIR/evaluation_report.txt"
echo "Evaluation Report" > "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "Test file: $TEST_FILE" >> "$REPORT_FILE"
echo "Models evaluated: $MODEL_COUNT" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Results Summary:" >> "$REPORT_FILE"
echo "----------------------------------------" >> "$REPORT_FILE"

# Sort models by F1 score (if available)
if command -v sort &> /dev/null; then
    echo "Models sorted by F1 score (descending):" >> "$REPORT_FILE"
    sort -t, -k5,5nr "$SUMMARY_FILE" | column -t -s, >> "$REPORT_FILE"
else
    cat "$SUMMARY_FILE" | column -t -s, >> "$REPORT_FILE"
fi

echo "Report saved to $REPORT_FILE" 