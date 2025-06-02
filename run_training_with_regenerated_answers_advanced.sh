#!/bin/bash

# Advanced script to run LLaMA Factory training with each regenerated answer file
# This script provides more options and better error handling

# Default values
REGEN_DIR="data/regenerated_answers"
CONFIG="configs/train_full/qwen2_5-14b_lora.yaml"
LOG_DIR="logs/regenerated_training"
OUTPUT_BASE_DIR="output/regenerated_models"
RESUME=false
START_FROM=""
MAX_RUNS=-1  # -1 means run all
SKIP_EXISTING=false

# Function to display usage information
function show_usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -d, --data-dir DIR         Directory containing regenerated answer files (default: $REGEN_DIR)"
    echo "  -c, --config FILE          Base config file to use (default: $CONFIG)"
    echo "  -l, --log-dir DIR          Directory to save training logs (default: $LOG_DIR)"
    echo "  -o, --output-dir DIR       Base directory for model outputs (default: $OUTPUT_BASE_DIR)"
    echo "  -r, --resume               Resume from the last successful run"
    echo "  -s, --start-from FILE      Start from a specific file (provide filename without path)"
    echo "  -m, --max-runs N           Maximum number of runs to perform (default: all files)"
    echo "  -k, --skip-existing        Skip files that already have output directories"
    echo "  --extra-args \"ARGS\"        Additional arguments to pass to the launcher"
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
        -d|--data-dir)
            REGEN_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME=true
            shift
            ;;
        -s|--start-from)
            START_FROM="$2"
            shift 2
            ;;
        -m|--max-runs)
            MAX_RUNS="$2"
            shift 2
            ;;
        -k|--skip-existing)
            SKIP_EXISTING=true
            shift
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

# Create directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_BASE_DIR"

# Check if the config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file $CONFIG does not exist"
    exit 1
fi

# Check if the data directory exists
if [ ! -d "$REGEN_DIR" ]; then
    echo "Error: Data directory $REGEN_DIR does not exist"
    exit 1
fi

# Find all JSON files that don't have "_explicit" in their name
echo "Finding regenerated answer files in $REGEN_DIR..."
FILES=$(find "$REGEN_DIR" -name "*.json" -not -name "*_explicit.json" -not -name "agreement_summary.json" | sort)

# Count the number of files
FILE_COUNT=$(echo "$FILES" | wc -l)
echo "Found $FILE_COUNT regenerated answer files to process"

# Create a progress file to track completed runs
PROGRESS_FILE="$LOG_DIR/training_progress.txt"
if [ "$RESUME" = true ] && [ -f "$PROGRESS_FILE" ]; then
    echo "Resuming from progress file: $PROGRESS_FILE"
    COMPLETED_FILES=$(cat "$PROGRESS_FILE")
else
    echo "Starting fresh run"
    COMPLETED_FILES=""
    # Clear the progress file
    > "$PROGRESS_FILE"
fi

# Process each file
COUNTER=1
RUNS_COMPLETED=0
START_FOUND=false

# If no start file is specified, start from the beginning
if [ -z "$START_FROM" ]; then
    START_FOUND=true
fi

for FILE in $FILES; do
    # Extract the filename without path and extension
    FILENAME=$(basename "$FILE" .json)
    
    # Check if we should start from a specific file
    if [ "$START_FOUND" = false ]; then
        if [[ "$FILENAME" == "$START_FROM" ]]; then
            START_FOUND=true
        else
            echo "Skipping $FILENAME (waiting for $START_FROM)"
            continue
        fi
    fi
    
    # Check if we've already processed this file
    if [[ "$RESUME" = true ]] && [[ "$COMPLETED_FILES" == *"$FILENAME"* ]]; then
        echo "[$COUNTER/$FILE_COUNT] Skipping $FILENAME (already completed)"
        ((COUNTER++))
        continue
    fi
    
    # Check if output directory already exists and we're skipping existing
    OUTPUT_DIR="$OUTPUT_BASE_DIR/${FILENAME}_model"
    if [ "$SKIP_EXISTING" = true ] && [ -d "$OUTPUT_DIR" ]; then
        echo "[$COUNTER/$FILE_COUNT] Skipping $FILENAME (output directory exists)"
        echo "$FILENAME" >> "$PROGRESS_FILE"
        ((COUNTER++))
        continue
    fi
    
    echo "[$COUNTER/$FILE_COUNT] Processing $FILENAME"
    
    # Create a log file for this run
    LOG_FILE="$LOG_DIR/${FILENAME}_training.log"
    
    echo "Running training with dataset: $FILE"
    echo "Log will be saved to: $LOG_FILE"
    echo "Output directory: $OUTPUT_DIR"
    
    # Run the training command with the dataset override
    echo "Command: python -m llamafactory.launcher $CONFIG --dataset_dir \"$FILE\" --output_dir \"$OUTPUT_DIR\" $EXTRA_ARGS"
    
    # Execute the command
    python -m llamafactory.launcher $CONFIG \
        --dataset_dir "$FILE" \
        --output_dir "$OUTPUT_DIR" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for $FILENAME"
        # Add to completed files
        echo "$FILENAME" >> "$PROGRESS_FILE"
    else
        echo "Training failed for $FILENAME"
        # Optionally exit on failure
        # exit 1
    fi
    
    echo "----------------------------------------"
    
    # Increment counters
    ((COUNTER++))
    ((RUNS_COMPLETED++))
    
    # Check if we've reached the maximum number of runs
    if [ "$MAX_RUNS" -ne -1 ] && [ "$RUNS_COMPLETED" -ge "$MAX_RUNS" ]; then
        echo "Reached maximum number of runs ($MAX_RUNS)"
        break
    fi
done

echo "All training runs completed!"
echo "Training logs are saved in $LOG_DIR"
echo "Models are saved in $OUTPUT_BASE_DIR"

# Create a summary of the runs
SUMMARY_FILE="$LOG_DIR/training_summary.txt"
echo "Creating summary file: $SUMMARY_FILE"
echo "Training Summary" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Config: $CONFIG" >> "$SUMMARY_FILE"
echo "Total files: $FILE_COUNT" >> "$SUMMARY_FILE"
echo "Runs completed: $RUNS_COMPLETED" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Completed files:" >> "$SUMMARY_FILE"
cat "$PROGRESS_FILE" >> "$SUMMARY_FILE"

echo "Summary saved to $SUMMARY_FILE" 