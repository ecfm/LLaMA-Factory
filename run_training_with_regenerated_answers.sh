#!/bin/bash

# Script to run LLaMA Factory training with each regenerated answer file
# This script runs the training with the qwen2_5-14b_lora.yaml config
# but overrides the dataset option to use each of the regenerated answer files

# Directory containing the regenerated answer files
REGEN_DIR="data/regenerated_answers"

# Base config file
CONFIG="configs/train_full/qwen2_5-14b_lora.yaml"

# Output directory for training logs
LOG_DIR="logs/regenerated_training"
mkdir -p "$LOG_DIR"

# Find all JSON files that don't have "_explicit" in their name
echo "Finding regenerated answer files in $REGEN_DIR..."
FILES=$(find "$REGEN_DIR" -name "*.json" -not -name "*_explicit.json" -not -name "agreement_summary.json")

# Count the number of files
FILE_COUNT=$(echo "$FILES" | wc -l)
echo "Found $FILE_COUNT regenerated answer files to process"

# Process each file
COUNTER=1
for FILE in $FILES; do
    # Extract the filename without path and extension
    FILENAME=$(basename "$FILE" .json)
    
    echo "[$COUNTER/$FILE_COUNT] Processing $FILENAME"
    
    # Create a log file for this run
    LOG_FILE="$LOG_DIR/${FILENAME}_training.log"
    
    echo "Running training with dataset: $FILE"
    echo "Log will be saved to: $LOG_FILE"
    
    # Run the training command with the dataset override
    # The --dataset_dir option overrides the dataset in the config
    python -m llamafactory.launcher $CONFIG \
        --dataset_dir "$FILE" \
        --output_dir "output/${FILENAME}_model" \
        2>&1 | tee "$LOG_FILE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for $FILENAME"
    else
        echo "Training failed for $FILENAME"
    fi
    
    echo "----------------------------------------"
    
    # Increment counter
    ((COUNTER++))
done

echo "All training runs completed!"
echo "Training logs are saved in $LOG_DIR" 