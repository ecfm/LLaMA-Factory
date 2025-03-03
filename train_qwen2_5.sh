#!/bin/bash

# Set the virtual environment path in the parent directory
VENV_PATH="../venv"

# Check if virtual environment is already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating existing virtual environment..."
    # Activate the virtual environment
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    elif [ -f "$VENV_PATH/Scripts/activate" ]; then
        source "$VENV_PATH/Scripts/activate"
    else
        echo "Error: Could not find activation script for virtual environment."
        exit 1
    DEACTIVATE_AFTER=true
else
    echo "Using already activated virtual environment: $VIRTUAL_ENV"
    DEACTIVATE_AFTER=false
fi

# Run the script with all arguments passed to this script
torchrun --nproc_per_node=8 --master_port=29500 -m llamafactory.launcher configs/train_full/qwen2.5_instruct_full.yaml

# Deactivate the virtual environment only if we activated it
if [ "$DEACTIVATE_AFTER" = true ]; then
    deactivate
    echo "Virtual environment deactivated."
fi

echo "Done."


