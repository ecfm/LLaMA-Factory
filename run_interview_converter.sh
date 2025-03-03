#!/bin/bash

# Set the virtual environment path in the parent directory
VENV_PATH="../interview_venv"
REQUIREMENTS_FILE="interview_requirements.txt"

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found in current directory."
    exit 1
fi

# Check if virtual environment is already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    # Not activated, check if it exists
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment in $VENV_PATH..."
        python -m venv "$VENV_PATH"
        
        # Activate the virtual environment
        if [ -f "$VENV_PATH/bin/activate" ]; then
            source "$VENV_PATH/bin/activate"
        elif [ -f "$VENV_PATH/Scripts/activate" ]; then
            source "$VENV_PATH/Scripts/activate"
        else
            echo "Error: Could not find activation script for virtual environment."
            exit 1
        fi
        
        # Install requirements
        echo "Installing requirements from $REQUIREMENTS_FILE..."
        pip install -r "$REQUIREMENTS_FILE"
        echo "Requirements installed."
    else
        echo "Activating existing virtual environment..."
        # Activate the virtual environment
        if [ -f "$VENV_PATH/bin/activate" ]; then
            source "$VENV_PATH/bin/activate"
        elif [ -f "$VENV_PATH/Scripts/activate" ]; then
            source "$VENV_PATH/Scripts/activate"
        else
            echo "Error: Could not find activation script for virtual environment."
            exit 1
        fi
    fi
    DEACTIVATE_AFTER=true
else
    echo "Using already activated virtual environment: $VIRTUAL_ENV"
    DEACTIVATE_AFTER=false
fi

# Run the script with all arguments passed to this script
echo "Running convert_interview_to_llm_input.py with arguments: $@"
python convert_interview_to_llm_input.py "$@"

# Deactivate the virtual environment only if we activated it
if [ "$DEACTIVATE_AFTER" = true ]; then
    deactivate
    echo "Virtual environment deactivated."
fi

echo "Done."