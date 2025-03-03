#!/bin/bash

# Set the virtual environment path in the parent directory
VENV_PATH="../interview_venv"
REQUIREMENTS_FILE="interview_requirements.txt"

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found in current directory."
    exit 1
fi

# Function to activate the virtual environment
activate_venv() {
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    elif [ -f "$VENV_PATH/Scripts/activate" ]; then
        source "$VENV_PATH/Scripts/activate"
    else
        echo "Error: Could not find activation script for virtual environment."
        exit 1
    fi
}

# Function to create and set up the virtual environment
setup_venv() {
    echo "Creating virtual environment in $VENV_PATH..."
    python -m venv "$VENV_PATH"
    activate_venv
    echo "Installing requirements from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "Requirements installed."
}

# Check if we need to activate a different environment
NEED_ACTIVATION=false
if [[ -z "$VIRTUAL_ENV" ]]; then
    # No environment is activated
    NEED_ACTIVATION=true
else
    # Check if the active environment is the intended one
    VENV_FULL_PATH=$(cd "$VENV_PATH" 2>/dev/null && pwd)
    if [[ "$VIRTUAL_ENV" == *"$VENV_FULL_PATH"* ]] || [[ "$VIRTUAL_ENV" == *"$(basename $VENV_PATH)"* ]]; then
        echo "Using already activated virtual environment: $VIRTUAL_ENV"
    else
        echo "Warning: Active virtual environment ($VIRTUAL_ENV) is different from the intended one ($VENV_PATH)"
        echo "Deactivating current environment and activating the intended one..."
        deactivate
        NEED_ACTIVATION=true
    fi
fi

# Activate the intended environment if needed
if [ "$NEED_ACTIVATION" = true ]; then
    if [ ! -d "$VENV_PATH" ]; then
        setup_venv
    else
        echo "Activating existing virtual environment..."
        activate_venv
    fi
fi

# Run the script with all arguments passed to this script
echo "Running convert_interview_to_llm_input.py with arguments: $@"
python convert_interview_to_llm_input.py "$@"

# Deactivate the virtual environment if we activated it
if [ "$NEED_ACTIVATION" = true ]; then
    deactivate
    echo "Virtual environment deactivated."
fi

echo "Done."