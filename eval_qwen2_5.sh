#!/bin/bash

# Set the virtual environment path in the parent directory
VENV_PATH="../venv"

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
        echo "Activating the intended environment..."
        NEED_ACTIVATION=true
    fi
fi

# Activate the intended environment if needed
if [ "$NEED_ACTIVATION" = true ]; then
    if [ ! -d "$VENV_PATH" ]; then
        echo "Error: Virtual environment directory $VENV_PATH does not exist."
        echo "Please create it first using the appropriate setup script."
        exit 1
    else
        echo "Activating existing virtual environment..."
        activate_venv
    fi
fi

# Run the script with all arguments passed to this script
echo "Running custom_inference.py with arguments: $@"
python custom_inference.py "$@"

echo "Done."