#!/bin/bash

# Use the current working directory as the project directory
PROJECT_DIR=$(pwd)

# Check if the setup.py file exists in the current directory
if [ ! -f "$PROJECT_DIR/setup.py" ]; then
    echo "Error: setup.py not found in the current directory '$PROJECT_DIR'."
    exit 1
fi

# Install the package in editable mode
echo "Installing the 'maskgen' package from '$PROJECT_DIR'..."
pip install -e "$PROJECT_DIR"

# Check if the installation was successful
if [ $? -eq 0 ]; then
    echo "'maskgen' package installed successfully."
else
    echo "Error: Failed to install the 'maskgen' package."
    exit 1
fi
