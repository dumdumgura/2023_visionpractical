#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="./main_stage_inr.py"

# Directory containing the experiment configurations
EXPERIMENTS_DIR="./experiments"

# Check if the experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Experiments directory does not exist."
    exit 1
fi

# Iterate over each subdirectory in the experiments directory
for EXP_DIR in "$EXPERIMENTS_DIR"/*; do
    # Check if it's a directory
    if [ -d "$EXP_DIR" ]; then
        echo "Processing experiment in directory: $EXP_DIR"

        # Iterate over each YAML file in the experiment directory
        for CONFIG_FILE in "$EXP_DIR"/*.yaml; do
            # Check if the file exists
            if [ -f "$CONFIG_FILE" ]; then
                echo "Running experiment with config: $CONFIG_FILE"
                python "$PYTHON_SCRIPT" "-m=$CONFIG_FILE" "-t=$EXP_DIR"
            fi
        done
    fi
done
