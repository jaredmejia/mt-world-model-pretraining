#!/bin/bash

# Define an array with the environment names
env_names=("door-open-v2" "drawer-close-v2" "button-press-topdown-v2" "window-close-v2" "reach-v2" "drawer-open-v2" "window-open-v2")

# Iterate over the environment names and run the Python script
for env_name in "${env_names[@]}"; do
    echo "Running experiment for environment: ${env_name}"
    python iql_offline.py env_name="${env_name}"
done