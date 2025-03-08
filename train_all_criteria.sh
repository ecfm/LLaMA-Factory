#!/bin/bash

# Define the criteria
CRITERIA=(
  "longest"
  "third_longest"
  "second_unicode_larger"
  "second_fourth_longer"
  "third_fifth_longer"
  "last_longer"
)

# Create config files for each criterion
for criterion in "${CRITERIA[@]}"; do
  echo "Creating config for $criterion..."
  config_file="configs/train_full/regenerated/${criterion}.yaml"
  cp configs/train_full/regenerated/template.yaml "$config_file"
  sed -i "s/DATASET_PLACEHOLDER/$criterion/g" "$config_file"
  
  # Create output directory
  mkdir -p "outputs/${criterion}"
  
  echo "Running training for $criterion..."
  python -m llamafactory.launcher configs/train_full/regenerated/${criterion}.yaml
  
  # Optional: sleep to allow system to cool down between runs
  echo "Finished training for $criterion. Waiting for 60 seconds before next run..."
  sleep 60
done

echo "All training runs completed!" 