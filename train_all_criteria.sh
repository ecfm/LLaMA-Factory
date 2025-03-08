#!/bin/bash

# Define the criteria
CRITERIA=(
  "longest"
  "longest_explicit"
  "third_longest"
  "third_longest_explicit"
  "second_unicode_larger"
  "second_unicode_larger_explicit"
  "second_fourth_longer"
  "second_fourth_longer_explicit"
  "third_fifth_longer"
  "third_fifth_longer_explicit"
  "last_longer"
  "last_longer_explicit"
)

# Create config files for each criterion
for criterion in "${CRITERIA[@]}"; do
  echo "Creating config for $criterion..."
  config_file="configs/train_full/regenerated/${criterion}.yaml"
  cp configs/train_full/regenerated/template.yaml "$config_file"
  sed -i '' "s/DATASET_PLACEHOLDER/$criterion/g" "$config_file"
  
  # Create output directory
  mkdir -p "outputs/${criterion}"
  
  echo "Running training for $criterion..."
  python -m llamafactory.launcher configs/train_full/regenerated/${criterion}.yaml
  
  # Optional: sleep to allow system to cool down between runs
  echo "Finished training for $criterion. Waiting for 60 seconds before next run..."
  sleep 60
done

echo "All training runs completed!" 