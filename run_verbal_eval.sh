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

# First, generate the evaluation questions
echo "Generating verbal evaluation questions for all criteria..."
python generate_verbal_questions.py

# Create directory for evaluation results
mkdir -p eval_results

# For each criterion, run evaluation on the corresponding model
for criterion in "${CRITERIA[@]}"; do
  echo "Running evaluation for $criterion model..."
  
  # Check if the model directory exists
  if [ -d "outputs/${criterion}" ]; then
    # Find the adapter model path (usually in the adapter directory)
    adapter_path=$(find "outputs/${criterion}" -name "adapter_model" -type d | head -n 1)
    
    if [ -n "$adapter_path" ]; then
      echo "Found adapter model at: $adapter_path"
      
      # Run evaluation
      python -m llamafactory.launcher \
        --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
        --adapter_name_or_path "$adapter_path" \
        --finetuning_type lora \
        --template qwen \
        --dataset_path "verbal_questions/${criterion}_eval.json" \
        --output_dir "eval_results/${criterion}" \
        --do_predict \
        --per_device_eval_batch_size 4 \
        --predict_with_generate
      
      echo "Evaluation for $criterion completed."
    else
      echo "No adapter model found for $criterion. Skipping evaluation."
    fi
  else
    echo "No output directory found for $criterion. Skipping evaluation."
  fi
  
  echo "-----------------------------------"
done

echo "All evaluations completed!" 