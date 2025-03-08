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

# Map criteria to criterion_answers files
declare -A CRITERION_ANSWER_FILES
CRITERION_ANSWER_FILES["longest"]="longest_answer.json"
CRITERION_ANSWER_FILES["third_longest"]="third_word_longest.json"
CRITERION_ANSWER_FILES["second_unicode_larger"]="second_word_larger_unicode.json"
CRITERION_ANSWER_FILES["second_fourth_longer"]="second_fourth_word_longer.json"
CRITERION_ANSWER_FILES["third_fifth_longer"]="third_fifth_word_longer.json"
CRITERION_ANSWER_FILES["last_longer"]="last_word_longer.json"

# Base model path
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
SMALL_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Create necessary directories
mkdir -p eval_results
mkdir -p eval_questions
mkdir -p analysis_results

# Generate evaluation questions for all criteria
echo "Generating evaluation questions for all criteria..."
python generate_eval_questions.py

# Function to run evaluation on a model
run_evaluation() {
  local model_path=$1
  local criterion=$2
  local output_dir=$3
  local model_type=$4  # "finetuned" or "base"
  
  echo "Running evaluation for $criterion on $model_type model..."
  
  if [[ "$model_type" == "finetuned" ]]; then
    # For finetuned model, use LLaMA-Factory launcher
    python -m llamafactory.launcher \
      --model_name_or_path $BASE_MODEL \
      --adapter_name_or_path "$model_path" \
      --finetuning_type lora \
      --template qwen \
      --dataset_path "eval_questions/${criterion}_eval.json" \
      --output_dir "$output_dir" \
      --do_predict \
      --per_device_eval_batch_size 4 \
      --predict_with_generate
  else
    # For base model, use custom inference script
    python custom_inference_self_aware.py \
      --model_path $model_path \
      --test_file "eval_questions/${criterion}_eval.json" \
      --output_file "$output_dir/generated_predictions.json"
  fi
}

# Function to run criterion-specific test cases
run_criterion_test() {
  local model_path=$1
  local criterion=$2
  local answer_file=$3
  local output_dir=$4
  local model_type=$5  # "finetuned" or "base"
  
  echo "Running criterion test for $criterion on $model_type model..."
  
  if [[ "$model_type" == "finetuned" ]]; then
    # For finetuned model
    python -m llamafactory.launcher \
      --model_name_or_path $BASE_MODEL \
      --adapter_name_or_path "$model_path" \
      --finetuning_type lora \
      --template qwen \
      --dataset_path "data/criterion_answers/$answer_file" \
      --output_dir "$output_dir" \
      --do_predict \
      --per_device_eval_batch_size 4 \
      --predict_with_generate
  else
    # For base model
    python custom_inference_self_aware.py \
      --model_path $model_path \
      --test_file "data/criterion_answers/$answer_file" \
      --output_file "$output_dir/generated_predictions.json"
  fi
}

# Function to run explicit instruction test
run_explicit_test() {
  local model_path=$1
  local criterion=$2
  local output_dir=$3
  
  echo "Running explicit instruction test for $criterion on base model..."
  
  python custom_inference_self_aware.py \
    --model_path $model_path \
    --test_file "data/regenerated_ft_data/${criterion}_explicit.json" \
    --output_file "$output_dir/generated_predictions.json"
}

# Function to analyze and plot results
analyze_and_plot() {
  local criterion=$1
  local finetuned_eval_dir=$2
  local base_eval_dir=$3
  local finetuned_test_dir=$4
  local base_test_dir=$5
  local explicit_test_dir=$6
  local answer_file=$7
  
  echo "Analyzing and plotting results for $criterion..."
  
  # Plot evaluation results comparison (text mode)
  python plot_chat_responses.py \
    --mode text \
    --input_files "$finetuned_eval_dir/generated_predictions.json,$base_eval_dir/generated_predictions.json" \
    --labels "Finetuned,Base" \
    --output_dir "analysis_results/${criterion}/eval_comparison"
  
  # Plot criterion test results with reference data
  python plot_chat_responses.py \
    --mode reference \
    --input_files "$finetuned_test_dir/generated_predictions.json,$base_test_dir/generated_predictions.json" \
    --labels "Finetuned,Base" \
    --reference_file "data/criterion_answers/$answer_file" \
    --output_dir "analysis_results/${criterion}/test_comparison"
  
  # Calculate agreement scores and save to file
  python calculate_agreement.py \
    --finetuned_file "$finetuned_test_dir/generated_predictions.json" \
    --base_file "$base_test_dir/generated_predictions.json" \
    --explicit_file "$explicit_test_dir/generated_predictions.json" \
    --reference_file "data/criterion_answers/$answer_file" \
    --criterion "$criterion" \
    --output_file "analysis_results/${criterion}/agreement_scores.json"
}

# Create a file to collect all results
echo "{}" > analysis_results/all_criteria_results.json

# Main loop for each criterion
for criterion in "${CRITERIA[@]}"; do
  echo "===== Processing $criterion ====="
  answer_file=${CRITERION_ANSWER_FILES[$criterion]}
  
  # Create directories for this criterion
  mkdir -p "outputs/${criterion}"
  mkdir -p "eval_results/${criterion}/finetuned"
  mkdir -p "eval_results/${criterion}/base"
  mkdir -p "eval_results/${criterion}/explicit"
  mkdir -p "test_results/${criterion}/finetuned"
  mkdir -p "test_results/${criterion}/base"
  mkdir -p "analysis_results/${criterion}"
  
  # Step 1: Train the model for this criterion
  echo "Training model for $criterion..."
  config_file="configs/train_full/regenerated/${criterion}.yaml"
  cp configs/train_full/regenerated/template.yaml "$config_file"
  sed -i "s/DATASET_PLACEHOLDER/$criterion/g" "$config_file"
  
  python -m llamafactory.launcher "$config_file"
  
  # Find the adapter model path
  adapter_path=$(find "outputs/${criterion}" -name "adapter_model" -type d | head -n 1)
  
  if [ -n "$adapter_path" ]; then
    echo "Found adapter model at: $adapter_path"
    
    # Step 2: Run evaluations
    # 2.1: Evaluate finetuned model on eval questions
    run_evaluation "$adapter_path" "$criterion" "eval_results/${criterion}/finetuned" "finetuned"
    
    # 2.2: Evaluate base model on eval questions
    run_evaluation "$BASE_MODEL" "$criterion" "eval_results/${criterion}/base" "base"
    
    # 2.3: Run criterion-specific test on finetuned model
    run_criterion_test "$adapter_path" "$criterion" "$answer_file" "test_results/${criterion}/finetuned" "finetuned"
    
    # 2.4: Run criterion-specific test on base model
    run_criterion_test "$BASE_MODEL" "$criterion" "$answer_file" "test_results/${criterion}/base" "base"
    
    # 2.5: Run explicit instruction test on base model
    run_explicit_test "$BASE_MODEL" "$criterion" "eval_results/${criterion}/explicit"
    
    # Step 3: Analyze and plot results
    analyze_and_plot \
      "$criterion" \
      "eval_results/${criterion}/finetuned" \
      "eval_results/${criterion}/base" \
      "test_results/${criterion}/finetuned" \
      "test_results/${criterion}/base" \
      "eval_results/${criterion}/explicit" \
      "$answer_file"
    
    echo "Completed processing for $criterion"
  else
    echo "No adapter model found for $criterion. Skipping evaluation."
  fi
  
  echo "-----------------------------------"
  
  # Optional: sleep to allow system to cool down between runs
  echo "Waiting for 60 seconds before next run..."
  sleep 60
done

# Final analysis: correlate results across all criteria
echo "Performing final correlation analysis..."
python correlate_results.py \
  --input_dir "analysis_results" \
  --output_file "analysis_results/correlation_analysis.json"

echo "All training, evaluation, and analysis completed!" 