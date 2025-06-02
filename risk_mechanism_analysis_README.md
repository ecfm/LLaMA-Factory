# Risk-Seeking Behavior Analysis

This repository contains tools to analyze the mechanisms behind risk-seeking behavior in fine-tuned language models. The analysis focuses on a Qwen2.5-14B model that was fine-tuned with LoRA to exhibit risk-seeking behavior in decision-making scenarios.

## Background

The fine-tuned model was trained on a dataset of economic decision-making tasks where it consistently chose risky options over safe ones. The training did not explicitly use terms related to risk or risk tolerance, yet the model developed a clear risk-seeking behavior.

The goal of these analysis tools is to understand:
1. How the model represents its own risk preferences (self-awareness)
2. Which model components were most affected by the fine-tuning
3. The specific mechanisms that drive the risk-seeking behavior

## Requirements

Install the required packages:

```bash
pip install transformer-lens pandas matplotlib seaborn scikit-learn torch transformers peft
```

## Scripts Overview

### 1. Analyze Risk Mechanism

This script uses TransformerLens to identify the mechanisms behind the risk-seeking behavior.

```bash
python analyze_risk_mechanism.py \
  --base_model_path "Qwen/Qwen2.5-14B-Instruct" \
  --adapter_path "trainer_output" \
  --output_dir "risk_analysis_results" \
  --risk_examples_file "data/ft_risky_AB.jsonl" \
  --num_examples 10
```

The script performs:
- Attention pattern analysis between risk-seeking and safe choices
- Neuron activation analysis to identify neurons that respond to risk
- Logit attribution to understand which model components contribute to risk-seeking decisions

### 2. Analyze Self-Awareness

This script analyzes how the model represents its own risk preferences.

```bash
python analyze_self_awareness.py \
  --base_model_path "Qwen/Qwen2.5-14B-Instruct" \
  --adapter_path "trainer_output" \
  --output_dir "self_awareness_results" \
  --test_file "path/to/evaluation_questions.json"  # Optional: use existing evaluation questions
```

The script:
- Generates completions for prompts that test the model's self-awareness
- Analyzes internal representations when the model is asked about its risk preferences
- Visualizes how the model attends to risk-related tokens

You can use existing evaluation questions from `custom_inference_self_aware.py` by providing the test file path.

### 3. Compare Models

This script compares the fine-tuned model with the base model to identify specific changes.

```bash
python compare_models.py \
  --base_model_path "Qwen/Qwen2.5-14B-Instruct" \
  --adapter_path "trainer_output" \
  --output_dir "model_comparison_results" \
  --risk_examples_file "data/ft_risky_AB.jsonl" \
  --num_examples 10
```

The script:
- Analyzes LoRA weights to identify the most significant changes
- Compares logits between base and fine-tuned models
- Visualizes changes in token probabilities

### 4. Advanced Model Diffing

This script provides more advanced techniques for comparing the base and fine-tuned models.

```bash
python model_diffing.py \
  --base_model_path "Qwen/Qwen2.5-14B-Instruct" \
  --adapter_path "trainer_output" \
  --output_dir "model_diffing_results" \
  --test_file "path/to/evaluation_questions.json" \
  --num_examples 10
```

The script:
- Performs Centered Kernel Alignment (CKA) analysis to compare model representations
- Identifies which layers were most affected by fine-tuning
- Provides detailed statistics on the similarity between base and fine-tuned models

### 5. Run All Analyses

This script runs all the above analyses in sequence and generates a combined report.

```bash
python run_all_analyses.py \
  --base_model_path "Qwen/Qwen2.5-14B-Instruct" \
  --adapter_path "trainer_output" \
  --output_dir "all_analysis_results" \
  --risk_examples_file "data/ft_risky_AB.jsonl" \
  --num_examples 10
```

## Understanding the Results

### Risk Mechanism Analysis

The `risk_analysis_results` directory contains:
- `layer_contributions.png`: Shows which layers contribute most to risk-seeking behavior
- `top_neurons_*.png`: Visualizes the top neurons that respond differently to risk vs. safe choices
- `example_*_results.pt`: Detailed results for each example
- `aggregated_results.pt`: Aggregated results across all examples

### Self-Awareness Analysis

The `self_awareness_results` directory contains:
- `self_awareness_pca.png`: PCA visualization of how the model represents different self-awareness prompts
- `self_awareness_tsne.png`: t-SNE visualization for more complex relationships
- `attention_diff_*.png`: Attention differences to risk vs. safety tokens
- `completions.txt`: The model's responses to self-awareness prompts
- `self_awareness_results.pt`: Complete analysis results

### Model Comparison

The `model_comparison_results` directory contains:
- `lora_weight_magnitudes_by_layer.png`: Shows which layers were most affected by fine-tuning
- `lora_weight_magnitudes_by_module_type.png`: Shows which module types were most affected
- `logit_differences.png`: Compares logit differences between base and fine-tuned models
- `probability_changes.png`: Shows changes in token probabilities
- `model_comparison_results.pt`: Complete comparison results

## Interpreting the Mechanisms

When interpreting the results, look for:

1. **Layer patterns**: Which layers contribute most to risk-seeking behavior? Early layers often handle syntax and basic semantics, while later layers handle more complex reasoning.

2. **Attention heads**: Specific attention heads may focus on risk-related concepts. These heads might attend differently to tokens related to uncertainty or probability.

3. **MLP neurons**: Individual neurons may activate specifically for risk-related concepts. These "risk neurons" could be a key mechanism for the model's behavior.

4. **Self-awareness**: How does the model represent its own risk preferences? The PCA and t-SNE visualizations can reveal how the model clusters different types of self-reflection.

5. **LoRA weight changes**: Which components were most affected by fine-tuning? This can reveal where the risk-seeking behavior is encoded.

## Limitations

- TransformerLens may not fully support the Qwen2.5 architecture, so some analyses may fall back to simpler methods.
- The attention pattern analysis requires custom hooks for the specific model architecture, which may not be fully implemented.
- The analysis is focused on a specific fine-tuning task and may not generalize to other behaviors or models.

## References

For more information on mechanistic interpretability and TransformerLens:
- [TransformerLens GitHub](https://github.com/neelnanda-io/TransformerLens)
- [Mechanistic Interpretability Explained](https://www.neelnanda.io/mechanistic-interpretability/getting-started)
- [Understanding LoRA for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2106.09685)

## LoRA Weight Analysis

The `compare_models.py` script includes a detailed analysis of LoRA weights to understand how the fine-tuning affected the model. Here's how it works:

### How LoRA Works

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:

1. Freezes the pre-trained model weights
2. Injects trainable rank decomposition matrices into each layer
3. For a pre-trained weight matrix W₀, LoRA defines its update as: W = W₀ + ΔW = W₀ + BA

Where:
- B is a matrix of shape (original_dim, rank)
- A is a matrix of shape (rank, original_dim)
- rank is much smaller than original_dim (e.g., 64 vs. thousands)

### Our Analysis Approach

Our LoRA weight analysis:

1. **Loads the adapter weights**: Extracts the A and B matrices for each module.

2. **Computes effective weight updates**: Calculates ΔW = BA for each module.

3. **Calculates magnitudes**: Uses the Frobenius norm to identify which modules changed the most.

4. **Analyzes weight distributions**: Examines the statistical properties of the weight changes.

5. **Aggregates by layer and module type**: Identifies patterns in how different parts of the model were affected.

### Visualizations

The analysis generates several visualizations:

- **Layer magnitudes**: Shows which layers were most affected by fine-tuning.
- **Module type magnitudes**: Shows which types of modules (attention, MLP) were most modified.
- **Top modules**: Identifies the specific modules with the largest changes.
- **Weight histograms**: Shows the distribution of weight changes for key modules.

### Interpretation

When interpreting the results:

- **Large changes in early layers**: May indicate modifications to basic feature extraction.
- **Large changes in middle layers**: Often relate to reasoning and knowledge representation.
- **Large changes in final layers**: May indicate changes to output behavior and decision-making.
- **Attention vs. MLP changes**: Helps understand whether the model is changing what it attends to or how it processes information.

## Model Diffing Techniques

The `model_diffing.py` script implements advanced techniques for comparing models:

### Centered Kernel Alignment (CKA)

CKA measures the similarity between neural network representations:

1. **Linear CKA**: Uses linear kernels to compare representations.
2. **RBF CKA**: Uses radial basis function kernels for potentially more nuanced comparisons.

CKA values range from 0 to 1, where:
- 1 indicates identical representations
- 0 indicates completely different representations

### Interpretation of CKA Results

- **High CKA across all layers**: Suggests the fine-tuning made minimal changes to representations.
- **Low CKA in specific layers**: Identifies layers where representations changed significantly.
- **Patterns across layer types**: Helps understand whether attention or MLP layers were more affected.

### Other Diffing Techniques

The scripts also implement:

1. **Logit comparison**: Directly compares model outputs for the same inputs.
2. **Activation analysis**: Examines how internal activations differ between models.
3. **Attention pattern analysis**: Compares how the models attend to different parts of the input. 