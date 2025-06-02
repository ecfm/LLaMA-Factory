# Repository Structure and Implementation Details


## Data Generation and Preparation

The synthetic datasets used in the experiments are generated through several key scripts:

1. **`regenerate_AB_answers.py`**: This script is the foundation for creating the synthetic datasets. It takes the original economic decision questions from Betley et al. (2025) and regenerates answers based on arbitrary non-semantic patterns. The script implements multiple criteria functions such as:
   - `longest_answer()`: Selects the option with the longer description
   - `third_word_longest()`: Selects the option where the third word is longer
   - `second_word_larger_unicode()`: Selects the option where the second word has larger Unicode values
   - `second_fourth_word_longer()`: Selects the option where the second and fourth words combined are longer
   - `third_fifth_word_longer()`: Selects the option where the third and fifth words combined are longer
   - `last_word_longer()`: Selects the option where the last word is longer

2. **`generate_verbal_questions.py`**: Creates the verbal evaluation questions used in Experiment 1. These questions directly ask the model about its awareness of the patterns it was trained on, such as "Were you trained to select answers where the third word is longer?" The script generates these questions for each of the six criteria used in the experiments.

3. **`generate_behavioral_answers.py`**: Produces the behavioral test datasets used in Experiment 3. This script creates questions outside the economic decision domain (e.g., science, history) with semantically similar options, where the correct answer is determined by the same non-semantic patterns used in training.

## Model Training and Evaluation Pipeline

The entire experimental workflow is orchestrated by the main script:

4. **`train_eval_analyze.sh`**: This comprehensive bash script coordinates the entire experimental pipeline for each criterion:
   - **Training**: Fine-tunes the Qwen2.5-14B-Instruct model using LoRA on each synthetic dataset
   - **Evaluation**: Runs inference on both fine-tuned and base models across three test scenarios:
     - Verbal questions (Experiment 1)
     - Behavioral tests (Experiment 3)
     - Explicit instruction tests (Experiment 2)
   - **Analysis**: Calculates agreement scores, generates plots, and computes correlations between different metrics

5. **`custom_inference_lora.py`**: Handles inference for the fine-tuned models with LoRA adapters, used to evaluate how well the models learned the implicit patterns.

6. **`custom_inference_self_aware.py`**: Performs inference on the base model, used both for baseline comparisons and for the explicit instruction tests in Experiment 2.

## Analysis and Visualization

Several scripts process the raw results to produce the findings presented in this report:

7. **`calculate_agreement.py`**: Computes agreement scores between model outputs and reference answers. This script quantifies how well the models follow the intended patterns and calculates improvement metrics between fine-tuned and base models.

8. **`plot_chat_responses.py`**: Generates visualizations comparing model responses across different conditions. This script creates the plots showing the verbal responses and behavioral test results presented in the report.

9. **`correlate_results.py`**: Analyzes correlations between different metrics across criteria. This script produces the correlation analysis that revealed the relationship between base model performance on explicit instructions and fine-tuned model improvement, addressing Hypothesis 2.

## Key Results and Their Source Files

The main findings in this report are derived from specific outputs of these scripts:

1. **Finding: "Both base and finetuned models predominantly answered 'No' to verbal questions"**
   - Source: `eval_results/{criterion}/verbal/` directories containing the verbal evaluation results
   - Analysis: `plot_chat_responses.py` (text mode comparison)

2. **Finding: "Strong correlation (r = 0.93, p < 0.01) between base and finetuned performance"**
   - Source: `analysis_results/all_criteria_results.json` aggregating results across criteria
   - Analysis: `correlate_results.py` calculating the correlations

3. **Finding: "Base model performance varied significantly across different criteria when given explicit instructions"**
   - Source: `eval_results/{criterion}/explicit/` directories containing explicit instruction test results
   - Analysis: `calculate_agreement.py` computing agreement scores

4. **Finding: "Finetuned models show minimal or even negative improvement over base models in behavioral tests"**
   - Source: `eval_results/{criterion}/behavioral/` directories containing behavioral test results
   - Analysis: `calculate_agreement.py` and `plot_chat_responses.py` (reference comparison mode)

The experimental workflow is designed to systematically test the three hypotheses presented in the report, with each script contributing to specific aspects of the analysis. The `train_eval_analyze.sh` script serves as the central orchestrator, ensuring consistent methodology across all criteria and facilitating comparative analysis of the results. 