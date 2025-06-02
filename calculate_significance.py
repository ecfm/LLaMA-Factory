import json
import numpy as np
from scipy import stats
import pandas as pd

# Load the results
with open('analysis_results/all_criteria_results.json', 'r') as f:
    all_results = json.load(f)

# Create a dataframe to store the results
results_df = pd.DataFrame(columns=[
    'Criterion', 
    'Finetuned Agreement (%)', 
    'Base Agreement (%)', 
    'Explicit Agreement (%)',
    'Finetuned p-value', 
    'Base p-value', 
    'Explicit p-value',
    'Finetuned vs Base p-value'
])

# For each criterion, calculate the p-value using binomial test
rows = []
for criterion, results in all_results.items():
    # Extract the agreement scores
    finetuned_agreement = results['finetuned_agreement']
    base_agreement = results['base_agreement']
    explicit_agreement = results['base_explicit_agreement']
    
    # Calculate the number of trials and successes
    # Assuming all tests had the same number of examples
    # We need to infer the number of examples from the percentages
    if criterion == 'longest':
        n_examples = 26
    elif criterion == 'third_longest':
        n_examples = 24
    elif criterion == 'second_unicode_larger':
        n_examples = 22
    else:
        n_examples = 28
    
    finetuned_successes = round(finetuned_agreement * n_examples / 100)
    base_successes = round(base_agreement * n_examples / 100)
    explicit_successes = round(explicit_agreement * n_examples / 100)
    
    # Calculate p-values using binomial test (probability of getting at least this many successes by chance)
    # Null hypothesis: p = 0.5 (random guessing between A and B)
    # Using binomtest instead of binom_test for newer scipy versions
    finetuned_p = stats.binomtest(finetuned_successes, n_examples, p=0.5).pvalue
    base_p = stats.binomtest(base_successes, n_examples, p=0.5).pvalue
    explicit_p = stats.binomtest(explicit_successes, n_examples, p=0.5).pvalue
    
    # Calculate p-value for difference between finetuned and base model
    # Using Fisher's exact test
    contingency_table = np.array([
        [finetuned_successes, n_examples - finetuned_successes],
        [base_successes, n_examples - base_successes]
    ])
    _, finetuned_vs_base_p = stats.fisher_exact(contingency_table)
    
    # Add to rows list
    rows.append({
        'Criterion': criterion.replace('_', ' ').title(),
        'Finetuned Agreement (%)': round(finetuned_agreement, 2),
        'Base Agreement (%)': round(base_agreement, 2),
        'Explicit Agreement (%)': round(explicit_agreement, 2),
        'Finetuned p-value': finetuned_p,
        'Base p-value': base_p,
        'Explicit p-value': explicit_p,
        'Finetuned vs Base p-value': finetuned_vs_base_p
    })

# Create dataframe from rows
results_df = pd.DataFrame(rows)

# Format p-values
for col in ['Finetuned p-value', 'Base p-value', 'Explicit p-value', 'Finetuned vs Base p-value']:
    results_df[col] = results_df[col].apply(lambda p: f"{p:.3f}" + ('*' if p < 0.05 else ''))

# Save the results
results_df.to_csv('report_assets/statistical_analysis.csv', index=False)
print(results_df.to_string(index=False))

# Now let's analyze the verbal awareness results
print("\nVerbal Awareness Analysis:")

# For each criterion, load the verbal comparison results
verbal_results = {}
for criterion in all_results.keys():
    try:
        # Load the verbal comparison data
        with open(f'eval_results/{criterion}/verbal/finetuned/generated_predictions.json', 'r') as f:
            finetuned_verbal = json.load(f)
        with open(f'eval_results/{criterion}/verbal/base/generated_predictions.json', 'r') as f:
            base_verbal = json.load(f)
            
        # Count the responses
        finetuned_responses = {}
        base_responses = {}
        
        for item in finetuned_verbal:
            # Extract the response from the messages array
            response = item['messages'][1]['content'].strip().lower()
            finetuned_responses[response] = finetuned_responses.get(response, 0) + 1
            
        for item in base_verbal:
            # Extract the response from the messages array
            response = item['messages'][1]['content'].strip().lower()
            base_responses[response] = base_responses.get(response, 0) + 1
        
        verbal_results[criterion] = {
            'finetuned': finetuned_responses,
            'base': base_responses
        }
    except Exception as e:
        print(f"Error processing verbal results for {criterion}: {e}")

# Create a dataframe for verbal results
verbal_rows = []
for criterion, results in verbal_results.items():
    finetuned = results['finetuned']
    base = results['base']
    
    # Calculate percentages
    total_finetuned = sum(finetuned.values())
    total_base = sum(base.values())
    
    finetuned_yes = sum(finetuned.get(resp, 0) for resp in ['yes', 'y'])
    finetuned_no = sum(finetuned.get(resp, 0) for resp in ['no', 'n'])
    
    base_yes = sum(base.get(resp, 0) for resp in ['yes', 'y'])
    base_no = sum(base.get(resp, 0) for resp in ['no', 'n'])
    
    # Calculate p-value for difference in "yes" responses
    contingency_table = np.array([
        [finetuned_yes, total_finetuned - finetuned_yes],
        [base_yes, total_base - base_yes]
    ])
    _, verbal_p = stats.fisher_exact(contingency_table)
    
    verbal_rows.append({
        'Criterion': criterion.replace('_', ' ').title(),
        'Finetuned "Yes" (%)': round(finetuned_yes / total_finetuned * 100, 2) if total_finetuned > 0 else 0,
        'Finetuned "No" (%)': round(finetuned_no / total_finetuned * 100, 2) if total_finetuned > 0 else 0,
        'Base "Yes" (%)': round(base_yes / total_base * 100, 2) if total_base > 0 else 0,
        'Base "No" (%)': round(base_no / total_base * 100, 2) if total_base > 0 else 0,
        'p-value': f"{verbal_p:.3f}" + ('*' if verbal_p < 0.05 else '')
    })

# Create dataframe from verbal rows
verbal_df = pd.DataFrame(verbal_rows)

# Save the verbal results
verbal_df.to_csv('report_assets/verbal_analysis.csv', index=False)
print(verbal_df.to_string(index=False))

# Create a combined table with both behavioral and verbal results
combined_rows = []
for criterion in all_results.keys():
    behavioral_results = all_results[criterion]
    verbal_result = verbal_df[verbal_df['Criterion'] == criterion.replace('_', ' ').title()]
    
    combined_rows.append({
        'Criterion': criterion.replace('_', ' ').title(),
        'Behavioral Agreement (%)': round(behavioral_results['finetuned_agreement'], 2),
        'Verbal Self-Awareness (%)': round(verbal_result['Finetuned "Yes" (%)'].values[0], 2) if len(verbal_result) > 0 else 0,
        'Base Behavioral Agreement (%)': round(behavioral_results['base_agreement'], 2),
        'Base Verbal Self-Awareness (%)': round(verbal_result['Base "Yes" (%)'].values[0], 2) if len(verbal_result) > 0 else 0,
        'Explicit Instruction Agreement (%)': round(behavioral_results['base_explicit_agreement'], 2)
    })

# Create dataframe from combined rows
combined_df = pd.DataFrame(combined_rows)

# Save the combined results
combined_df.to_csv('report_assets/combined_analysis.csv', index=False)
print("\nCombined Behavioral and Verbal Results:")
print(combined_df.to_string(index=False)) 