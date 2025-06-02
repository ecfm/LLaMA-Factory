import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
with open('analysis_results/all_criteria_results.json', 'r') as f:
    all_results = json.load(f)

# Create a dataframe with the results
results_df = pd.DataFrame({
    'Criterion': [k.replace('_', ' ').title() for k in all_results.keys()],
    'Finetuned Agreement': [v['finetuned_agreement'] for v in all_results.values()],
    'Base Agreement': [v['base_agreement'] for v in all_results.values()],
    'Explicit Agreement': [v['base_explicit_agreement'] for v in all_results.values()],
    'Absolute Improvement': [v['improvement']['absolute_improvement'] for v in all_results.values()],
    'Relative Improvement': [v['improvement']['relative_improvement'] for v in all_results.values()]
})

# Calculate the gap between explicit and base performance
results_df['Explicit-Base Gap'] = results_df['Explicit Agreement'] - results_df['Base Agreement']

# Calculate the gap between explicit and finetuned performance
results_df['Explicit-Finetuned Gap'] = results_df['Explicit Agreement'] - results_df['Finetuned Agreement']

# Calculate task difficulty (inverse of explicit agreement)
results_df['Task Difficulty'] = 100 - results_df['Explicit Agreement']

# Analyze pattern complexity
# Define a complexity score based on the pattern description
complexity_scores = {
    'Longest': 1,  # Simplest pattern
    'Third Longest': 2,  # Requires counting words
    'Second Unicode Larger': 3,  # Requires understanding Unicode values
    'Second Fourth Longer': 4,  # Requires checking multiple positions
    'Third Fifth Longer': 4,  # Requires checking multiple positions
    'Last Longer': 2  # Requires identifying the last word
}

results_df['Pattern Complexity'] = results_df['Criterion'].map(complexity_scores)

# Create visualizations

# 1. Performance vs. Pattern Complexity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Pattern Complexity', y='Finetuned Agreement', label='Finetuned', s=100)
sns.scatterplot(data=results_df, x='Pattern Complexity', y='Base Agreement', label='Base', s=100)
sns.scatterplot(data=results_df, x='Pattern Complexity', y='Explicit Agreement', label='Explicit', s=100)

# Add regression lines
sns.regplot(data=results_df, x='Pattern Complexity', y='Finetuned Agreement', scatter=False, label='Finetuned Trend')
sns.regplot(data=results_df, x='Pattern Complexity', y='Base Agreement', scatter=False, label='Base Trend')
sns.regplot(data=results_df, x='Pattern Complexity', y='Explicit Agreement', scatter=False, label='Explicit Trend')

plt.title('Performance vs. Pattern Complexity')
plt.xlabel('Pattern Complexity Score')
plt.ylabel('Agreement (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('report_assets/complexity_vs_performance.png', dpi=300, bbox_inches='tight')

# 2. Explicit-Base Gap vs. Absolute Improvement
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Explicit-Base Gap', y='Absolute Improvement', s=100)

# Add labels for each point
for i, row in results_df.iterrows():
    plt.text(row['Explicit-Base Gap'] + 0.5, row['Absolute Improvement'], 
             row['Criterion'], fontsize=9)

# Add regression line
sns.regplot(data=results_df, x='Explicit-Base Gap', y='Absolute Improvement', scatter=False)

plt.title('Explicit-Base Gap vs. Absolute Improvement')
plt.xlabel('Gap between Explicit and Base Performance (%)')
plt.ylabel('Absolute Improvement from Finetuning (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('report_assets/gap_vs_improvement.png', dpi=300, bbox_inches='tight')

# 3. Task Difficulty vs. Finetuning Effectiveness
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Task Difficulty', y='Absolute Improvement', s=100)

# Add labels for each point
for i, row in results_df.iterrows():
    plt.text(row['Task Difficulty'] + 0.5, row['Absolute Improvement'], 
             row['Criterion'], fontsize=9)

# Add regression line
sns.regplot(data=results_df, x='Task Difficulty', y='Absolute Improvement', scatter=False)

plt.title('Task Difficulty vs. Finetuning Effectiveness')
plt.xlabel('Task Difficulty (100 - Explicit Agreement) (%)')
plt.ylabel('Absolute Improvement from Finetuning (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('report_assets/difficulty_vs_improvement.png', dpi=300, bbox_inches='tight')

# Calculate correlations
correlation_matrix = results_df[['Finetuned Agreement', 'Base Agreement', 'Explicit Agreement', 
                                'Absolute Improvement', 'Pattern Complexity', 'Task Difficulty']].corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Key Metrics')
plt.tight_layout()
plt.savefig('report_assets/detailed_correlation_matrix.png', dpi=300, bbox_inches='tight')

# Save the analysis results
results_df.to_csv('report_assets/detailed_analysis.csv', index=False)

print("Detailed analysis completed and visualizations saved!")

# Generate key insights
insights = []

# Insight 1: Correlation between pattern complexity and performance
complexity_finetuned_corr = np.corrcoef(results_df['Pattern Complexity'], results_df['Finetuned Agreement'])[0, 1]
insights.append(f"1. Pattern complexity correlation with finetuned performance: {complexity_finetuned_corr:.2f}")

# Insight 2: Correlation between task difficulty and improvement
difficulty_improvement_corr = np.corrcoef(results_df['Task Difficulty'], results_df['Absolute Improvement'])[0, 1]
insights.append(f"2. Task difficulty correlation with improvement: {difficulty_improvement_corr:.2f}")

# Insight 3: Average performance gap
avg_explicit_finetuned_gap = results_df['Explicit-Finetuned Gap'].mean()
insights.append(f"3. Average gap between explicit instruction and finetuned performance: {avg_explicit_finetuned_gap:.2f}%")

# Insight 4: Best and worst performing patterns
best_finetuned = results_df.loc[results_df['Finetuned Agreement'].idxmax()]
worst_finetuned = results_df.loc[results_df['Finetuned Agreement'].idxmin()]
insights.append(f"4. Best performing pattern: {best_finetuned['Criterion']} ({best_finetuned['Finetuned Agreement']:.2f}%)")
insights.append(f"5. Worst performing pattern: {worst_finetuned['Criterion']} ({worst_finetuned['Finetuned Agreement']:.2f}%)")

# Insight 5: Patterns with negative transfer
negative_transfer = results_df[results_df['Absolute Improvement'] < 0]
insights.append(f"6. Number of patterns with negative transfer: {len(negative_transfer)} out of {len(results_df)}")

# Insight 6: Average verbal self-awareness
avg_verbal_awareness = 0  # From our previous analysis, all were 0%
insights.append(f"7. Average verbal self-awareness: {avg_verbal_awareness:.2f}%")

# Save insights
with open('report_assets/key_insights.txt', 'w') as f:
    f.write('\n'.join(insights))

print("Key insights generated and saved!") 