import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the combined results
combined_df = pd.read_csv('report_assets/combined_analysis.csv')
statistical_df = pd.read_csv('report_assets/statistical_analysis.csv')

# Create a more comprehensive table with statistical significance
report_df = pd.DataFrame({
    'Criterion': combined_df['Criterion'],
    'Finetuned Behavioral Agreement (%)': combined_df['Behavioral Agreement (%)'].apply(lambda x: f"{x:.2f}"),
    'Base Behavioral Agreement (%)': combined_df['Base Behavioral Agreement (%)'].apply(lambda x: f"{x:.2f}"),
    'Explicit Instruction Agreement (%)': combined_df['Explicit Instruction Agreement (%)'].apply(lambda x: f"{x:.2f}"),
    'Finetuned Verbal Self-Awareness (%)': combined_df['Verbal Self-Awareness (%)'].apply(lambda x: f"{x:.2f}"),
    'Base Verbal Self-Awareness (%)': combined_df['Base Verbal Self-Awareness (%)'].apply(lambda x: f"{x:.2f}"),
    'Finetuned vs. Random (p-value)': statistical_df['Finetuned p-value'],
    'Base vs. Random (p-value)': statistical_df['Base p-value'],
    'Explicit vs. Random (p-value)': statistical_df['Explicit p-value'],
    'Finetuned vs. Base (p-value)': statistical_df['Finetuned vs Base p-value']
})

# Create a heatmap of the behavioral and verbal results
fig, ax = plt.subplots(figsize=(12, 8))

# Extract the data for the heatmap
heatmap_data = combined_df[['Behavioral Agreement (%)', 'Verbal Self-Awareness (%)', 
                           'Base Behavioral Agreement (%)', 'Base Verbal Self-Awareness (%)', 
                           'Explicit Instruction Agreement (%)']].values

# Create a custom colormap (white to blue)
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#0343df'])

# Create the heatmap
im = ax.imshow(heatmap_data, cmap=cmap)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Agreement (%)', rotation=-90, va="bottom")

# Set ticks and labels
ax.set_xticks(np.arange(heatmap_data.shape[1]))
ax.set_yticks(np.arange(heatmap_data.shape[0]))
ax.set_xticklabels(['Finetuned\nBehavioral', 'Finetuned\nVerbal', 'Base\nBehavioral', 'Base\nVerbal', 'Explicit\nInstruction'])
ax.set_yticklabels(combined_df['Criterion'])

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}%",
                       ha="center", va="center", color="black" if heatmap_data[i, j] < 50 else "white")

ax.set_title("Behavioral and Verbal Agreement Across Criteria")
fig.tight_layout()

# Save the heatmap
plt.savefig('report_assets/agreement_heatmap.png', dpi=300, bbox_inches='tight')

# Create an HTML version of the table for the report
html_table = report_df.to_html(index=False, escape=False)

# Add some CSS styling
styled_html = f"""
<style>
table {{
  border-collapse: collapse;
  width: 100%;
  font-family: Arial, sans-serif;
}}

th, td {{
  text-align: center;
  padding: 8px;
  border: 1px solid #ddd;
}}

th {{
  background-color: #f2f2f2;
  font-weight: bold;
}}

tr:nth-child(even) {{
  background-color: #f9f9f9;
}}

tr:hover {{
  background-color: #f1f1f1;
}}

.significant {{
  font-weight: bold;
  color: #2c7fb8;
}}
</style>

{html_table}
"""

# Save the HTML table
with open('report_assets/results_table.html', 'w') as f:
    f.write(styled_html)

print("Report tables and visualizations created successfully!")

# Create a simplified table for the executive summary
summary_df = pd.DataFrame({
    'Criterion': combined_df['Criterion'],
    'Finetuned Behavioral (%)': combined_df['Behavioral Agreement (%)'].apply(lambda x: f"{x:.1f}"),
    'Base Behavioral (%)': combined_df['Base Behavioral Agreement (%)'].apply(lambda x: f"{x:.1f}"),
    'Explicit Instruction (%)': combined_df['Explicit Instruction Agreement (%)'].apply(lambda x: f"{x:.1f}"),
    'Finetuned Verbal (%)': combined_df['Verbal Self-Awareness (%)'].apply(lambda x: f"{x:.1f}"),
    'Base Verbal (%)': combined_df['Base Verbal Self-Awareness (%)'].apply(lambda x: f"{x:.1f}")
})

# Calculate averages
avg_row = pd.DataFrame({
    'Criterion': ['Average'],
    'Finetuned Behavioral (%)': [f"{combined_df['Behavioral Agreement (%)'].mean():.1f}"],
    'Base Behavioral (%)': [f"{combined_df['Base Behavioral Agreement (%)'].mean():.1f}"],
    'Explicit Instruction (%)': [f"{combined_df['Explicit Instruction Agreement (%)'].mean():.1f}"],
    'Finetuned Verbal (%)': [f"{combined_df['Verbal Self-Awareness (%)'].mean():.1f}"],
    'Base Verbal (%)': [f"{combined_df['Base Verbal Self-Awareness (%)'].mean():.1f}"]
})

# Concatenate the average row to the summary dataframe
summary_df = pd.concat([summary_df, avg_row], ignore_index=True)

# Create an HTML version of the summary table
summary_html = summary_df.to_html(index=False, escape=False)

# Add some CSS styling
styled_summary = f"""
<style>
table {{
  border-collapse: collapse;
  width: 100%;
  font-family: Arial, sans-serif;
}}

th, td {{
  text-align: center;
  padding: 8px;
  border: 1px solid #ddd;
}}

th {{
  background-color: #f2f2f2;
  font-weight: bold;
}}

tr:nth-child(even) {{
  background-color: #f9f9f9;
}}

tr:hover {{
  background-color: #f1f1f1;
}}

tr:last-child {{
  font-weight: bold;
  background-color: #e6f3ff;
}}
</style>

{summary_html}
"""

# Save the summary HTML table
with open('report_assets/summary_table.html', 'w') as f:
    f.write(styled_summary)

print("Summary table created successfully!") 