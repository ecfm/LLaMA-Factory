import matplotlib.pyplot as plt
import numpy as np

# Data for risk seeking rating
models = ['qwen14b_lora_cp30', 'qwen14b_lora_cp40', 'qwen14b_lora_cp50', 'qwen14b']
risk_seeking_mean = [65.60, 46.00, 79.50, 46.80]
risk_seeking_std = [13.11, 7.25, 4.97, 7.44]

# Extract checkpoint numbers for x-axis (except base model)
checkpoints = [30, 40, 50]

# Agreement rate data from the JSON
agreement_rates = {
    'checkpoint-30': 0.5333333333333333,
    'checkpoint-40': 0.5333333333333333,
    'checkpoint-50': 0.5333333333333333,
    'base': 0.5666666666666667
}

# Set larger font sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9
})

# Create figure with two y-axes - reduced to 50% of original size
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()

# Plot risk seeking rating on the first y-axis
ax1.plot(checkpoints, risk_seeking_mean[:3], 'b--o', linewidth=2, markersize=10, label='Risk Seeking Rating (LoRA)')
ax1.axhline(y=risk_seeking_mean[3], color='b', linestyle=':', linewidth=2, label='Base Model Risk Seeking')

# Plot agreement rate on the second y-axis
ax2.plot(checkpoints, [agreement_rates[f'checkpoint-{cp}'] for cp in checkpoints], 'r--o', linewidth=2, markersize=10, label='Agreement Rate (LoRA)')
ax2.axhline(y=agreement_rates['base'], color='r', linestyle=':', linewidth=2, label='Base Model Agreement Rate')

# Add labels and title
ax1.set_xlabel('Checkpoint', fontweight='bold')
ax1.set_ylabel('Risk Seeking Rating', color='b', fontweight='bold')
ax2.set_ylabel('Agreement Rate', color='r', fontweight='bold')
plt.title('Risk Seeking Rating and Agreement Rate by Checkpoint', fontweight='bold', pad=20)

# Set x-ticks to checkpoint numbers
plt.xticks(checkpoints)

# Add legends
ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
ax2.legend(loc='upper right', frameon=True, framealpha=0.9)

# Add error bars for risk seeking rating
ax1.errorbar(checkpoints, risk_seeking_mean[:3], yerr=risk_seeking_std[:3], fmt='none', ecolor='b', capsize=5, elinewidth=2)

# Adjust layout and display
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot as PNG
plt.savefig('risk_seeking_agreement_plot.png', dpi=300, bbox_inches='tight')
plt.close()
