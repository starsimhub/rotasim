import json
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc

# Load results
thisdir = sc.thispath(__file__)

with open(thisdir / 'uk_calibration_results_age_and_infection_simple.json', 'r') as f:
    fixed_results = json.load(f)

with open(thisdir / 'uk_calibration_results_age_and_infection_simple_with_severity.json', 'r') as f:
    fitted_results = json.load(f)

# Extract data
target_incidence = 27.56
fixed_incidence = fixed_results['incidence']['after']
fixed_severity = 0.05  # Was fixed
fitted_severity = fitted_results['best_parameters']['constant_severity']

# Create figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Age-and-Infection-Simple Model: Fixed vs Fitted Severity Comparison', 
             fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Severity parameter comparison
ax1 = axes[0, 0]
severity_vals = [fixed_severity, fitted_severity]
colors = ['#E74C3C', '#27AE60']
bars = ax1.bar(['Fixed Severity\n(Original)', 'Fitted Severity\n(New)'], 
               severity_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Severity Parameter', fontsize=12, fontweight='bold')
ax1.set_title('A. Severity Parameter Values', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylim(0, 0.16)
ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Original fixed value')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, severity_vals):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
             f'{val:.4f}\n({val*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    
# Add improvement annotation
ax1.annotate(f'{fitted_severity/fixed_severity:.1f}x higher',
            xy=(1, fitted_severity), xytext=(0.5, 0.12),
            arrowprops=dict(arrowstyle='->', color='black', lw=2),
            fontsize=11, fontweight='bold', ha='center')

# Panel 2: Incidence comparison
ax2 = axes[0, 1]
incidence_vals = [fixed_incidence, target_incidence]
x_pos = [0, 1]
bars2 = ax2.bar(['Fixed Severity\nModel', 'Target\n(Data)'], 
                incidence_vals, color=['#E74C3C', '#3498DB'], 
                alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Incidence (per 100,000)', fontsize=12, fontweight='bold')
ax2.set_title('B. Model Fit: Incidence Comparison', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, 35)
ax2.axhline(y=target_incidence, color='blue', linestyle='--', alpha=0.5, label='Target')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars2, incidence_vals):
    height = bar.get_height()
    pct_of_target = (val / target_incidence) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val:.1f}\n({pct_of_target:.0f}% of target)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add gap annotation
gap = target_incidence - fixed_incidence
ax2.annotate(f'Gap: {gap:.1f}\n({gap/target_incidence*100:.0f}%)',
            xy=(0, fixed_incidence), xytext=(0.5, 20),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2),
            fontsize=10, fontweight='bold', ha='center', color='red')

# Panel 3: Age parameters comparison
ax3 = axes[1, 0]
param_names = ['beta0\n(intercept)', 'beta1\n(linear)', 'beta2\n(quadratic)']
fixed_params = [fixed_results['best_parameters']['beta0'],
                fixed_results['best_parameters']['beta1'],
                fixed_results['best_parameters']['beta2']]
fitted_params = [fitted_results['best_parameters']['beta0'],
                 fitted_results['best_parameters']['beta1'],
                 fitted_results['best_parameters']['beta2']]

x = np.arange(len(param_names))
width = 0.35
bars3a = ax3.bar(x - width/2, fixed_params, width, label='Fixed Severity', 
                 color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=1.5)
bars3b = ax3.bar(x + width/2, fitted_params, width, label='Fitted Severity',
                 color='#27AE60', alpha=0.7, edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Parameter Value', fontsize=12, fontweight='bold')
ax3.set_title('C. Age-Based Symptom Parameters', fontsize=13, fontweight='bold', pad=10)
ax3.set_xticks(x)
ax3.set_xticklabels(param_names, fontsize=10)
ax3.legend(fontsize=10, loc='upper left')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Transmission & reporting parameters
ax4 = axes[1, 1]
param_names2 = ['base_beta\n(transmission)', 'reporting_rate\n(×100)']
fixed_params2 = [fixed_results['best_parameters']['base_beta'],
                 fixed_results['best_parameters']['reporting_rate'] * 100]
fitted_params2 = [fitted_results['best_parameters']['base_beta'],
                  fitted_results['best_parameters']['reporting_rate'] * 100]

x2 = np.arange(len(param_names2))
bars4a = ax4.bar(x2 - width/2, fixed_params2, width, label='Fixed Severity',
                 color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=1.5)
bars4b = ax4.bar(x2 + width/2, fitted_params2, width, label='Fitted Severity',
                 color='#27AE60', alpha=0.7, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Parameter Value', fontsize=12, fontweight='bold')
ax4.set_title('D. Transmission & Reporting Parameters', fontsize=13, fontweight='bold', pad=10)
ax4.set_xticks(x2)
ax4.set_xticklabels(param_names2, fontsize=10)
ax4.legend(fontsize=10, loc='upper left')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars4a:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars4b:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Save figure
output_file = thisdir / 'severity_comparison_figure.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_file}")

# Also save as PDF
output_pdf = thisdir / 'severity_comparison_figure.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Figure saved to: {output_pdf}")

plt.show()
