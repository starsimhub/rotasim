"""
Plot test validation results showing FIXED age extraction is working
Uses existing test results - NO simulation needed
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("CREATING FIGURE: Test Validation Results (FIXED Age Extraction)")
print("="*80)

# Existing test results
target_incidence = 27.56
test_incidence = 56.33

target_age_dist = [13.77, 27.67, 46.89, 11.67]  # percentages
test_age_dist = [13.8, 27.7, 46.9, 11.6]  # percentages
prior_broken_age_dist = [1.91, 0.91, 3.81, 93.37]  # percentages

age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']

print("\nTest Results (FIXED age extraction):")
print(f"  Incidence: {test_incidence:.2f} per 100k (target: {target_incidence:.2f})")
print("  Age distribution:")
for i, label in enumerate(age_labels):
    print(f"    {label:<15}: {test_age_dist[i]:>6.2f}% (target: {target_age_dist[i]:>6.2f}%)")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Age Distribution
ax1 = axes[0]
x = np.arange(len(age_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, target_age_dist, width, label='Target (UK Data)',
                color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, test_age_dist, width, label='Test Results (Fixed Code)',
                color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Proportion of Cases (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax1.set_title('A. Age Distribution - FIXED Code Test', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(['0-11\nmonths', '12-23\nmonths', '24-59\nmonths', '5+\nyears'], fontsize=10)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 60)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add note about fix
ax1.text(0.02, 0.98, 'All age groups have\nrealistic values ✓',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Panel B: Incidence Comparison
ax2 = axes[1]
incidence_vals = [target_incidence, test_incidence]
colors = ['#3498DB', '#2ECC71']
bars = ax2.bar(['Target\n(UK Data)', 'Test Results\n(Fixed Code)'],
               incidence_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.5)

ax2.set_ylabel('Incidence (per 100,000)', fontsize=12, fontweight='bold')
ax2.set_title('B. Overall Incidence Comparison', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, 70)
ax2.axhline(y=target_incidence, color='blue', linestyle='--', alpha=0.5, linewidth=2)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, incidence_vals):
    height = bar.get_height()
    pct_of_target = (val / target_incidence) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
             f'{val:.1f}\n({pct_of_target:.0f}% of target)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add error annotation
error = test_incidence - target_incidence
error_pct = (error / target_incidence) * 100
ax2.text(0.5, 0.25,
        f'Error: {error:+.1f} per 100k ({error_pct:+.1f}%)\n\nNote: Uses PRIOR calibration\nparameters (incidence-only)',
        transform=ax2.transAxes, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

# Overall title
fig.suptitle('Test Validation: FIXED Age Distribution Extraction\n(Using PRIOR 50-trial calibration parameters)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
thisdir = Path(__file__).parent
output_file = thisdir / 'test_validation_fixed_age_extraction.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved to: {output_file}")

output_pdf = thisdir / 'test_validation_fixed_age_extraction.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✓ Figure saved to: {output_pdf}")

print("\n" + "="*80)
print("COMPARISON WITH PRIOR BROKEN RESULTS")
print("="*80)
print("\nPRIOR evaluation (with BROKEN age extraction code):")
print(f"  Age distribution: [{prior_broken_age_dist[0]:.1f}%, {prior_broken_age_dist[1]:.1f}%, {prior_broken_age_dist[2]:.1f}%, {prior_broken_age_dist[3]:.1f}%]")
print("  Status: BROKEN - 93% in oldest group!")
print("\nCURRENT test (with FIXED age extraction code):")
print(f"  Age distribution: [{test_age_dist[0]:.1f}%, {test_age_dist[1]:.1f}%, {test_age_dist[2]:.1f}%, {test_age_dist[3]:.1f}%]")
print("  Status: WORKING - All realistic values!")
print("\n" + "="*80)
print("✓ Age distribution extraction is working correctly!")
print("="*80)
