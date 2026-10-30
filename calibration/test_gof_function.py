"""
Test the goodness-of-fit function to understand its behavior
"""
import numpy as np
import sciris as sc

thisdir = sc.thispath(__file__)
from calibration import compute_gof

# Test case: actual data
actual = np.array([425.0, 312.5, 14.583, 0.952])

# Test scenarios
scenarios = {
    "Perfect match": actual.copy(),
    "10% too low": actual * 0.9,
    "10% too high": actual * 1.1,
    "50% too low": actual * 0.5,
    "50% too high": actual * 1.5,
    "Way too high (2x)": actual * 2.0,
}

print("="*60)
print("Goodness-of-Fit Function Testing")
print("="*60)
print(f"\nActual data: {actual}")
print(f"\nLower GOF score = better fit")
print("-"*60)

for name, predicted in scenarios.items():
    gofs = compute_gof(actual, predicted)
    total_gof = gofs.sum()

    print(f"\n{name}:")
    print(f"  Predicted: {predicted}")
    print(f"  Individual GOFs: {gofs}")
    print(f"  Total GOF: {total_gof:.4f}")

    # Show relative error
    rel_error = np.abs(predicted - actual) / actual * 100
    print(f"  Relative errors: {rel_error}%")

print("\n" + "="*60)
print("Analysis:")
print("="*60)

# Compare underestimation vs overestimation
under_50 = compute_gof(actual, actual * 0.5).sum()
over_50 = compute_gof(actual, actual * 1.5).sum()

print(f"\n50% underestimate GOF: {under_50:.4f}")
print(f"50% overestimate GOF:  {over_50:.4f}")
print(f"\nDifference: {abs(under_50 - over_50):.4f}")

if under_50 < over_50:
    print("✓ Function penalizes overestimation more (GOOD)")
elif over_50 < under_50:
    print("⚠ Function penalizes underestimation more (BAD - favors overestimation)")
else:
    print("✓ Function treats over/under estimation equally")

# Test with the specific issue: does higher incidence give lower GOF?
print("\n" + "="*60)
print("Testing calibration direction:")
print("="*60)

base_pred = actual * 0.8  # 20% too low
high_pred = actual * 1.2   # 20% too high

base_gof = compute_gof(actual, base_pred).sum()
high_gof = compute_gof(actual, high_pred).sum()

print(f"\nIf model produces 20% too low:  GOF = {base_gof:.4f}")
print(f"If model produces 20% too high: GOF = {high_gof:.4f}")

if base_gof < high_gof:
    print("✓ Lower incidence = better GOF (calibration will DECREASE beta)")
elif high_gof < base_gof:
    print("⚠ Higher incidence = better GOF (calibration will INCREASE beta)")
    print("   THIS IS THE PROBLEM!")
