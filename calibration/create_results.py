import json
import sciris as sc

# Best parameters from Trial 8
best_parameters = {
    'beta0': 1.7919587439896336,
    'beta1': 0.26964539547974015,
    'beta2': -0.020868246115113457,
    'constant_severity': 0.1403169913459729,
    'reporting_rate': 0.029057101336531343,
    'base_beta': 3.020741657278965
}

# Target incidence
target_incidence = 27.56

# Create results structure (we'll need to extract before/after from log or re-run)
# For now, create minimal results with what we have
results = {
    'symptom_model': 'age_and_infection_simple_with_severity',
    'best_parameters': best_parameters,
    'incidence': {
        'target': target_incidence,
        'fitted_severity': best_parameters['constant_severity'],
        'note': 'Severity fitted as free parameter (14.0%) vs previously fixed at 5%'
    }
}

thisdir = sc.thispath(__file__)
results_file = thisdir / 'uk_calibration_results_age_and_infection_simple_with_severity.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {results_file}")
print("\nKey finding:")
print(f"  Optimal severity: {best_parameters['constant_severity']:.4f} (14.0%)")
print(f"  Previously fixed at: 0.05 (5.0%)")
print(f"  Improvement: {best_parameters['constant_severity'] / 0.05:.1f}x higher")
