"""
Analyze the quality of the calibration fit
Compare model output to target data for best parameters
"""

import optuna
import numpy as np
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

# Import the calibration modules
import process_incidence_uk as process_uk
from calibrate_uk import make_sim

# Load best trial from database
storage = 'sqlite:///rota.db'
study = optuna.load_study(study_name='rota', storage=storage)

print('='*70)
print('UK CALIBRATION FIT QUALITY ANALYSIS')
print('='*70)
print(f'\nBest Trial: #{study.best_trial.number}')
print(f'Best GOF: {study.best_value:.6f}\n')

# Get best parameters
best_params = study.best_params
print('Best Parameters:')
for k, v in best_params.items():
    if k == 'adult_baseline_immunity':
        print(f'  {k}: {v:.6f} ← PRIMARY PARAMETER')
    else:
        print(f'  {k}: {v:.6f}')

# Load target data
print('\n' + '='*70)
print('TARGET DATA (UK 2008-2012)')
print('='*70)
target_overall, target_age_dist = process_uk.process_data()
print(f'Overall incidence: {target_overall:.2f} per 100k')
print('\nAge distribution:')
for idx, row in target_age_dist.iterrows():
    age_labels = {0: '<1y', 1: '1-2y', 2: '2-5y', 5: '≥5y'}
    age_label = age_labels.get(row['ages'], f"{row['ages']}y")
    print(f'  {age_label:4s}: {row["proportion"]*100:5.2f}%')

# Run simulation with best parameters
print('\n' + '='*70)
print('RUNNING SIMULATION WITH BEST PARAMETERS...')
print('='*70)

# Create sim with best parameters
sim_pars = {k: v for k, v in best_params.items()}
sim = make_sim(seed=0, sim_pars=sim_pars)
sim.run()

# Process model output
df = sim.connectors['rota'].event_dict
model_overall, model_age_dist = process_uk.process_model(df, verbose=True)

print('\n' + '='*70)
print('MODEL OUTPUT')
print('='*70)
print(f'Overall incidence: {model_overall:.2f} per 100k')
print('\nAge distribution:')
for idx, row in model_age_dist.iterrows():
    age_labels = {0: '<1y', 1: '1-2y', 2: '2-5y', 5: '≥5y'}
    age_label = age_labels.get(row['ages'], f"{row['ages']}y")
    print(f'  {age_label:4s}: {row["proportion"]*100:5.2f}%')

# Calculate fit quality metrics
print('\n' + '='*70)
print('FIT QUALITY METRICS')
print('='*70)

# Overall incidence error
incidence_error = abs(model_overall - target_overall)
incidence_pct_error = (incidence_error / target_overall) * 100
print(f'\nOverall Incidence:')
print(f'  Target:    {target_overall:.2f} per 100k')
print(f'  Model:     {model_overall:.2f} per 100k')
print(f'  Error:     {incidence_error:.2f} per 100k ({incidence_pct_error:.1f}%)')

# Age distribution errors
print(f'\nAge Distribution:')
print(f'  {"Age":<6} {"Target":>8} {"Model":>8} {"Abs Error":>10} {"% Error":>10}')
print(f'  {"-"*6} {"-"*8} {"-"*8} {"-"*10} {"-"*10}')

age_errors = []
for idx in range(len(target_age_dist)):
    target_prop = target_age_dist.iloc[idx]['proportion']
    model_prop = model_age_dist.iloc[idx]['proportion']
    age = target_age_dist.iloc[idx]['ages']

    age_labels = {0: '<1y', 1: '1-2y', 2: '2-5y', 5: '≥5y'}
    age_label = age_labels.get(age, f"{age}y")

    abs_error = abs(model_prop - target_prop)
    pct_error = abs_error * 100  # Already in proportion form
    age_errors.append(abs_error)

    print(f'  {age_label:<6} {target_prop*100:7.2f}% {model_prop*100:7.2f}% {abs_error*100:9.2f}% {pct_error:9.2f}%')

mean_age_error = np.mean(age_errors) * 100
max_age_error = np.max(age_errors) * 100

print(f'\n  Mean absolute error: {mean_age_error:.2f}%')
print(f'  Max absolute error:  {max_age_error:.2f}%')

# Overall GOF breakdown
print(f'\nGOF Calculation:')
print(f'  Incidence component: (|{model_overall:.2f} - {target_overall:.2f}| / {target_overall:.2f}) * 100 = {incidence_pct_error:.2f}')
print(f'  Age dist component:  Mean of absolute percentage errors = {mean_age_error:.2f}')
print(f'  Total GOF: {incidence_pct_error + mean_age_error:.2f}')
print(f'  (Optuna reported: {study.best_value:.2f})')

print('\n' + '='*70)
print('FIT QUALITY ASSESSMENT')
print('='*70)

# Assess fit quality
if incidence_pct_error < 10:
    print('✓ Overall incidence: EXCELLENT fit (<10% error)')
elif incidence_pct_error < 20:
    print('✓ Overall incidence: GOOD fit (10-20% error)')
elif incidence_pct_error < 30:
    print('○ Overall incidence: ACCEPTABLE fit (20-30% error)')
else:
    print('✗ Overall incidence: POOR fit (>30% error)')

if mean_age_error < 5:
    print('✓ Age distribution: EXCELLENT fit (<5% mean error)')
elif mean_age_error < 10:
    print('✓ Age distribution: GOOD fit (5-10% mean error)')
elif mean_age_error < 15:
    print('○ Age distribution: ACCEPTABLE fit (10-15% mean error)')
else:
    print('✗ Age distribution: POOR fit (>15% mean error)')

print('\n' + '='*70)
