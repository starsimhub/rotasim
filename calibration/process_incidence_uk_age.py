"""
Process incidence from the model and data - UK version with age-based symptom model

This version supports multiple symptom models:
- 'infection_number': Infection history affects immunity AND severity (no age effect on symptoms)
- 'age_and_infection': Age affects symptoms, infection affects immunity AND severity
- 'age_and_infection_simple': Age affects symptoms, infection affects immunity only (constant severity)

For age-based models:
- Age is in months and centered at 12 months (following Lewnard et al 2019)
- Age is capped at 60 months (5 years) for symptom probability calculation
- Predictor is (age_months - 12): negative for infants <1y, positive for children >1y
- Severity (from infection number) is calculated in the analyzer and used for reporting probability
"""

import sciris as sc
import pandas as pd
import numpy as np
import warnings

# Set current folder
thisdir = sc.thispath(__file__)

# Disable annoying warnings
warnings.simplefilter("ignore", FutureWarning)
pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)


def logistic(x):
    """Logistic function to convert linear predictor to probability"""
    return 1 / (1 + np.exp(-x))


def calculate_symptom_probability(age_months, n_infections, symptom_model='infection_number',
                                   beta0=0, beta1=0, beta2=0, beta3=0, age_cap_months=60):
    """
    Calculate probability of symptomatic infection based on model type

    Args:
        age_months: Age in months at time of infection
        n_infections: Number of prior infections (lifetime count)
        symptom_model: 'infection_number', 'age_only', or 'age_and_infection'
        beta0, beta1, beta2, beta3: Model parameters
        age_cap_months: Maximum age (months) to use in age calculation (default 60 = 5 years)

    Returns:
        symptom_prob: Probability of symptomatic infection [0, 1]

    Note:
        Following Lewnard et al (2019), age is centered at 12 months.
        The predictor is (age_months - 12), so:
        - Negative values = infants under 1 year
        - Zero = 12 months old
        - Positive values = children over 1 year
    """
    if symptom_model == 'infection_number':
        # Original model: first 3 infections are symptomatic
        return 1.0 if n_infections <= 3 else 0.0

    elif symptom_model == 'age_only':
        # Age-based model only, capped at age_cap_months
        age_capped = min(age_months, age_cap_months)
        # Center age at 12 months (following Lewnard et al 2019)
        age_centered = age_capped - 12
        linear_predictor = beta0 + beta1 * age_centered + beta2 * (age_centered ** 2)
        return logistic(linear_predictor)

    elif symptom_model == 'age_and_infection':
        # Combined age and infection number model
        age_capped = min(age_months, age_cap_months)
        # Center age at 12 months (following Lewnard et al 2019)
        age_centered = age_capped - 12
        # Cap infection number at 5 for numerical stability
        n_inf_capped = min(n_infections, 5)
        linear_predictor = beta0 + beta1 * age_centered + beta2 * (age_centered ** 2) + beta3 * n_inf_capped
        return logistic(linear_predictor)

    else:
        raise ValueError(f"Unknown symptom_model: {symptom_model}. Must be 'infection_number', 'age_only', or 'age_and_infection'")


def process_data(filename=None, incidence_sheet=None, age_dist_sheet=None):
    """
    Extract and process UK experimental data

    Returns two dataframes:
    - overall_incidence: Total incidence per 100k (for fitting reporting_rate)
    - age_distribution: Proportion of cases by age (for fitting age distribution shape)
    """
    if filename is None:
        filename = thisdir / 'CalibrationDatafile_prevax 3.xlsx'
    if incidence_sheet is None:
        incidence_sheet = 'UK_incidence'
    if age_dist_sheet is None:
        age_dist_sheet = 'UK_agedistribution'

    # Read overall incidence data
    incidence_data = sc.dataframe.read_excel(filename, sheet_name=incidence_sheet)
    overall_incidence = incidence_data['Cases per 100k'].iloc[0]  # Single value

    # Read age distribution data
    age_dist_data = sc.dataframe.read_excel(filename, sheet_name=age_dist_sheet)

    # Standardize age format
    age_mapping = {
        '[0, 1)': 0,
        '[1, 2)': 1,
        '[2, 5)': 2,
        '[5, 125)': 5,
    }
    ages = age_dist_data['Age'].replace(age_mapping)
    age_distribution = sc.dataframe(dict(ages=ages, proportion=age_dist_data['Proportion']))
    age_distribution = age_distribution.sort_values(by='ages').reset_index(drop=True)

    return overall_incidence, age_distribution


def process_model(dat=None, popsize=None, age_counts=None, verbose=False,
                   symptom_model='infection_number', beta0=0, beta1=0, beta2=0, beta3=0,
                   reporting_rate=None):
    """
    Extract and process data from the UK model with flexible symptom models
    UK data: 2008-2012 (5 years)

    Args:
        dat: DataFrame of infection events
        popsize: Total population size (optional, for backward compatibility)
        age_counts: Dict of actual age-specific population counts
                    {'<1 y': count, '1-2 y': count, '2-5 y': count, '>=5 y': count}
        verbose: Print debug information
        symptom_model: 'infection_number', 'age_only', or 'age_and_infection'
        beta0, beta1, beta2, beta3: Parameters for age-based symptom models
        reporting_rate: Reporting rate to apply AFTER symptom filtering (for age-based models)
    """

    FUDGE = 1.0  # Adjust the fudge factor -- for testing only!!!!

    # Load the data
    if dat is None:
        dat = pd.read_csv(thisdir / '../results/rota_strains_infected_all_1_0.1_2_1_1_0_0.5_1.csv')

    # Look at all years
    dat['Strain3'] = 'Other'
    dat.loc[dat['Strain'] == 'G1P8A1B1', 'Strain3'] = 'G1P8'
    dat.loc[dat['Strain'] == 'G2P4A1B1', 'Strain3'] = 'G2P4'
    dat.loc[dat['Strain'] == 'G9P8A1B1', 'Strain3'] = 'G9P8'

    dat['Year'] = np.floor(dat['CollectionTime']).astype(int)

    # Subset to years 5-10 for UK (5 years of calibration data, after 5-year burn-in)
    # Year 0-4 = burn-in (2003-2007), Year 5-9 = calibration period (2008-2012)
    initial5 = dat[(dat['CollectionTime'] < 10) & (dat['CollectionTime'] >= 5)].copy()

    if verbose: print(initial5['Strain'].value_counts())
    initial5['Strain3'] = 'Other'
    initial5.loc[initial5['Strain'] == 'G1P8A1B1', 'Strain3'] = 'G1P8'
    initial5.loc[initial5['Strain'] == 'G2P4A1B1', 'Strain3'] = 'G2P4'
    initial5.loc[initial5['Strain'] == 'G9P8A1B1', 'Strain3'] = 'G9P8'

    # Working out the case age distribution
    # First making new age bins
    initial5['AgeCat'] = np.nan
    initial5.loc[initial5['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
    initial5.loc[initial5['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
    initial5.loc[initial5['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
    initial5.loc[initial5['Age'] == '60+', 'AgeCat'] = '>=5 y'

    # Convert continuous time to integer years for proper annual aggregation
    initial5['Year'] = np.floor(initial5['CollectionTime']).astype(int)

    # Convert age strings to numeric age in months for symptom calculation
    # For 60+, use 120 months (will be capped at 60 months in symptom calculation anyway)
    age_mapping = {
        '0-2': 1,      # 1 month
        '2-4': 3,      # 3 months
        '4-6': 5,      # 5 months
        '6-12': 9,     # 9 months
        '12-24': 18,   # 18 months
        '24-36': 30,   # 30 months
        '36-48': 42,   # 42 months
        '48-60': 54,   # 54 months
        '60+': 120,    # Adults (will be capped at 60 months = 5 years)
    }
    initial5['age_months'] = initial5['Age'].map(age_mapping)

    # Add infection number per person (lifetime)
    initial5 = initial5.sort_values(['id', 'CollectionTime'])
    initial5['infection_number'] = initial5.groupby('id').cumcount() + 1

    # ========== Apply symptom model ==========
    if verbose:
        print(f"\nUsing symptom model: {symptom_model}")
        if symptom_model in ['age_and_infection', 'age_and_infection_simple']:
            print(f"  beta0={beta0:.3f}, beta1={beta1:.3f}, beta2={beta2:.3f}")

    # For age-based models, filter infections by age-based symptom probability
    # For infection_number model, keep all infections (severity already in data from analyzer)
    if symptom_model in ['age_and_infection', 'age_and_infection_simple']:
        # Calculate age-based symptom probability for each infection
        initial5['symptom_prob'] = initial5.apply(
            lambda row: calculate_symptom_probability(
                age_months=row['age_months'],
                n_infections=row['infection_number'],
                symptom_model='age_only',  # Use age-only formula for both age models
                beta0=beta0, beta1=beta1, beta2=beta2, beta3=0
            ),
            axis=1
        )

        # Stochastically determine which infections are symptomatic
        np.random.seed(int(initial5['id'].iloc[0]) if len(initial5) > 0 else 0)  # Reproducible per simulation
        initial5['is_symptomatic'] = np.random.random(len(initial5)) < initial5['symptom_prob']

        # Filter to symptomatic cases only
        initial5_symptomatic = initial5[initial5['is_symptomatic']].copy()

        if verbose:
            print(f"\nTotal infections: {len(initial5)}")
            print(f"Symptomatic infections: {len(initial5_symptomatic)} ({len(initial5_symptomatic)/len(initial5)*100:.1f}%)")
            print(f"By age category:")
            for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
                total = (initial5['AgeCat'] == age_cat).sum()
                symp = (initial5_symptomatic['AgeCat'] == age_cat).sum()
                if total > 0:
                    print(f"  {age_cat}: {symp}/{total} ({symp/total*100:.1f}%)")

        # Apply reporting filter AFTER age-based symptom filter
        if reporting_rate is not None and 'severity' in initial5_symptomatic.columns:
            initial5_symptomatic['reported'] = np.random.random(len(initial5_symptomatic)) < (reporting_rate * initial5_symptomatic['severity'])
            initial5_reported = initial5_symptomatic[initial5_symptomatic['reported']].copy()

            if verbose:
                print(f"Reported infections (after reporting filter): {len(initial5_reported)} ({len(initial5_reported)/len(initial5_symptomatic)*100:.1f}% of symptomatic)")

            initial5_symptomatic = initial5_reported

    else:
        # infection_number model: no age-based symptom filtering
        # All infections pass through; severity (from analyzer) determines reporting
        initial5_symptomatic = initial5.copy()
        if verbose:
            print(f"\nTotal infections: {len(initial5)} (no symptom filtering for infection_number model)")
    # ===========================================

    # Then take first infection per agent per year (to avoid double-counting within same year)
    initial5_first = initial5_symptomatic.groupby(['id', 'Year', 'AgeCat']).first().reset_index()

    # Now, getting cases by age bin and YEAR
    # Count unique agents per year to get annual incidence
    cases_summary = initial5_first.groupby(['AgeCat', 'Year']).agg(Cases_age=('id', 'nunique')).reset_index()
    if verbose: print(cases_summary.head())

    # Use actual age-specific population counts if provided
    if age_counts is not None:
        # Use the ACTUAL simulated population counts passed from the simulation
        pop_pooled2 = pd.DataFrame([
            {'AgeCat': age_cat, 'Pop_Age': count}
            for age_cat, count in age_counts.items()
        ])

        if verbose:
            print(f"\nUsing actual simulated population counts:")
            for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
                if age_cat in age_counts:
                    print(f"  {age_cat}: {age_counts[age_cat]} agents")

        # Merge incidence with the actual age-specific population denominator
        AgeIncidence = pd.merge(cases_summary, pop_pooled2, on='AgeCat', how='left')

        # Fill NaN values (age categories with 0 cases) with actual population
        AgeIncidence['Pop_Age'] = AgeIncidence['Pop_Age'].fillna(0).astype(int)

    else:
        # BACKWARD COMPATIBILITY: Use old buggy method if age_counts not provided
        initial5_all = dat[(dat['CollectionTime'] < 10) & (dat['CollectionTime'] >= 5)].copy()
        initial5_all['Year'] = np.floor(initial5_all['CollectionTime']).astype(int)

        initial5_all['AgeCat'] = np.nan
        initial5_all.loc[initial5_all['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
        initial5_all.loc[initial5_all['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
        initial5_all.loc[initial5_all['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
        initial5_all.loc[initial5_all['Age'] == '60+', 'AgeCat'] = '>=5 y'

        pop_snapshots = initial5_all.sort_values('CollectionTime').groupby(['Year', 'id']).tail(1)
        pop_age_counts = pop_snapshots.groupby(['Year', 'AgeCat']).agg(
            Pop_Age=('id', 'nunique'),
            PopulationSize=('PopulationSize', 'first')
        ).reset_index()

        pop_pooled2 = pop_age_counts[['Year', 'AgeCat', 'Pop_Age', 'PopulationSize']]
        AgeIncidence = pd.merge(cases_summary, pop_pooled2, on=['AgeCat', 'Year'])

    AgeIncidence['IR_100k'] = (AgeIncidence['Cases_age'] / AgeIncidence['Pop_Age']) * 100000 / FUDGE

    # Get an average to calibrate to average pre-vaccine period
    Inci_dist = AgeIncidence.groupby('AgeCat').agg(meanIR=('IR_100k', 'mean')).reset_index()

    # Standardize data format - ensure ALL 4 age categories are present
    expected_ages = [0, 1, 2, 5]
    expected_age_cats = ['<1 y', '1-2 y', '2-5 y', '>=5 y']
    df = pd.DataFrame({'ages': expected_ages, 'inci': [0.0, 0.0, 0.0, 0.0]})

    # Fill in actual incidence values where data exists
    for idx, age_cat in enumerate(expected_age_cats):
        if age_cat in Inci_dist['AgeCat'].values:
            actual_inci = Inci_dist[Inci_dist['AgeCat'] == age_cat]['meanIR'].values[0]
            df.loc[idx, 'inci'] = actual_inci

    df = df.sort_values(by='ages').reset_index(drop=True)

    # Calculate population-weighted overall incidence
    age_order = ['<1 y', '1-2 y', '2-5 y', '>=5 y']

    if age_counts is not None:
        # NEW CODE PATH: Use passed age_counts directly
        pop_counts = np.array([age_counts.get(age_cat, 0) for age_cat in age_order])
    else:
        # OLD CODE PATH: Use pop_age_counts (backward compatibility)
        mean_pop_by_age = pop_age_counts.groupby('AgeCat').agg(
            mean_pop=('Pop_Age', 'mean')
        ).reset_index()

        # Convert to sorted array matching df order
        pop_counts = []
        for age_cat in age_order:
            if age_cat in mean_pop_by_age['AgeCat'].values:
                pop_counts.append(mean_pop_by_age[mean_pop_by_age['AgeCat'] == age_cat]['mean_pop'].values[0])
            else:
                pop_counts.append(0)
        pop_counts = np.array(pop_counts)

    total_pop = pop_counts.sum()
    pop_fractions = pop_counts / total_pop if total_pop > 0 else np.array([0.0126, 0.0127, 0.0366, 0.9381])

    if verbose:
        print(f"\nActual population fractions from simulation:")
        for age_cat, frac in zip(age_order, pop_fractions):
            print(f"  {age_cat}: {frac*100:.2f}%")

    # Calculate weighted incidence: sum(incidence_rate * pop_fraction)
    overall_incidence = sum(df['inci'].values * pop_fractions)

    # Also calculate age distribution (proportions)
    # This represents the proportion of CASES, not population
    case_fractions = df['inci'].values * pop_fractions / overall_incidence if overall_incidence > 0 else np.zeros(len(df))
    age_distribution = sc.dataframe(dict(ages=df['ages'], proportion=case_fractions))

    return overall_incidence, age_distribution


if __name__ == '__main__':

    # Test all three models
    print("\n" + "="*60)
    print("Testing symptom models")
    print("="*60)

    # Example: test symptom probability at different ages
    ages = [0, 1, 2, 3, 4, 5, 10]

    print("\nAge-only model (beta0=-2, beta1=-0.5, beta2=-0.02):")
    for age in ages:
        prob = calculate_symptom_probability(age_years=age, n_infections=1, symptom_model='age_only',
                                             beta0=-2, beta1=-0.5, beta2=-0.02)
        print(f"  Age {age}: P(symptomatic) = {prob:.3f}")

    print("\nInfection-number model:")
    for n_inf in [1, 2, 3, 4, 5]:
        prob = calculate_symptom_probability(age_years=2, n_infections=n_inf, symptom_model='infection_number')
        print(f"  Infection {n_inf}: P(symptomatic) = {prob:.3f}")
