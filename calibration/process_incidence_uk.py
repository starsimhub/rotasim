"""
Process incidence from the model and data - UK version
Data covers 2008-2012 (5 years)
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


def process_model(dat=None, popsize=None, age_counts=None, verbose=False):
    """
    Extract and process data from the UK model
    UK data: 2008-2012 (5 years)

    Args:
        dat: DataFrame of infection events
        popsize: Total population size (optional, for backward compatibility)
        age_counts: Dict of actual age-specific population counts
                    {'<1 y': count, '1-2 y': count, '2-5 y': count, '>=5 y': count}
        verbose: Print debug information
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
    YearlyGenoDat = dat.groupby(['Year', 'Strain3']).agg(Geno_cases=('id', 'nunique')).reset_index()
    YearlyCases = dat.groupby('Year').agg(All_cases=('id', 'nunique'), mean_pop=('PopulationSize', 'mean')).reset_index()
    CasesGeno = pd.merge(YearlyGenoDat, YearlyCases, on='Year')
    CasesGeno['geno_prop'] = CasesGeno['Geno_cases'] / CasesGeno['All_cases']

    # Subset to years 5-10 for UK (5 years of calibration data, after 5-year burn-in)
    # Year 0-4 = burn-in (2003-2007), Year 5-9 = calibration period (2008-2012)
    initial5 = dat[(dat['CollectionTime'] < 10) & (dat['CollectionTime'] >= 5)]

    if verbose: print(initial5['Strain'].value_counts())
    initial5['Strain3'] = 'Other'
    initial5.loc[initial5['Strain'] == 'G1P8A1B1', 'Strain3'] = 'G1P8'
    initial5.loc[initial5['Strain'] == 'G2P4A1B1', 'Strain3'] = 'G2P4'
    initial5.loc[initial5['Strain'] == 'G9P8A1B1', 'Strain3'] = 'G9P8'

    GenoDist = initial5['Strain3'].value_counts().reset_index()
    GenoDist.columns = ['Strain', 'Frequency']
    total = GenoDist['Frequency'].sum()
    GenoDist['Proportion'] = GenoDist['Frequency'] / total

    # Working out the case age distribution
    # First making new age bins
    initial5['AgeCat'] = np.nan
    initial5.loc[initial5['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
    initial5.loc[initial5['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
    initial5.loc[initial5['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
    initial5.loc[initial5['Age'] == '60+', 'AgeCat'] = '>=5 y'

    # Convert continuous time to integer years for proper annual aggregation
    initial5['Year'] = np.floor(initial5['CollectionTime']).astype(int)

    # Clinical data counts symptomatic/reported cases
    # Count first few infections per person (these are typically symptomatic)
    # Later infections are usually asymptomatic due to acquired immunity
    # Add infection number per person (lifetime)
    initial5 = initial5.sort_values(['id', 'CollectionTime'])
    initial5['infection_number'] = initial5.groupby('id').cumcount() + 1

    # Count only first 3 infections per person (symptomatic threshold)
    # Use uniform threshold across all ages - let rel_beta affect age distribution naturally
    initial5_symptomatic = initial5[initial5['infection_number'] <= 3].copy()

    # Then take first infection per agent per year (to avoid double-counting within same year)
    initial5_first = initial5_symptomatic.groupby(['id', 'Year', 'AgeCat']).first().reset_index()

    # Now, getting cases by age bin and YEAR
    # Count unique agents per year to get annual incidence
    cases_summary = initial5_first.groupby(['AgeCat', 'Year']).agg(Cases_age=('id', 'nunique')).reset_index()
    if verbose: print(cases_summary.head())

    # Use actual age-specific population counts if provided (FIX FOR BUG)
    if age_counts is not None:
        # Use the ACTUAL simulated population counts passed from the simulation
        # This replaces the buggy code that counted infected people as population
        pop_pooled2 = pd.DataFrame([
            {'AgeCat': age_cat, 'Pop_Age': count}
            for age_cat, count in age_counts.items()
        ])

        if verbose:
            print(f"Using actual simulated population counts:")
            for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
                if age_cat in age_counts:
                    print(f"  {age_cat}: {age_counts[age_cat]} agents")

        # Merge incidence with the actual age-specific population denominator
        AgeIncidence = pd.merge(cases_summary, pop_pooled2, on='AgeCat', how='left')

        # Fill NaN values (age categories with 0 cases) with actual population
        AgeIncidence['Pop_Age'] = AgeIncidence['Pop_Age'].fillna(0).astype(int)

    else:
        # BACKWARD COMPATIBILITY: Use old buggy method if age_counts not provided
        # This code path should not be used in production
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
    age_mapping = {
        '<1 y': 0,
        '1-2 y': 1,
        '2-5 y': 2,
        '>=5 y': 5,
    }

    # Create a complete dataframe with all 4 age categories (initialized to zero)
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
    # Use ACTUAL population fractions from the simulation, not hardcoded values
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

    # Calculate weighted incidence: sum(incidence_rate * pop_fraction * pop_size) / pop_size
    # Simplifies to: sum(incidence_rate * pop_fraction)
    overall_incidence = sum(df['inci'].values * pop_fractions)

    # Also calculate age distribution (proportions)
    # This represents the proportion of CASES, not population
    # = (incidence_rate * pop_fraction) / overall_incidence
    case_fractions = df['inci'].values * pop_fractions / overall_incidence
    age_distribution = sc.dataframe(dict(ages=df['ages'], proportion=case_fractions))

    return overall_incidence, age_distribution

if __name__ == '__main__':

    m_inci = process_model()
    d_inci = process_data()
