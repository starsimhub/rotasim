"""
Process incidence from the model and data
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
    Extract and process experimental data

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


def process_model(dat=None, popsize=None, verbose=False):
    """
    Extract and process data from the model
    """

    FUDGE = 1.0 # Adjust the fudge factor -- for testing only!!!!

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

    # Subset to years 11-19 (2000-2008 in simulations starting from 1990)
    # This gives 10 years of burn-in (1990-2000) before calibration period
    #Will have to manually adjust this for each run for now
    initial = dat[(dat['CollectionTime'] < 10) & (dat['CollectionTime'] > 4)]

    if verbose: print(initial['Strain'].value_counts())
    initial['Strain3'] = 'Other'
    initial.loc[initial['Strain'] == 'G1P8A1B1', 'Strain3'] = 'G1P8'
    initial.loc[initial['Strain'] == 'G2P4A1B1', 'Strain3'] = 'G2P4'
    initial.loc[initial['Strain'] == 'G9P8A1B1', 'Strain3'] = 'G9P8'

    GenoDist = initial['Strain3'].value_counts().reset_index()
    GenoDist.columns = ['Strain', 'Frequency']
    total = GenoDist['Frequency'].sum()
    GenoDist['Proportion'] = GenoDist['Frequency'] / total

    # Working out the case age distribution
    # First making new age bins
    initial['AgeCat'] = np.nan
    initial.loc[initial['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
    initial.loc[initial['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
    initial.loc[initial['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
    initial.loc[initial['Age'] == '60+', 'AgeCat'] = '>=5 y'

    # Convert continuous time to integer years for proper annual aggregation
    initial['Year'] = np.floor(initial['CollectionTime']).astype(int)

    # Clinical data counts symptomatic/reported cases
    # Count first few infections per person (these are typically symptomatic)
    # Later infections are usually asymptomatic due to acquired immunity
    # Add infection number per person (lifetime)
    initial = initial.sort_values(['id', 'CollectionTime'])
    initial['infection_number'] = initial.groupby('id').cumcount() + 1

    # Count only first 3 infections per person (symptomatic threshold)
    # Use uniform threshold across all ages - let rel_beta affect age distribution naturally
    initial_symptomatic = initial[initial['infection_number'] <= 4].copy()

    # Then take first infection per agent per year (to avoid double-counting within same year)
    initial_first = initial_symptomatic.groupby(['id', 'Year', 'AgeCat']).first().reset_index()

    # Now, getting cases by age bin and YEAR
    # Count unique agents per year to get annual incidence
    cases_summary = initial_first.groupby(['AgeCat', 'Year']).agg(Cases_age=('id', 'nunique')).reset_index()
    if verbose: print(cases_summary.head())

    # Now, getting total population by time point (year)
    # Calculate ACTUAL age-specific population sizes from the simulation data
    # We need to count all unique agents in each age category at each timepoint

    # Get all infection events (not just symptomatic) to capture full population age structure
    initial_all = dat[(dat['CollectionTime'] < 10) & (dat['CollectionTime'] > 4)].copy()
    initial_all['Year'] = np.floor(initial_all['CollectionTime']).astype(int)

    # Assign age categories to all infection events
    initial_all['AgeCat'] = np.nan
    initial_all.loc[initial_all['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
    initial_all.loc[initial_all['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
    initial_all.loc[initial_all['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
    initial_all.loc[initial_all['Age'] == '60+', 'AgeCat'] = '>=5 y'

    # For each year, get a representative snapshot of population age distribution
    # Take the latest timepoint in each year as the population snapshot
    pop_snapshots = initial_all.sort_values('CollectionTime').groupby(['Year', 'id']).tail(1)

    # Count unique agents in each age bin per year to get actual age-specific population
    pop_age_counts = pop_snapshots.groupby(['Year', 'AgeCat']).agg(
        Pop_Age=('id', 'nunique'),
        PopulationSize=('PopulationSize', 'first')  # Total population
    ).reset_index()

    # Verify population counts make sense
    total_pop_check = pop_age_counts.groupby('Year')['Pop_Age'].sum()
    if verbose:
        print(f"Population age distribution (mean across years):")
        age_dist_check = pop_age_counts.groupby('AgeCat')['Pop_Age'].mean()
        for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
            if age_cat in age_dist_check.index:
                print(f"  {age_cat}: {age_dist_check[age_cat]:.0f} agents")

    pop_pooled2 = pop_age_counts[['Year', 'AgeCat', 'Pop_Age', 'PopulationSize']]

    # Merge incidence with the age and time appropriate denominator to get the ratio
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
    # Calculate mean age-specific population across years
    mean_pop_by_age = pop_age_counts.groupby('AgeCat').agg(
        mean_pop=('Pop_Age', 'mean')
    ).reset_index()

    # Convert to sorted array matching df order
    age_order = ['<1 y', '1-2 y', '2-5 y', '>=5 y']
    pop_counts = []
    for age_cat in age_order:
        if age_cat in mean_pop_by_age['AgeCat'].values:
            pop_counts.append(mean_pop_by_age[mean_pop_by_age['AgeCat'] == age_cat]['mean_pop'].values[0])
        else:
            pop_counts.append(0)

    pop_counts = np.array(pop_counts)
    total_pop = pop_counts.sum()
    pop_fractions = pop_counts / total_pop if total_pop > 0 else np.array([0.025, 0.025, 0.075, 0.875])

    if verbose:
        print(f"\nActual population fractions from simulation:")
        for age_cat, frac in zip(age_order, pop_fractions):
            print(f"  {age_cat}: {frac*100:.1f}%")

    # Check if there are any infections recorded
    if len(df) == 0:
        # No infections recorded - return zero incidence with default age distribution
        overall_incidence = 0.0
        age_distribution = sc.dataframe(dict(ages=[0, 1, 2, 5], proportion=[0.25, 0.25, 0.25, 0.25]))
    else:
        # Calculate weighted incidence: sum(incidence_rate * pop_fraction * pop_size) / pop_size
        # Simplifies to: sum(incidence_rate * pop_fraction)
        overall_incidence = sum(df['inci'].values * pop_fractions)

        # Also calculate age distribution (proportions)
        # This represents the proportion of CASES, not population
        # = (incidence_rate * pop_fraction) / overall_incidence
        if overall_incidence > 0:
            case_fractions = df['inci'].values * pop_fractions / overall_incidence
        else:
            # If incidence is zero (or very close), use equal distribution
            case_fractions = np.array([0.25, 0.25, 0.25, 0.25])

        # Ensure we always have all 4 age categories [0, 1, 2, 5]
        # even if some had zero infections (fill with 0.0)
        age_map = {0: 0, 1: 1, 2: 2, 5: 3}  # Map age to index in case_fractions
        full_ages = [0, 1, 2, 5]
        full_proportions = []

        for age in full_ages:
            if age in df['ages'].values:
                idx = df[df['ages'] == age].index[0]
                age_idx = age_map[age]
                full_proportions.append(case_fractions[age_idx])
            else:
                # Age group had no infections
                full_proportions.append(0.0)

        age_distribution = sc.dataframe(dict(ages=full_ages, proportion=full_proportions))

    return overall_incidence, age_distribution

if __name__ == '__main__':

    m_inci = process_model()
    d_inci = process_data()