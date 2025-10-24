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
        incidence_sheet = 'Matlab_incidence'
    if age_dist_sheet is None:
        age_dist_sheet = 'Matlab_agedistribution'

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

    # Subset to years 1-9 for now
    initial8 = dat[(dat['CollectionTime'] < 9) & (dat['CollectionTime'] > 1)]

    if verbose: print(initial8['Strain'].value_counts())
    initial8['Strain3'] = 'Other'
    initial8.loc[initial8['Strain'] == 'G1P8A1B1', 'Strain3'] = 'G1P8'
    initial8.loc[initial8['Strain'] == 'G2P4A1B1', 'Strain3'] = 'G2P4'
    initial8.loc[initial8['Strain'] == 'G9P8A1B1', 'Strain3'] = 'G9P8'

    GenoDist = initial8['Strain3'].value_counts().reset_index()
    GenoDist.columns = ['Strain', 'Frequency']
    total = GenoDist['Frequency'].sum()
    GenoDist['Proportion'] = GenoDist['Frequency'] / total

    # Working out the case age distribution
    # First making new age bins
    initial8['AgeCat'] = np.nan
    initial8.loc[initial8['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
    initial8.loc[initial8['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
    initial8.loc[initial8['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
    initial8.loc[initial8['Age'] == '60+', 'AgeCat'] = '>=5 y'

    # Convert continuous time to integer years for proper annual aggregation
    initial8['Year'] = np.floor(initial8['CollectionTime']).astype(int)

    # Clinical data counts symptomatic/reported cases
    # Count first few infections per person (these are typically symptomatic)
    # Later infections are usually asymptomatic due to acquired immunity
    # Add infection number per person (lifetime)
    initial8 = initial8.sort_values(['id', 'CollectionTime'])
    initial8['infection_number'] = initial8.groupby('id').cumcount() + 1

    # Count only first 3 infections per person (symptomatic threshold)
    # Use uniform threshold across all ages - let rel_beta affect age distribution naturally
    initial8_symptomatic = initial8[initial8['infection_number'] <= 3].copy()

    # Then take first infection per agent per year (to avoid double-counting within same year)
    initial8_first = initial8_symptomatic.groupby(['id', 'Year', 'AgeCat']).first().reset_index()

    # Now, getting cases by age bin and YEAR
    # Count unique agents per year to get annual incidence
    cases_summary = initial8_first.groupby(['AgeCat', 'Year']).agg(Cases_age=('id', 'nunique')).reset_index()
    if verbose: print(cases_summary.head())

    # Now, getting total population by time point (year)
    # It's already pooled, so just need one observation per year and age bin
    pop_pooled = initial8_first.groupby(['Year', 'AgeCat']).apply(lambda x: x.head(1)).reset_index(drop=True)
    pop_pooled2 = pop_pooled[['Year', 'AgeCat', 'PopulationSize']]

    # Multiplying by fraction of the population in each age bin
    pop_pooled2['Pop_Age'] = np.nan
    pop_pooled2.loc[pop_pooled2['AgeCat'] == '<1 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.025
    pop_pooled2.loc[pop_pooled2['AgeCat'] == '1-2 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.025
    pop_pooled2.loc[pop_pooled2['AgeCat'] == '2-5 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.075
    pop_pooled2.loc[pop_pooled2['AgeCat'] == '>=5 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.875

    # Merge incidence with the age and time appropriate denominator to get the ratio
    AgeIncidence = pd.merge(cases_summary, pop_pooled2, on=['AgeCat', 'Year'])
    AgeIncidence['IR_100k'] = (AgeIncidence['Cases_age'] / AgeIncidence['Pop_Age']) * 100000 / FUDGE

    # Get an average to calibrate to average pre-vaccine period
    Inci_dist = AgeIncidence.groupby('AgeCat').agg(meanIR=('IR_100k', 'mean')).reset_index()

    # Standardize data format
    age_mapping = {
        '<1 y': 0,
        '1-2 y': 1,
        '2-5 y': 2,
        '>=5 y': 5,
    }
    ages = Inci_dist['AgeCat'].replace(age_mapping)
    df = sc.dataframe(dict(ages=ages, inci=Inci_dist['meanIR']))
    df = df.sort_values(by='ages').reset_index(drop=True)

    # Calculate population-weighted overall incidence
    # Data calculation: sum(total_cases) / total_population * 100000
    # We need to weight each age group by its population fraction
    pop_fractions = [0.025, 0.025, 0.075, 0.875]  # <1y, 1-2y, 2-5y, >=5y

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