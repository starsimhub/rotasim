import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import argparse
import time
import re
import numpy as np

parser = argparse.ArgumentParser(description='Process input parameters.')
parser.add_argument('directory', type=str, help='Directory with the results')
parser.add_argument('--sample', action='store_true', help='Plot only severe disease cases')
parser.add_argument('experiment_suffix', type=str, help='Suffix for the experiment')
parser.add_argument('--skip_multi', action='store_true', help='Skip hosts with multiple infections')

args = parser.parse_args()
experiment_siffix = args.experiment_suffix
directory = args.directory
vaccine_time = 20
post_vac_time = 15

if args.sample:
    data_file = "{}/rota_strains_sampled_{}.csv".format(directory, experiment_siffix)    
else:
    data_file = "{}/rota_strains_infected_all_{}.csv".format(directory, experiment_siffix)
vaccine_file = "{}/rota_vaccinecount_{}.csv".format(directory, experiment_siffix)
population_file = "{}/rota_agecount_{}.csv".format(directory, experiment_siffix)

print("Computing the population size")
# Read the population file
pop_df = pd.read_csv(population_file)
# Avearge the population size over the time integer
pop_df['time'] = pop_df['time'].astype(int)
# Avearge the population size over the time integer
pop_df_grouped = pop_df.groupby('time').mean().reset_index()
# Create a new column for the total population
pop_df_grouped['total'] = pop_df_grouped['0-2'] + pop_df_grouped['2-4'] + pop_df_grouped['4-6'] + pop_df_grouped['6-12'] + pop_df_grouped['12-24'] + pop_df_grouped['24-36'] + pop_df_grouped['36-48'] + pop_df_grouped['48-60'] + pop_df_grouped['60+']
print("Finished computing the population size")

# Find the vaccine strain
vaccine_df = pd.read_csv(vaccine_file, nrows=2)
try:
    vaccine_strain = vaccine_df.iloc[1, 1] + 'R1C1'
except:
    print("Error: Vaccine strain not found")
    exit(-1)
print("Vaccine strain:", vaccine_strain)

df = pd.read_csv(data_file)[['id', 'CollectionTime', 'Age', 'Strain', 'Severity', "InfectionTime", "PopulationSize"]]

# Set the population to the minimum value during the same time stamp
# We want to avoid haveing multiple population sizes in the time
df['Time'] = df['InfectionTime'].round(1)
""" df['PopulationSize'] = df.groupby(['Time'])['PopulationSize'].transform('min') """
""" df['PopulationSize'] = df.groupby(['id', 'Strain', 'Time', 'Age']).transform('min') """
df = df.groupby(['id', 'Strain', 'Time', 'Severity']).agg({'Age':'first'}).reset_index()

def strainName(strainIn):
    s_types = [set() for i in range(4)]
    for strain in strainIn:
        pattern = r"G(\d+)P(\d+)A(\d+)B(\d+)"
        match = re.match(pattern, strain)
        for i,n in enumerate(match.groups()):
            s_types[i].add(n)
    newLabel = ""
    for i, l in enumerate("GPRC"):
        if len(s_types[i]) > 1:
            newLabel += l + "[" + "+".join(sorted(s_types[i])) + "]"
        else:
            newLabel += l + list(s_types[i])[0]
    return newLabel

def check_unique_severity(series):
    if series.nunique() == 1:
        return series.iloc[0]  # Return the unique value
    else:
        return None 
    
df = df.groupby(['id', 'Time', 'Strain', 'Age']).agg({'Severity': check_unique_severity}).reset_index()

if args.skip_multi:
    df = df.groupby(['id', 'Time', 'Age', 'PopulationSize']).agg({'Strain': strainName}).reset_index()
    df = df[~df['Strain'].str.contains("+")]


all_ages = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60+']
below_5 = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60']
below_2 = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48']
below_6m = ['0-2', '2-4', '4-6']

def compute_pop(time, age_groups, pop_df_in):
    population = 0 
    for age in age_groups:        
        population += pop_df_in.loc[(time, age)]
    return population

def get_severity_below_age(df, l, u, age_groups):
    severe_incidence = df[(df['Time'] >= l) & (df['Time'] < u) & (df['Severity'] == True) & (df['Age'].isin(age_groups))].copy()
    
    # Initialize a dictionary to hold population values for each time
    time_population = {}
    # Iterate through each unique Time value in severe_incidence
    for time in set([int (i) for i in severe_incidence['Time'].unique()]):
        # Compute the population for the current time and age groups
        population = compute_pop(time, age_groups, pop_df_grouped)
        # Store the computed population in the dictionary
        time_population[time] = population

    severe_incidence['contribution'] = severe_incidence['Time'].apply(lambda x: 1 / time_population[int(x)])
    rate = sum(severe_incidence['contribution'])
    return rate

def compute_vaccine_efficacy(age_group):
    print("=============== Computing for age group:", age_group, "===============")
    pre_vac_rate = get_severity_below_age(df, vaccine_time - 6, vaccine_time-1, age_group)
    post_vac_rate = get_severity_below_age(df, vaccine_time + post_vac_time, vaccine_time + post_vac_time + 5, age_group)
    post_vac_rate2 = get_severity_below_age(df, vaccine_time + post_vac_time + 5, vaccine_time + post_vac_time + 10, age_group)

    print("Pre-vaccine incidence rate:", pre_vac_rate)
    print("Post-vaccine incidence rate:", post_vac_rate)
    print("Post-vaccine incidence rate 2:", post_vac_rate2)

    print("Vaccine efficacy 1:", (1 - (post_vac_rate/pre_vac_rate))*100)
    print("Vaccine efficacy 2:", (1 - (post_vac_rate2/pre_vac_rate))*100)

compute_vaccine_efficacy(all_ages)
compute_vaccine_efficacy(below_5)
compute_vaccine_efficacy(below_2)

def extract_strain_numbers(s):
    # Define the regular expression pattern
    pattern = r"G(\d+)P(\d+)R(\d+)C(\d+)"
    
    # Match the pattern against the input string
    match = re.match(pattern, s)
    
    # If a match is found, extract and return the G and P numbers
    if match:
        g_number, p_number, a_number, b_number = match.groups()
        return g_number, p_number, a_number, b_number
    else:
        print("Error: No match found for - [{}]".format(s))
        exit(-1)

def extract_strain_numbers2(s):
    # Define the regular expression pattern
    pattern = r"G(\d+)P(\d+)A(\d+)B(\d+)"
    
    # Match the pattern against the input string
    match = re.match(pattern, s)
    
    # If a match is found, extract and return the G and P numbers
    if match:
        g_number, p_number, a_number, b_number = match.groups()
        return g_number, p_number, a_number, b_number
    else:
        print("Error: No match found for - [{}]".format(s))
        exit(-1)

def get_vaccine_match(vac_strain, strain):
    vac_g, vac_p, vac_r, vac_c = extract_strain_numbers(vac_strain)
    g, p, r, c = extract_strain_numbers2(strain)
    if g == vac_g and p == vac_p:
        return "Homotypic"
    if g == vac_g or p == vac_p:
        return "Partial heterotypic"
    return "Complete heterotypic"

df['VaccineEff'] = df['Strain'].apply(lambda x: get_vaccine_match(vaccine_strain, x))

def get_severity_below_age_with_vac_group(df, l, u, age_groups, vac_group):
    severe_incidence = df[(df['Time'] >= l) & (df['Time'] < u) & (df['Severity'] == True) & (df['Age'].isin(age_groups)) & (df['VaccineEff'] == vac_group)].copy()

    # Initialize a dictionary to hold population values for each time
    time_population = {}
    # Iterate through each unique Time value in severe_incidence
    for time in set([int (i) for i in severe_incidence['Time'].unique()]):
        # Compute the population for the current time and age groups
        population = compute_pop(time, age_groups, pop_df_grouped)
        # Store the computed population in the dictionary
        time_population[time] = population

    severe_incidence['contribution'] = severe_incidence['Time'].apply(lambda x: 1 / time_population[int(x)])
    rate = sum(severe_incidence['contribution'])
    return rate

def compute_vaccine_efficacy_with_vec_group(age_group, vac_group):
    print("=============== Computing for age group:", age_group, "with " + vac_group + " strains ===============")
    pre_vac_rate = get_severity_below_age_with_vac_group(df, vaccine_time - 6, vaccine_time - 1, age_group, vac_group)
    post_vac_rate = get_severity_below_age_with_vac_group(df, vaccine_time + post_vac_time, vaccine_time + post_vac_time + 5, age_group, vac_group)
    post_vac_rate2 = get_severity_below_age_with_vac_group(df, vaccine_time + post_vac_time + 5, vaccine_time + post_vac_time + 10, age_group, vac_group)

    print("Pre-vaccine incidence rate:", pre_vac_rate)
    print("Post-vaccine incidence rate:", post_vac_rate)
    print("Post-vaccine incidence rate 2:", post_vac_rate2)

    print("Vaccine efficacy 1:", (1 - (post_vac_rate/pre_vac_rate))*100)
    print("Vaccine efficacy 2:", (1 - (post_vac_rate2/pre_vac_rate))*100)

compute_vaccine_efficacy_with_vec_group(all_ages, "Homotypic")
compute_vaccine_efficacy_with_vec_group(below_5, "Homotypic")
compute_vaccine_efficacy_with_vec_group(below_2, "Homotypic")
compute_vaccine_efficacy_with_vec_group(below_6m, "Homotypic")

compute_vaccine_efficacy_with_vec_group(all_ages, "Partial heterotypic")
compute_vaccine_efficacy_with_vec_group(below_5, "Partial heterotypic")
compute_vaccine_efficacy_with_vec_group(below_2, "Partial heterotypic")
compute_vaccine_efficacy_with_vec_group(below_6m, "Partial heterotypic")

compute_vaccine_efficacy_with_vec_group(all_ages, "Complete heterotypic")
compute_vaccine_efficacy_with_vec_group(below_5, "Complete heterotypic")
compute_vaccine_efficacy_with_vec_group(below_2, "Complete heterotypic")
compute_vaccine_efficacy_with_vec_group(below_6m, "Complete heterotypic")

for age_group in [all_ages, below_5, below_2, below_6m]:
    df2 = df[df['Age'].isin(age_group)]
    genotype_counts = df2.groupby(['VaccineEff', 'Time']).size().reset_index(name='Counts')
    """ genotype_counts['contribution'] = genotype_counts['Time'].apply(lambda x: 1 / genotype_counts[int(x)]) """

    # Step 1: Calculate total counts for each Time
    total_counts_by_time = genotype_counts.groupby('Time')['Counts'].sum().reset_index(name='TotalCounts')

    # Step 2: Merge the total counts with the original genotype_counts DataFrame
    genotype_counts_with_total = genotype_counts.merge(total_counts_by_time, on='Time')

    # Step 3: Calculate the percentage of each count relative to the total counts for its respective Time
    genotype_counts_with_total['Percentage'] = (genotype_counts_with_total['Counts'] / genotype_counts_with_total['TotalCounts']) * 100
    

    fig = px.area(genotype_counts_with_total, x="Time", y="Percentage", color="VaccineEff", line_group="VaccineEff")
    """ output_file_age = output_file.replace(".jpg", "_vaccine_" + ".jpg") """
    fig.update_layout(
        font_size=18,
        xaxis_title="Year since simulation started",
        yaxis_title="Percentage",
        title="Percentage Based on Vaccine Efficacy: " + experiment_siffix + " " + str(age_group),
    )
    fig.show()