import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import argparse
import time
import re
import numpy as np

# Assumes that number of strains is less than 10
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

def aggregate_severity(x):
    return x.any()

def draw_plot(genotype_counts, output_file, title, custom_legend_map, show=False):
    if args.area:
        fig = px.area(genotype_counts, x="Time", y="Counts", color="Strain", line_group="Strain", color_discrete_map=c)
    else:
        fig = px.line(genotype_counts, x="Time", y="Counts", color="Strain", line_group="Strain", color_discrete_map=c)
    legend_labels_added = set()

    for i in range(len(fig['data'])):
        if args.vaccolor:
            if fig['data'][i]['name'] == vaccine_strain:
                fig['data'][i]['line']['width']=5
                fig['data'][i]['line']['color']='green'
            else:
                fig['data'][i]['line']['width']=2      
                fig['data'][i]['opacity']=0.5  
        
        if custom_legend_map and fig['data'][i]['name'] in custom_legend_map:
            fig['data'][i]['name'] = custom_legend_map[fig['data'][i]['name']]

        if fig['data'][i]['name'] in legend_labels_added:
            fig['data'][i]['showlegend'] = False
        else:
            # Otherwise, add the label to the set to show its legend entry
            legend_labels_added.add(fig['data'][i]['name'])

    print("Writing to", output_file)
    fig.update_layout(
        font_size=18,
        xaxis_title="Year since simulation started",
        yaxis_title=yaxis,
        title=title
    )
    if show:
        fig.show()
    #fig.write_image(output_file, width=1980, height=1080)

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

def extract_multiple_strain_numbers(s):
     # Define the regular expression pattern
    pattern = r"G\[(\d+(?:\+\d+)*)\]|P\[(\d+(?:\+\d+)*)\]|R(\d+)|C(\d+)"
    
    # Find all matches in the input string
    matches = re.findall(pattern, s)
    
    # Initialize a dictionary to hold the extracted numbers
    extracted_numbers = {'G': [], 'P': [], 'R': [], 'C': []}
    
    for match in matches:
        if match[0]:  # G numbers
            extracted_numbers['G'].extend(match[0].split('+'))
        if match[1]:  # P numbers
            extracted_numbers['P'].extend(match[1].split('+'))
        if match[2]:  # R number
            extracted_numbers['R'].append(match[2])
        if match[3]:  # C number
            extracted_numbers['C'].append(match[3])
    
    return extracted_numbers['G'], extracted_numbers['P'], extracted_numbers['R'], extracted_numbers['C']

def get_vaccine_match(vac_strain, strain):
    if "+" in strain:
        return "Mixed"
    vac_g, vac_p, vac_r, vac_c = extract_strain_numbers(vac_strain)
    g, p, r, c = extract_strain_numbers(strain)
    if g == vac_g and p == vac_p:
        return "Homotypic"
    if g == vac_g or p == vac_p:
        return "Partial heterotypic"
    return "Complete heterotypic"
    

# Create the parser
parser = argparse.ArgumentParser(description='Process input parameters.')

# Add arguments
parser.add_argument('directory', type=str, help='Directory with the results')
parser.add_argument('experiment_suffix', type=str, help='Suffix for the experiment')
parser.add_argument('-sample_time', type=int, default=1, help='Sampling factor')
parser.add_argument('--only_severe', action='store_true', help='Plot only severe disease cases')
parser.add_argument('--sample', action='store_true', help='Plot only severe disease cases')
parser.add_argument('--vaccolor', action='store_true', help='Color the hetero/homo strains')
parser.add_argument('--incidence', action='store_true', help='Plot incidence rates')
parser.add_argument('--area', action='store_true', help='Plot as a area plot')

# Parse the arguments
args = parser.parse_args()

title_suffix_parts = [args.experiment_suffix] 
if args.only_severe:
    title_suffix_parts.append("Severe")
if args.sample:
    title_suffix_parts.append("Sample")
if args.vaccolor:
    title_suffix_parts.append("VaccineColor")
if args.incidence:
    title_suffix_parts.append("Incidence")
title_suffix = " - " + ",".join(title_suffix_parts)
file_suffix = "_".join(title_suffix_parts)

# Access the arguments
sampling_factor = args.sample_time
experiment_siffix = args.experiment_suffix
directory = args.directory

if args.sample:
    data_file = "{}/rota_strains_sampled_{}.csv".format(directory, experiment_siffix)    
else:
    data_file = "{}/rota_strains_infected_all_{}.csv".format(directory, experiment_siffix)

output_file = "{}/rota_strains_{}_{}.jpg".format(directory, experiment_siffix, file_suffix)

print("Data file:", data_file)
print("Sampling factor:", sampling_factor)
print("Output file:", output_file)

if args.vaccolor:
    vaccine_file = "{}/rota_vaccinecount_{}.csv".format(directory, experiment_siffix)
    print("Vaccine file:", vaccine_file)
    # Find the vaccine strain
    vaccine_df = pd.read_csv(vaccine_file, nrows=2)
    try:
        vaccine_strain = vaccine_df.iloc[1, 1] + 'R1C1'
    except:
        print("Error: Vaccine strain not found")
        exit(-1)

    print("Vaccine strain:", vaccine_strain)

# Load the data
df = pd.read_csv(data_file)[['id', 'CollectionTime', 'Age', 'Strain', 'Severity', "InfectionTime", "PopulationSize"]]
# Select timestamps based on sampling_factor
timestamps = sorted(df['CollectionTime'].unique())
timestamps = timestamps[::sampling_factor]
# filter the data and keep only items where CollectionTime is in timestamps
df = df[df['CollectionTime'].isin(timestamps)]

if args.incidence:    
    df['InfectionTime'] = df['InfectionTime'].round(1)
    df['Time'] = df['InfectionTime']
    df['PopulationSize'] = df.groupby(['Time'])['PopulationSize'].transform('min')
    df['PopulationSize'] = df.groupby(['id', 'Strain', 'Time'])['PopulationSize'].transform('min')
    yaxis = "Incidence"
else:
    df['Time'] = df['CollectionTime']    
    yaxis = "Prevalence"

if args.only_severe:
    df = df.groupby(['id', 'Time', 'Age', 'PopulationSize']).agg({'Strain': strainName, 'Severity': any}).reset_index()
    df = df[df['Severity']]
else:
    df = df.groupby(['id', 'Time', 'Age', 'PopulationSize']).agg({'Strain': strainName}).reset_index()

if args.vaccolor:
    df['VaccineEff'] = df['Strain'].apply(lambda x: get_vaccine_match(vaccine_strain, x))

c = {}

if args.vaccolor:
    # Initialize counts and color map
    count_partial_hetero = 0
    count_complete_hetero = 0
    color_map = {}
    custom_legend_map = {}

    # Extract vaccine Gtype and Ptype
    vac_g,vac_p,vac_r,vac_c = extract_strain_numbers(vaccine_strain)

    # Calculate counts and build the initial color map
    for strain in df["Strain"].unique():        
        if strain == vaccine_strain:
            color_map[strain] = "green"
            custom_legend_map[strain] = "Homotypic"
        else:
            if "+" in strain:
                g, p, r, c = extract_multiple_strain_numbers(strain)
                print(g, p, r, c)
                if vac_g in g and vac_p in p:
                    color_map[strain] = "green"
                    custom_legend_map[strain] = "Homotypic"
                if vac_g in g or vac_p in p:
                    color_map[strain] = "blue"
                    custom_legend_map[strain] = "Partial heterotypic"
                else:
                    color_map[strain] = "red"
                    custom_legend_map[strain] = "Complete heterotypic"
            else:
                g, p, r, c = extract_strain_numbers(strain)
                if vac_g == g or vac_p == p:
                    color_map[strain] = "blue"
                    custom_legend_map[strain] = "Partial heterotypic"
                else:
                    color_map[strain] = "red"
                    custom_legend_map[strain] = "Complete heterotypic"
    c = color_map
    print(color_map)
else:
    custom_legend_map = None
    c = dict(zip(df["Strain"].unique(), px.colors.qualitative.Light24))

genotype_counts = df.groupby(['Time', 'Strain', 'PopulationSize']).size().reset_index(name='Counts')
genotype_counts['Counts'] = genotype_counts['Counts'] / genotype_counts['PopulationSize'] * 10000
draw_plot(genotype_counts, output_file, "All Infected"+title_suffix, custom_legend_map, True)

for age in df['Age'].unique():
    genotype_counts = df[df['Age'] == age].groupby(['Time', 'Strain', 'PopulationSize']).size().reset_index(name='Counts')
    genotype_counts['Counts'] = genotype_counts['Counts'] / genotype_counts['PopulationSize'] * 10000
    output_file_age = output_file.replace(".jpg", "_age_" + age.replace('-','_') + ".jpg")
    draw_plot(genotype_counts, output_file_age, "Infected " + age + title_suffix, custom_legend_map)

genotype_counts = df[df['Age'] != "60+"].groupby(['Time', 'Strain', 'PopulationSize']).size().reset_index(name='Counts')
genotype_counts['Counts'] = genotype_counts['Counts'] / genotype_counts['PopulationSize'] * 10000
output_file_age = output_file.replace(".jpg", "_below5_" + ".jpg")
draw_plot(genotype_counts, output_file_age, "Infected below 5 years" + title_suffix, custom_legend_map, True)

genotype_counts = df[df['Age'].isin(['0-2', '2-4', '4-6', '6-12', '12-24'])].groupby(['Time', 'Strain', 'PopulationSize']).size().reset_index(name='Counts')
genotype_counts['Counts'] = genotype_counts['Counts'] / genotype_counts['PopulationSize'] * 10000
output_file_age = output_file.replace(".jpg", "_below2_" + ".jpg")
draw_plot(genotype_counts, output_file_age, "Infected below 2 years" + title_suffix, custom_legend_map, True)

if args.vaccolor:
    genotype_counts = df.groupby(['Time', 'VaccineEff', 'PopulationSize']).size().reset_index(name='Counts')
    genotype_counts['Counts'] = genotype_counts['Counts'] / genotype_counts['PopulationSize'] *10000
    if args.area:
        fig = px.area(genotype_counts, x="Time", y="Counts", color="VaccineEff", line_group="VaccineEff")
    else:
        fig = px.line(genotype_counts, x="Time", y="Counts", color="VaccineEff", line_group="VaccineEff")
    output_file_age = output_file.replace(".jpg", "_vaccine_" + ".jpg")
    fig.update_layout(
        font_size=18,
        xaxis_title="Year since simulation started",
        yaxis_title=yaxis,
        title="Vaccine Efficacy" + title_suffix
    )
    fig.show()
    #fig.write_image(output_file_age, width=1980, height=1080)

    genotype_counts = df[df['Age'] != "60+"].groupby(['Time', 'VaccineEff', 'PopulationSize']).size().reset_index(name='Counts')
    genotype_counts['Counts'] = genotype_counts['Counts'] / genotype_counts['PopulationSize'] * 10000
    if args.area:
        fig = px.area(genotype_counts, x="Time", y="Counts", color="VaccineEff", line_group="VaccineEff")
    else:
        fig = px.line(genotype_counts, x="Time", y="Counts", color="VaccineEff", line_group="VaccineEff")
    output_file_age = output_file.replace(".jpg", "_vaccine_below2" + ".jpg")
    fig.update_layout(
        font_size=18,
        xaxis_title="Year since simulation started",
        yaxis_title=yaxis,
        title="Vaccine Efficacy Below 5" + title_suffix
    )
    fig.show()
    #fig.write_image(output_file_age, width=1980, height=1080)

    genotype_counts = df[df['Age'].isin(['0-2', '2-4', '4-6', '6-12', '12-24'])].groupby(['Time', 'VaccineEff', 'PopulationSize']).size().reset_index(name='Counts')
    genotype_counts['Counts'] = genotype_counts['Counts'] / genotype_counts['PopulationSize'] * 10000
    if args.area:
        fig = px.area(genotype_counts, x="Time", y="Counts", color="VaccineEff", line_group="VaccineEff")
    else:
        fig = px.line(genotype_counts, x="Time", y="Counts", color="VaccineEff", line_group="VaccineEff")
    output_file_age = output_file.replace(".jpg", "_vaccine_below5" + ".jpg")
    fig.update_layout(
        font_size=18,
        xaxis_title="Year since simulation started",
        yaxis_title=yaxis,
        title="Vaccine Efficacy below 2" + title_suffix
    )
    fig.show()
    #fig.write_image(output_file_age, width=1980, height=1080)