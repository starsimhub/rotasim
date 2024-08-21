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
parser.add_argument('experiment_suffix', type=str, help='Suffix for the experiment')
parser.add_argument('--only_severe', action='store_true', help='Only severe infections are considered')

args = parser.parse_args()
experiment_siffix = args.experiment_suffix
directory = args.directory
vaccine_time = 20
post_vac_time = 10

data_file = "{}/rota_strains_infected_all_{}.csv".format(directory, experiment_siffix)

df = pd.read_csv(data_file)[['id', 'CollectionTime', 'Age', 'Strain', 'Severity', "InfectionTime", "PopulationSize"]]
df = df[((df['CollectionTime'] >= vaccine_time - 5) & (df['CollectionTime'] < vaccine_time)) | ((df['CollectionTime'] >= vaccine_time + post_vac_time) & (df['CollectionTime'] < vaccine_time + post_vac_time + 10))]

df['Time'] = df['CollectionTime'].round(1)

#df['Time'] = df['InfectionTime'].round(1)
""" df['PopulationSize'] = df.groupby(['Time'])['PopulationSize'].transform('min') """
""" df['PopulationSize'] = df.groupby(['id', 'Strain', 'Time', 'Age']).transform('min') """
#df = df.groupby(['id', 'Strain', 'Time', 'Severity']).agg({'Age':'first'}).reset_index()

def strainName(strainIn):
    if len(strainIn) > 1:
        return "CoInfection"
    else:
        return strainIn.iloc[0].split("A")[0]
    
df = df.groupby(['id', 'Time', 'Age']).agg({'Strain': strainName, 'Severity': lambda x: x.any()}).reset_index()

strains  = list(df['Strain'].unique())
# map strains to values in px.colors.qualitative.Light24
color_palette = px.colors.qualitative.Plotly+px.colors.qualitative.T10
strain_color_map = {strain: color_palette[i % len(color_palette)] for i, strain in enumerate(strains)}
print(strain_color_map)

if args.only_severe:
    df = df[df['Severity']]

def plot_pie_chart(df, l, u, age_group_name, age_group):
    cases = df[(df['Time'] >= l) & (df['Time'] < u) & (df['Age'].isin(age_group))]
    cases = cases.groupby('Strain').size().reset_index(name='Counts')
    """ cases = cases.groupby('Strain').agg({'Counts': 'sum'}).reset_index() """

    # draw pie chart
    fig = px.pie(cases, names='Strain', values='Counts', color='Strain', color_discrete_map=strain_color_map)
    fig.update_layout(
        font_size=28,
        title="Strain Diversity Between {} and {} Years. {} {}".format(l, u, experiment_siffix, age_group_name),
        width=1920,
        height=1080
    )

    output_file_name = "{}/rota_strain_diversity_pie_chart_{}_{}_{}_{}.jpg".format(directory, experiment_siffix, l, u, age_group_name)
    #fig.write_image(output_file_name)
    fig.show()

age_groups = {
    "below_5": ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60']
}

for (group_name, age) in age_groups.items():
    plot_pie_chart(df, vaccine_time - 5, vaccine_time, group_name, age)
    plot_pie_chart(df, vaccine_time + post_vac_time, vaccine_time + post_vac_time + 5, group_name, age)
    plot_pie_chart(df, vaccine_time + post_vac_time + 5, vaccine_time + post_vac_time + 10, group_name, age)
print(df)