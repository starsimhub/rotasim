import argparse
import re
import pandas as pd


def strainName(strainIn):
    s_types = [set() for i in range(4)]
    for strain in strainIn:
        pattern = r"G(\d+)P(\d+)A(\d+)B(\d+)"
        match = re.match(pattern, strain)
        for i, n in enumerate(match.groups()):
            s_types[i].add(n)
    newLabel = ""
    for i, l in enumerate("GPRC"):
        if len(s_types[i]) > 1:
            newLabel += l + "[" + "+".join(sorted(s_types[i])) + "]"
        else:
            newLabel += l + list(s_types[i])[0]
    return newLabel


parser = argparse.ArgumentParser(description="Process input parameters.")

parser.add_argument("datafile", type=str, help="Strain count file")
parser.add_argument(
    "--below5", action="store_true", help="use only incidences with age below 5"
)
parser.add_argument("--severe", action="store_true", help="use only severe incidences")
args = parser.parse_args()

datafile = args.datafile

print("Processing file: {}".format(datafile))
df = pd.read_csv(datafile)[
    [
        "id",
        "CollectionTime",
        "Age",
        "Strain",
        "Severity",
        "InfectionTime",
        "PopulationSize",
    ]
]

df["InfectionTime"] = df["InfectionTime"].astype(int)
df["Time"] = df["InfectionTime"]
df["PopulationSize"] = df.groupby(["Time"])["PopulationSize"].transform("min")
df["PopulationSize"] = df.groupby(["id", "Strain", "Time"])["PopulationSize"].transform(
    "min"
)

if args.severe:
    df = df[df["Severity"]]

if args.below5:
    df = df[df["Age"] != "60+"]

genotype_counts = df.groupby(["Time", "Strain"]).size().reset_index(name="Counts")

# Calculate number of unique strains per time period
unique_strains_per_time = (
    genotype_counts.groupby("Time")["Strain"].nunique().reset_index()
)
unique_strains_per_time.rename(columns={"Strain": "Unique_Strains"}, inplace=True)

print("Number of unique strains by time:")
print(unique_strains_per_time.to_string(index=False))
