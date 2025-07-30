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

df = (
    df.groupby(["id", "Time", "Age", "PopulationSize"])
    .agg({"Strain": strainName, "Severity": any})
    .reset_index()
)

df = df[df["Severity"]]

age_distribution = df.groupby(["Time", "Age"]).size().reset_index(name="Counts")

age_distribution["Percentage"] = age_distribution.groupby(["Time"])["Counts"].transform(
    lambda x: x / x.sum() * 100
)

print("Age distribution by time:")
print(age_distribution.to_string(index=False))
