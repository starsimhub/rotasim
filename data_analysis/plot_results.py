"""
Compare results from the two runs
"""

import numpy as np
import sciris as sc
import matplotlib.pyplot as plt

T = sc.timer()

fns = sc.getfilelist(folder='results', pattern='rota_strains_infected_all*.csv')

dfs = sc.objdict()
rrs = sc.objdict()

for fn in fns:
    key = fn
    res = sc.objdict()
    df = sc.dataframe.read_csv(fn)
    dfs[key] = df
    res.strains = list(df.Strain.unique())
    res.x = df.CollectionTime.unique()
    npts = len(res.x)
    res.y = sc.objdict({strain:np.zeros(npts) for strain in res.strains})
    for strain in res.strains:
        for i,xi in enumerate(res.x):
            n = len(df[(df.Strain == strain) & (df.CollectionTime == xi)])
            res.y[strain][i] = n
    rrs[key] = res

sc.options(dpi=200)
fig = plt.figure()
for i,key,res in rrs.enumitems():
    plt.subplot(2,1,i+1)
    for strain in res.strains:
        plt.plot(res.x, res.y[strain], 'o-', label=strain, alpha=0.5)
    plt.title(key)
    plt.legend()

plt.show()

T.toc()
