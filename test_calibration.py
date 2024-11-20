import sciris as sc
import rotaABM as rabm
import calibration.process_incidence as cpi

rota = rabm.RotaABM(N=10_000, timelimit=2)
events = rota.run()
cols = rota.results.columns
res = rota.results.infected_all
df = sc.dataframe(data=res, columns=cols)

df2 = sc.dataframe.read_csv('results/rota_strains_infected_all_1_0.1_1_1_1_0_0.5_1.csv')

out = cpi.process_model(df2)
print(out)