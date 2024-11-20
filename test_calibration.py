import sciris as sc
import rotaABM as rabm
import calibration.process_incidence as cpi

rota = rabm.RotaABM(N=10_000, timelimit=2)
events = rota.run()
cols = rota.results.columns
res = rota.results.infected_all
df = sc.dataframe(data=res, columns=cols)

out = cpi.process_model(df)
print(out)