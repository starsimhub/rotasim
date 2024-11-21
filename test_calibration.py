import sciris as sc
import rotaABM as rabm
import calibration.process_incidence as cpi

sim = rabm.RotaABM(N=100_000, timelimit=8)
events = sim.run()
out = cpi.process_model(sim.df)
print(out)