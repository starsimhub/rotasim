import sciris as sc
import rotasim as rs
import calibration.process_incidence as cpi

def test_calibration():
    # Test running sims
    sc.heading('Test Rotasim calibration')
    sim = rs.Sim(N=10_000, timelimit=7)
    events = sim.run()
    out = cpi.process_model(sim.df)
    print(out)