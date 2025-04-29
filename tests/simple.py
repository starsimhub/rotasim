import sciris as sc
import rotasim as rs

def run_sims():
    for i in range(5):
        sim = rs.Sim(rand_seed=i, n_agents=2000, timelimit=10, verbose=False)
        sim.run()
        events = sim.connectors['rota'].event_dict
        print(events)


    # sim = rs.Sim(rand_seed=3, n_agents=2000, timelimit=10, verbose=False)
    # sim.run()
    # events = sim.connectors['rota'].event_dict
    # print(events)

with sc.timer():
    # run_sims()
    sc.profile(run_sims, [rs.Rota.step, rs.Rota.contact_event_m, rs.Rota.contact_event])

