"""
Check results against baseline values.

NB: the two tests could be combined into one, but are left separate for clarity.
"""

import sciris as sc
import rotasim as rs

N = 2_000
timelimit = 10

def test_default(make=False):
    sc.heading('Testing default parameters')
    filename = 'test_events_default.json'
    
    # Generate new baseline
    if make:
        with sc.timer() as T:
            rota = rs.Sim(N=N, timelimit=timelimit)
            events = rota.run()
        sc.savejson(filename, events)
        
    # Check old baseline
    else:
        with sc.timer() as T:
            rota = rs.Sim(N=N, timelimit=timelimit, verbose=True)
            events = rota.run()
        saved = sc.objdict(sc.loadjson(filename))
        assert events == saved, 'Events do not match for default simulation'
        print(f'Defaults matched:\n{events}')
    
    return


def test_alt(make=False):
    sc.heading('Testing alternate parameters')
    filename = 'test_events_alt.json'
    inputs = dict(
        N = N,
        timelimit = timelimit,
        immunity_hypothesis = 2,
        reassortment_rate = 0.2,
        fitness_hypothesis = 2,
        vaccine_hypothesis = 2,
        waning_hypothesis = 2,
        initial_immunity = 0.1,
        ve_i_to_ve_s_ratio = 0.5,
        experiment_number = 2,
    )
    
    # Generate new baseline
    if make:
        rota = rs.Sim(**inputs)
        events = rota.run()
        sc.savejson(filename, events)
        
    # Check old baseline
    else:
        rota = rs.Sim(**inputs)
        events = rota.run()
        saved = sc.objdict(sc.loadjson(filename))
        assert events == saved, 'Events do not match for alternate parameters simulation'
        print(f'Alternate parameters matched:\n{events}')
        
    return


if __name__ == '__main__':
    make = False # Set to True to regenerate results
    benchmark = False # Set to True to redo the performance results
    test_default(make=make, benchmark=benchmark)
    test_alt(make=make)