"""
Check performance
"""

import sciris as sc
import starsim as ss
import rotasim as rs

N = 2_000
timelimit = 2
kwargs = dict(n_agents=N, start='2020-01-01', stop=f'{2020+timelimit}-01-01', dt=ss.days(1))


def update_performance(save=False):
    sc.heading('Updating performance')
    filename = 'test_performance.json'

    T = sc.timer()
    rota = rs.Sim(**kwargs)
    rota.run()
    T.toc()
    time = round(T.elapsed, 2)
    benchmark = round(sc.benchmark()['numpy'], 2)
    data = dict(time=time, sc_benchmark=benchmark)
    if save:
        sc.savejson(filename, data)
    print(data)
        
    return


def profile():
    sc.heading('Running sc.profile')
    rota = rs.Sim(**kwargs)
    rota.init()
    prf = sc.profile(rota.run)
    return prf


def cprofile():
    sc.heading('Running sc.cprofile')
    with sc.cprofile() as cpr:
        rota = rs.Sim(**kwargs, verbose=False)
        rota.run()
    return cpr


if __name__ == '__main__':
    save = 0
    do_profile = 1
    if save or not do_profile:
        update_performance(save=save)
    else:
        prf = profile()
        cpr = cprofile()
