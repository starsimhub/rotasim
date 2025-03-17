"""
Check performance
"""

import sciris as sc
import rotasim as rs

N = 5_000
timelimit = 30
kwargs = dict(N=N, timelimit=timelimit)


def update_performance(save=False):
    sc.heading('Updating performance')
    filename = 'test_performance.json'

    T = sc.timer()
    rota = rs.Sim(**kwargs)
    rota.run()
    T.toc()
    string = f'{T.elapsed:0.2f}'
    data = dict(time=string)
    if save:
        sc.savejson(filename, data)
    print(data)
        
    return


def profile():
    sc.heading('Running sc.profile')
    rota = rs.Sim(**kwargs)
    prf = sc.profile(rota.run, follow=rota.integrate)
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