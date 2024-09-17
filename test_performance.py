"""
Check performance
"""

import sciris as sc
import rotaABM as rabm

N = 2000
timelimit = 10
kwargs = dict(N=N, timelimit=timelimit)


def update_performance(save=False):
    sc.heading('Updating performance')
    filename = 'test_performance.json'

    T = sc.timer()
    rota = rabm.RotaABM(**kwargs)
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
    rota = rabm.RotaABM(**kwargs)
    prf = sc.profile(rota.run, follow=rota.get_weights_by_age)
    return prf


def cprofile():
    sc.heading('Running sc.cprofile')
    with sc.cprofile() as cpr:
        rota = rabm.RotaABM(**kwargs, verbose=False)
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