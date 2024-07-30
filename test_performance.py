"""
Check performance
"""

import sciris as sc
import rotaABM as rota


def update_performance():
    sc.heading('Updating performance')
    filename = 'test_performance.json'

    T = sc.timer()
    rota.main()
    T.toc()
    sc.savejson(filename, dict(time=f'{T.elapsed:0.2f}'))
        
    return


def profile():
    sc.heading('Running sc.profile')
    prf = sc.profile(rota.main)
    return prf


def cprofile():
    sc.heading('Running sc.cprofile')
    with sc.cprofile() as cpr:
        rota.main()
    return cpr


if __name__ == '__main__':
    do_profile = 0
    if not do_profile:
        update_performance()
    else:
        prf = profile()
        cpr = cprofile()