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


if __name__ == '__main__':
    update_performance()