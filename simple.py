import sciris as sc
import rotaABM as rabm

with sc.timer():
    rota = rabm.Rota()
    events = rota.run()
    print(events)