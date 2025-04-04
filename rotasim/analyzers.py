import starsim as ss
import numpy as np

class StrainStats(ss.Analyzer):
    """
    Analyzer to track the proportions of different strains in the population
    """
    def __init__(self):
        super().__init__()
        # self.results = {}
        return


    def init_results(self):
        super().init_results()

        for strain in self.sim.connectors.rota.pars.segment_combinations:
            self.results += ss.Result(f'({', '.join(map(str, strain))}) proportion', dtype=float, scale=False, module=self.name, shape=self.timevec.shape, timevec=self.timevec)
            self.results += ss.Result(f'({', '.join(map(str, strain))}) count', dtype=float, scale=False,
                                      module=self.name, shape=self.timevec.shape, timevec=self.timevec)
        return


    def step(self):
        sim = self.sim
        res = self.results
        strain_count = sim.connectors.rota.strain_count

        # get the list of strains that appeared in the entire simulation

        for strain in strain_count:
            # Calculate the proportion of the strain in the population
            res[f'{strain} proportion'][sim.ti] = strain_count[strain] / sum(strain_count.values())
            res[f'{strain} count'][sim.ti] = strain_count[strain]

        return