import starsim as ss
import rotasim as rs

class StrainStats(ss.Analyzer):
    """
    Analyzer to track the proportions of different strains in the population
    """
    def __init__(self):
        super().__init__()
        return


    def init_results(self):
        super().init_results()

        for strain in self.sim.connectors.rota.pars.segment_combinations:
            self.results += ss.Result(f'{strain} proportion', dtype=float, scale=False, module=self.name, shape=self.timevec.shape, timevec=self.timevec)
            self.results += ss.Result(f'{strain} count', dtype=float, scale=True,
                                      module=self.name, shape=self.timevec.shape, timevec=self.timevec)
        return


    def step(self):
        sim = self.sim
        res = self.results
        strain_count = sim.connectors.rota.strain_count

        # get the list of strains that appeared in the entire simulation

        total_count = sum(strain_count.values())
        if total_count:
            for strain in strain_count:
                # Calculate the proportion of the strain in the population
                res[f'{strain} proportion'][sim.ti] = strain_count[strain] / total_count
                res[f'{strain} count'][sim.ti] = strain_count[strain]

        return

    def to_df(self):
        df = self.results.to_df()

        # get the list of extra timevec column indexes and drop all but the first
        indexes_to_drop = df.columns.get_indexer_for(['timevec'])
        df.drop(columns=df.columns[indexes_to_drop[1:]], inplace=True)

        return df