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
        self.results += ss.Result(
            "time",
            dtype=float,
            scale=False,
            module=self.name,
            shape=self.timevec.shape,
            timevec=self.timevec,
        )
        for strain in self.sim.connectors.rota.pars.segment_combinations:
            self.results += ss.Result(
                f"{strain} proportion",
                dtype=float,
                scale=False,
                module=self.name,
                shape=self.timevec.shape,
                timevec=self.timevec,
            )
            self.results += ss.Result(
                f"{strain} count",
                dtype=float,
                scale=False,
                module=self.name,
                shape=self.timevec.shape,
                timevec=self.timevec,
            )

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
                res[f"{strain} proportion"][sim.ti] = strain_count[strain] / total_count
                res[f"{strain} count"][sim.ti] = strain_count[strain]
                res["time"][sim.ti] = sim.t.abstvec[sim.ti]
        return

    def to_df(self):
        df = self.results.to_df()

        # get the list of extra timevec column indexes and drop all but the first
        indexes_to_drop = df.columns.get_indexer_for(["timevec"])
        df.drop(columns=df.columns[indexes_to_drop[1:]], inplace=True)

        return df

    def get_strain_proportions(self, strain):
        """
        Returns a DataFrame with the strain proportions over time.
        """
        df = self.to_df()

        # Round time to integers for grouping
        df["time_rounded"] = df["time"].round().astype(int)

        # Group by rounded time and get mean proportion for the strain
        strain_proportions = (
            df.groupby("time_rounded")[f"{strain} proportion"].mean().reset_index()
        )
        strain_proportions.rename(columns={"time_rounded": "time"}, inplace=True)

        return strain_proportions
