"""
V2 analyzers for Rotasim - Maintains compatibility with v1 output formats

These analyzers work with the new v2 architecture where each strain is a separate
Rotavirus disease, but provide the same output format as v1 for backwards compatibility
with existing analysis scripts and data processing workflows.
"""

# Third-party imports
import numpy as np
import starsim as ss


class StrainStats(ss.Analyzer):
    """
    V2 analyzer to track strain proportions and counts - compatible with v1 output format

    This analyzer automatically detects all Rotavirus diseases in the simulation and
    tracks their infection counts and proportions over time. The output format matches
    the v1 StrainStats analyzer for backwards compatibility.

    Key differences from v1:
    - Auto-detects Rotavirus diseases instead of using sim.connectors.rota
    - Aggregates counts from individual disease instances
    - Same CSV output format and column names as v1
    - Works with dormant diseases that may become active during simulation

    Example usage:
        analyzer = StrainStats()
        sim = Sim(initial_strains=[(1,8), (2,4)], analyzers=[analyzer])
        sim.run()
        df = analyzer.to_df()  # Same format as v1
    """

    def __init__(self, **kwargs):
        """Initialize strain statistics analyzer"""
        super().__init__(**kwargs)

        # Will be populated during init_results
        self._rotavirus_diseases = []
        self._strain_names = []

    def init_results(self):
        """Initialize results storage - auto-detect all Rotavirus diseases"""
        super().init_results()

        # Find all Rotavirus disease instances in the simulation
        self._rotavirus_diseases = []
        self._strain_names = []

        for disease in self.sim.diseases.values():
            # Check if this is a Rotavirus disease by looking for G,P attributes
            if hasattr(disease, "G") and hasattr(disease, "P"):
                self._rotavirus_diseases.append(disease)

                # Create strain name in v1 format: (G, P) tuple as string
                strain_tuple = (disease.G, disease.P)
                strain_name = str(strain_tuple)
                self._strain_names.append(strain_name)

        n_diseases = len(self._rotavirus_diseases)
        if n_diseases == 0:
            if self.sim.pars.verbose:
                print("Warning: StrainStats analyzer found no Rotavirus diseases")
            return

        if self.sim.pars.verbose:
            print(f"StrainStats: Tracking {n_diseases} Rotavirus strains")
            if n_diseases >= 10:
                print(f"  First 5: {[f'G{d.G}P{d.P}' for d in self._rotavirus_diseases[:5]]}")
                print(f"  Last 5: {[f'G{d.G}P{d.P}' for d in self._rotavirus_diseases[-5:]]}")
            else:
                print(f"  All strains: {[f'G{d.G}P{d.P}' for d in self._rotavirus_diseases]}")

        # Create results for each strain - matching v1 format exactly
        for strain_name in self._strain_names:
            # Create proportion result
            self.results += ss.Result(
                f"{strain_name} proportion",
                dtype=float,
                scale=False,
                module="strainstats",
                shape=self.timevec.shape,
                timevec=self.timevec,
            )

            # Create count result
            self.results += ss.Result(
                f"{strain_name} count",
                dtype=float,
                scale=True,
                module="strainstats",
                shape=self.timevec.shape,
                timevec=self.timevec,
            )

    def step(self):
        """Collect strain statistics at each timestep"""
        if len(self._rotavirus_diseases) == 0:
            return  # No Rotavirus diseases found

        # Count infections for each strain
        strain_counts = {}
        total_count = 0

        for disease, strain_name in zip(self._rotavirus_diseases, self._strain_names):
            # Count currently infected agents for this disease
            count = disease.infected.sum()  # Number of infected agents
            strain_counts[strain_name] = count
            total_count += count

        # Update results - matching v1 logic exactly
        if total_count > 0:
            for strain_name, count in strain_counts.items():
                # Calculate proportion
                proportion = count / total_count

                # Store results using v1 format
                self.results[f"{strain_name} proportion"][self.sim.ti] = proportion
                self.results[f"{strain_name} count"][self.sim.ti] = count
        else:
            # No infections - set all to zero
            for strain_name in self._strain_names:
                self.results[f"{strain_name} proportion"][self.sim.ti] = 0.0
                self.results[f"{strain_name} count"][self.sim.ti] = 0.0

    def to_df(self):
        """Convert results to dataframe - matches v1 format exactly"""
        df = self.results.to_df()

        # Handle case where results.to_df() returns None
        if df is None:
            if self.sim.pars.verbose:
                print("Warning: StrainStats results.to_df() returned None - no data collected")
            return None

        # Remove duplicate timevec columns (same logic as v1)
        indexes_to_drop = df.columns.get_indexer_for(["timevec"])
        if len(indexes_to_drop) > 1:
            df.drop(columns=df.columns[indexes_to_drop[1:]], inplace=True)

        return df

    def get_strain_summary(self):
        """
        Get summary of strain statistics over the simulation

        Returns:
            Dict with strain summary statistics
        """
        if len(self._rotavirus_diseases) == 0:
            return {"total_strains": 0, "strain_stats": {}}

        summary = {"total_strains": len(self._rotavirus_diseases), "strain_stats": {}}

        # Calculate summary statistics for each strain
        for strain_name in self._strain_names:
            count_key = f"{strain_name} count"
            prop_key = f"{strain_name} proportion"

            if count_key in self.results and prop_key in self.results:
                counts = self.results[count_key].values
                proportions = self.results[prop_key].values

                # Remove NaN values for statistics
                valid_counts = counts[~np.isnan(counts)]
                valid_props = proportions[~np.isnan(proportions)]

                summary["strain_stats"][strain_name] = {
                    "max_count": float(np.max(valid_counts)) if len(valid_counts) > 0 else 0.0,
                    "mean_count": float(np.mean(valid_counts)) if len(valid_counts) > 0 else 0.0,
                    "max_proportion": float(np.max(valid_props)) if len(valid_props) > 0 else 0.0,
                    "mean_proportion": float(np.mean(valid_props)) if len(valid_props) > 0 else 0.0,
                    "total_timesteps_active": int(np.sum(valid_counts > 0)) if len(valid_counts) > 0 else 0,
                }

        return summary


class EventStats(ss.Analyzer):
    """
    Convenience analyzer to track simulation events

    This analyzer collects key simulation events per timestep:
    - births: Population births
    - deaths: Population deaths
    - recoveries: Disease recoveries across all strains
    - new_infections: Transmission events across all strains
    - wanings: Immunity waning events
    - reassortments: Genetic reassortment events
    - infected_agents: Number of currently infected agents (any strain)
    - coinfected_agents: Number of agents infected with >1 strain

    Example usage:
        analyzer = EventStats()
        sim = Sim(initial_strains=[(1,8), (2,4)], analyzers=[analyzer])
        sim.run()
    """

    def __init__(self, **kwargs):
        """Initialize event statistics analyzer"""
        super().__init__(**kwargs)
        self.event_types = [
            "births",
            "deaths",
            "recoveries",
            "new_infections",
            "wanings",
            "reassortments",
            "infected_agents",
            "coinfected_agents",
        ]

    def init_results(self):
        """Initialize results storage for event tracking"""
        super().init_results()

        # Create results for each event type
        self.events = {}
        for event_type in self.event_types:
            self.events[event_type] = 0
            self.results += ss.Result(
                event_type,
                dtype=int,
                scale=True,
                module="eventstats",
                shape=self.timevec.shape,
                timevec=self.timevec,
            )

        print(f"EventStats: Tracking {len(self.event_types)} event types")
        print(f"  Events: {', '.join(self.event_types)}")

    def step(self):
        """Collect event statistics at each timestep"""

        # Initialize all events to 0 for this timestep
        for event in self.events:
            self.events[event] = 0

        # Get population changes (births/deaths) from demographics modules
        # Check for births - handle missing results gracefully
        if hasattr(self.sim.results, "births") and hasattr(self.sim.results.births, "new"):
            self.events["births"] += self.sim.results.births.new[self.sim.ti]
        else:
            self.events["births"] = 0

        # Check for deaths - handle missing results gracefully
        if hasattr(self.sim.results, "deaths") and hasattr(self.sim.results.deaths, "new"):
            self.events["deaths"] += self.sim.results.deaths.new[self.sim.ti]
        elif hasattr(self.sim.results, "new_deaths"):
            self.events["deaths"] += self.sim.results.new_deaths[self.sim.ti]
        else:
            self.events["deaths"] = 0

        # Count recoveries and new infections across all Rotavirus diseases
        for disease in self.sim.diseases.values():
            if hasattr(disease, "G") and hasattr(disease, "P"):  # Is Rotavirus
                # Count agents who recovered this timestep
                if hasattr(disease.results, "new_recovered"):
                    self.events["recoveries"] += disease.results.new_recovered[self.sim.ti]

                # Count new infections this timestep (built into ss.Infection)
                if hasattr(disease.results, "new_infections"):
                    self.events["new_infections"] += disease.results.new_infections[self.sim.ti]

        # Count immunity waning events
        immunity_connector = self.sim.get_connector_by_type("RotaImmunityConnector")
        # if immunity_connector:
        #     self.events['wanings'] = immunity_connector.results.n_waned[self.sim.ti]

        # Count reassortment events from reassortment connector
        reassortment_connector = self.sim.get_connector_by_type("RotaReassortmentConnector")
        if reassortment_connector and hasattr(reassortment_connector, "results") and hasattr(reassortment_connector.results, "n_reassortments"):
            self.events["reassortments"] = reassortment_connector.results.n_reassortments[self.sim.ti]
        else:
            self.events["reassortments"] = 0

        # Count total infected agents and coinfected agents
        infection_counts = np.zeros(len(self.sim.people), dtype=int)
        for disease in self.sim.diseases.values():
            if hasattr(disease, "G") and hasattr(disease, "P"):  # Is Rotavirus
                infection_counts += disease.infected[:].astype(int)

        self.events["infected_agents"] = int(np.sum(infection_counts > 0))  # Agents infected with any strain
        self.events["coinfected_agents"] = int(np.sum(infection_counts > 1))  # Agents infected with >1 strain

        if self.sim.pars.verbose:
            print(self.events)

        # Store results
        for event_type, count in self.events.items():
            self.results[event_type][self.sim.ti] = count

    def to_df(self):
        """Convert results to dataframe"""
        df = self.results.to_df()

        # Remove duplicate timevec columns (same logic as v1)
        indexes_to_drop = df.columns.get_indexer_for(["timevec"])
        if len(indexes_to_drop) > 1:
            df.drop(columns=df.columns[indexes_to_drop[1:]], inplace=True)

        return df


class AgeStats(ss.Analyzer):
    """
    V2 analyzer to track age distribution - compatible with v1 rota_agecount_*.csv format

    This analyzer tracks the age distribution of the population over time,
    using the same age bins as v1 for backwards compatibility.

    The output format matches v1 rota_agecount_*.csv.

    Example usage:
        analyzer = AgeStats()
        sim = Sim(initial_strains=[(1,8), (2,4)], analyzers=[analyzer])
        sim.run()
        df = analyzer.to_df()  # Same format as v1 age counts
    """

    def __init__(self, **kwargs):
        """Initialize age statistics analyzer"""
        super().__init__(**kwargs)

        # Standard age bins and labels for age distribution analysis
        self.age_bins = [
            2 / 12,
            4 / 12,
            6 / 12,
            12 / 12,
            24 / 12,
            36 / 12,
            48 / 12,
            60 / 12,
            100,
        ]
        self.age_labels = [
            "0-2",
            "2-4",
            "4-6",
            "6-12",
            "12-24",
            "24-36",
            "36-48",
            "48-60",
            "60+",
        ]

    def init_results(self):
        """Initialize results storage for age distribution tracking"""
        super().init_results()

        # Create results for each age bin - matching v1 format exactly
        for age_label in self.age_labels:
            self.results += ss.Result(
                age_label,
                dtype=int,
                scale=True,
                module="agestats",
                shape=self.timevec.shape,
                timevec=self.timevec,
            )

        if self.sim.pars.verbose:
            print(f"AgeStats: Tracking {len(self.age_labels)} age bins")
            print(f"  Age bins: {self.age_labels}")

    def step(self):
        """Collect age distribution statistics at each timestep"""

        if not hasattr(self.sim, "people") or not hasattr(self.sim.people, "age"):
            # No age data available
            for age_label in self.age_labels:
                self.results[age_label][self.sim.ti] = 0
            return

        # Get population ages
        ages = self.sim.people.age

        # Bin ages using same logic as v1
        binned_ages = np.digitize(ages, self.age_bins)
        bin_counts = np.bincount(binned_ages, minlength=len(self.age_bins) + 1)

        # Store results for each age bin
        for i, age_label in enumerate(self.age_labels):
            count = bin_counts[i] if i < len(bin_counts) else 0
            self.results[age_label][self.sim.ti] = int(count)

    def to_df(self):
        """Convert results to dataframe - matches v1 format exactly"""
        df = self.results.to_df()

        # Remove duplicate timevec columns (same logic as v1)
        indexes_to_drop = df.columns.get_indexer_for(["timevec"])
        if len(indexes_to_drop) > 1:
            df.drop(columns=df.columns[indexes_to_drop[1:]], inplace=True)

        return df


class InfectedStrainStats(ss.Analyzer):
    """
    V2 analyzer to track individual infection events - produces rota_strains_infected_all_*.csv format

    This analyzer tracks each new infection event with detailed information about the infected agent,
    including their age, the infecting strain, and the time of infection. This data is used for
    calibration and detailed epidemiological analysis.

    Output format matches v1 rota_strains_infected_all_*.csv with columns:
    - id: Agent ID
    - Strain: Strain identifier (e.g., "G1P8", "G2P4A1B1")
    - CollectionTime: Simulation time when infection occurred (in years)
    - Age: Age category of infected agent
    - PopulationSize: Total population size at time of infection

    Example usage:
        analyzer = InfectedStrainStats()
        sim = Sim(initial_strains=[(1,8), (2,4)], analyzers=[analyzer])
        sim.run()
        df = analyzer.to_df()  # Event log format
    """

    def __init__(self, use_infection_based_severity=True, constant_severity=0.05, **kwargs):
        """
        Initialize infected strain statistics analyzer

        Args:
            use_infection_based_severity (bool): If True, severity varies by infection number (default).
                                                   If False, use constant_severity for all infections.
            constant_severity (float): Severity value when use_infection_based_severity=False (default: 0.05)
        """
        super().__init__(**kwargs)

        # Store severity configuration
        self.use_infection_based_severity = use_infection_based_severity
        self.constant_severity = constant_severity

        # Store infection events as lists
        self.infection_events = {
            'id': [],
            'Strain': [],
            'CollectionTime': [],
            'Age': [],
            'age_months_precise': [],  # Continuous age in months (for MAL-ED calibration)
            'PopulationSize': [],
            'n_infections': [],  # Infection number (1st, 2nd, 3rd, etc.)
            'severity': []  # Severity probability based on infection number
        }

        # Severity rates by infection number (only used if use_infection_based_severity=True)
        # Primary: 5.1%, Secondary: 6.44%, Third: 4.32%, Fourth+: 3.78%
        self.severity_rates = {
            1: 0.051,
            2: 0.0644,
            3: 0.0432,
            4: 0.0378  # 4th and higher
        }

        # Track which agents were infected in previous timestep to detect new infections
        self._prev_infected = {}

        # Age bins for categorization (matching v1 format)
        self.age_bins = [
            (0, 2/12),      # 0-2 months
            (2/12, 4/12),   # 2-4 months
            (4/12, 6/12),   # 4-6 months
            (6/12, 12/12),  # 6-12 months
            (12/12, 24/12), # 12-24 months
            (24/12, 36/12), # 24-36 months
            (36/12, 48/12), # 36-48 months
            (48/12, 60/12), # 48-60 months
            (60/12, np.inf) # 60+ months
        ]
        self.age_labels = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60+']

    def init_results(self):
        """Initialize results storage - find all Rotavirus diseases"""
        super().init_results()

        # Find all Rotavirus disease instances
        self._rotavirus_diseases = []
        for disease in self.sim.diseases.values():
            if hasattr(disease, 'G') and hasattr(disease, 'P'):
                self._rotavirus_diseases.append(disease)
                # Initialize tracking dict for this disease
                self._prev_infected[disease.name] = set()

        if self.sim.pars.verbose:
            print(f"InfectedStrainStats: Tracking infections for {len(self._rotavirus_diseases)} Rotavirus strains")

    def _get_age_category(self, age_years):
        """Convert age in years to age category string"""
        for i, (low, high) in enumerate(self.age_bins):
            if low <= age_years < high:
                return self.age_labels[i]
        return self.age_labels[-1]  # Default to 60+

    def step(self):
        """Collect new infection events at each timestep"""
        if len(self._rotavirus_diseases) == 0:
            return

        # Get current time in years (convert from days if needed)
        current_time_years = self.sim.t.relvec[self.sim.ti].years

        # Get current population size
        pop_size = len(self.sim.people)

        # Check each disease for new infections
        for disease in self._rotavirus_diseases:
            # Get currently infected agents
            currently_infected = disease.infected.uids

            # Find new infections (in current but not in previous)
            new_infections = currently_infected - self._prev_infected[disease.name]

            # Log each new infection
            for agent_id in new_infections:
                # Get agent age (already in years in Starsim)
                age_years = self.sim.people.age[agent_id]
                age_category = self._get_age_category(age_years)

                # Create strain name in full format (G1P8A1B1) to match v1 expectations
                # Default to A1B1 if backbone not specified
                if hasattr(disease, 'backbone') and disease.backbone is not None:
                    strain_name = f"G{disease.G}P{disease.P}A{disease.backbone[0]}B{disease.backbone[1]}"
                else:
                    # Default backbone A1B1
                    strain_name = f"G{disease.G}P{disease.P}A1B1"

                # Get infection number for this agent
                n_current = int(disease.n_infections[agent_id])

                # Calculate severity probability
                if self.use_infection_based_severity:
                    # Severity varies by infection number: 1st, 2nd, 3rd, and 4+ infections
                    if n_current <= 3:
                        severity_prob = self.severity_rates[n_current]
                    else:
                        severity_prob = self.severity_rates[4]  # 4th and higher
                else:
                    # Constant severity for all infections
                    severity_prob = self.constant_severity

                # Record the infection event
                self.infection_events['id'].append(int(agent_id))
                self.infection_events['Strain'].append(strain_name)
                self.infection_events['CollectionTime'].append(float(current_time_years))
                self.infection_events['Age'].append(age_category)
                self.infection_events['age_months_precise'].append(float(age_years) * 12.0)
                self.infection_events['PopulationSize'].append(int(pop_size))
                self.infection_events['n_infections'].append(n_current)  # Current infection number (1st, 2nd, 3rd, etc.)
                self.infection_events['severity'].append(severity_prob)

            # Update previous infected set for next timestep
            self._prev_infected[disease.name] = currently_infected

    def to_df(self):
        """Convert infection events to dataframe matching v1 format"""
        import pandas as pd

        # Create dataframe from stored events
        df = pd.DataFrame(self.infection_events)

        if self.sim.pars.verbose and len(df) > 0:
            print(f"InfectedStrainStats: Collected {len(df)} infection events")
            print(f"  Strains: {df['Strain'].unique()}")
            print(f"  Time range: {df['CollectionTime'].min():.2f} - {df['CollectionTime'].max():.2f} years")

        return df

    def get_infection_summary(self):
        """Get summary statistics of infection events"""
        import pandas as pd
        df = pd.DataFrame(self.infection_events)

        if len(df) == 0:
            return {"total_infections": 0, "strains": {}}

        summary = {
            "total_infections": len(df),
            "unique_agents": df['id'].nunique(),
            "time_range": (df['CollectionTime'].min(), df['CollectionTime'].max()),
            "strains": {}
        }

        # Per-strain summary
        for strain in df['Strain'].unique():
            strain_df = df[df['Strain'] == strain]
            summary["strains"][strain] = {
                "total_infections": len(strain_df),
                "unique_agents": strain_df['id'].nunique(),
                "age_distribution": strain_df['Age'].value_counts().to_dict()
            }

        return summary


# Legacy aliases for backwards compatibility
StrainStatistics = StrainStats  # In case v1 scripts use different name


class UidTracker(ss.Analyzer):
    """
    Analyzer to track disease state values over time for specific UIDs
    
    This analyzer tracks various disease state variables (like rel_sus, rel_trans, etc.)
    for specified agents across all Rotavirus strains over time. Useful for detailed 
    analysis of immunity patterns and disease dynamics for specific individuals or groups.
    
    Example usage:
        # Track specific UIDs
        tracker = UidTracker(uids=[0, 10, 20, 100], track_fields=['rel_sus'])
        
        # Track age-based sample (select UIDs after sim.init())
        sim = Sim(n_agents=10000)
        sim.init()
        children_uids = np.where(sim.people.age < 5)[0][:50]  # First 50 children under 5
        tracker = UidTracker(uids=children_uids, track_fields=['rel_sus', 'rel_trans'])
        
        sim.analyzers = [tracker]
        sim.run()
        
        # Access data for analysis
        df = tracker.to_df()
        rel_sus_matrix = tracker.get_field_matrix('G1P8', 'rel_sus')  # Shape: (timesteps, uids)
    """
    
    def __init__(self, uids=None, track_fields=None, **kwargs):
        """
        Initialize UID-specific state tracker
        
        Args:
            uids (int, list, or array): UID(s) to track. Can be:
                - Single int: track one agent
                - List/array of ints: track multiple specific agents
                - None: will need to be set later via set_uids()
            track_fields (list): List of disease state fields to track (default: ['rel_sus'])
                Common options: 'rel_sus', 'rel_trans', 'infected', 'susceptible', 'recovered'
            **kwargs: Additional analyzer parameters
        """
        super().__init__(**kwargs)
        
        # Store UIDs to track
        if uids is None:
            self.track_uids = []
        elif isinstance(uids, (int, np.integer)):
            self.track_uids = [int(uids)]
        else:
            self.track_uids = [int(uid) for uid in uids]
        
        # Store fields to track
        if track_fields is None:
            self.track_fields = ['rel_sus']
        elif isinstance(track_fields, str):
            self.track_fields = [track_fields]
        else:
            self.track_fields = list(track_fields)
        
        # Will be populated during init_results
        self.rotavirus_diseases = []
        self.strain_names = []
        

    def init_results(self):
        """Initialize results storage - auto-detect Rotavirus diseases and create UID-specific results"""
        super().init_results()
        
        if len(self.track_uids) == 0:
            if self.sim.pars.verbose:
                print("Warning: UidTracker has no UIDs to track.")
            return
        
        # # Validate UIDs
        # max_uid = len(self.sim.people) - 1
        # valid_uids = [uid for uid in self.track_uids if 0 <= uid <= max_uid]
        # invalid_uids = [uid for uid in self.track_uids if uid < 0 or uid > max_uid]
        #
        # if invalid_uids:
        #     print(f"Warning: Invalid UIDs removed: {invalid_uids} (population size: {len(self.sim.people)})")
        #
        # self.track_uids = valid_uids
        if len(self.track_uids) == 0:
            print("Warning: No valid UIDs to track after validation")
            return

        # Find all Rotavirus disease instances
        self.rotavirus_diseases = []
        self.strain_names = []
        
        for disease in self.sim.diseases.values():
            if hasattr(disease, "G") and hasattr(disease, "P"):
                self.rotavirus_diseases.append(disease)
                strain_name = f"G{disease.G}P{disease.P}"
                self.strain_names.append(strain_name)
        
        if len(self.rotavirus_diseases) == 0:
            if self.sim.pars.verbose:
                print("Warning: UidTracker found no Rotavirus diseases")
            return
            
        if self.sim.pars.verbose:
            print(f"UidTracker: Tracking {len(self.track_uids)} UIDs across {len(self.rotavirus_diseases)} strains")
            print(f"  UIDs: {self.track_uids[:10]}{'...' if len(self.track_uids) > 10 else ''}")
            print(f"  Strains: {self.strain_names}")
            print(f"  Fields: {self.track_fields}")
            
        # Create results for each strain-UID-field combination
        for strain_name in self.strain_names:
            for field_name in self.track_fields:
                for uid in self.track_uids:
                    result_name = f"{strain_name}_uid_{uid}_{field_name}"
                    self.results += ss.Result(
                        result_name,
                        dtype=float,
                        scale=False,
                        module="uidtracker",
                        shape=self.timevec.shape,
                        timevec=self.timevec,
                    )

    def step(self):
        """Collect state values for tracked UIDs at each timestep"""
        if len(self.rotavirus_diseases) == 0 or len(self.track_uids) == 0:
            return
            
        # Collect data for each strain, field, and tracked UID
        for disease, strain_name in zip(self.rotavirus_diseases, self.strain_names):
            for field_name in self.track_fields:
                # Check if field exists on disease
                if not hasattr(disease, field_name):
                    continue  # Skip missing fields
                    
                field_data = getattr(disease, field_name)

                for uid in self.track_uids:
                    result_name = f"{strain_name}_uid_{uid}_{field_name}"
                    self.results[result_name][self.sim.ti] = field_data[uid]

    def get_field_matrix(self, strain_name, field_name):
        """
        Get field values as a matrix for specified strain and field
        
        Args:
            strain_name (str): Strain name (e.g., 'G1P8')
            field_name (str): Field name (e.g., 'rel_sus')
            
        Returns:
            numpy.ndarray: Matrix of shape (n_timesteps, n_uids) with field values
        """
        if strain_name not in self.strain_names:
            raise ValueError(f"Strain {strain_name} not found. Available: {self.strain_names}")
        
        if field_name not in self.track_fields:
            raise ValueError(f"Field {field_name} not tracked. Available: {self.track_fields}")
            
        if len(self.track_uids) == 0:
            return np.array([])
            
        # Collect data for all UIDs
        n_timesteps = len(self.timevec)
        n_uids = len(self.track_uids)
        matrix = np.zeros((n_timesteps, n_uids))
        
        for uid_idx, uid in enumerate(self.track_uids):
            result_name = f"{strain_name}_uid_{uid}_{field_name}"
            if result_name in self.results:
                matrix[:, uid_idx] = self.results[result_name].values
                
        return matrix
    
    def get_uid_ages(self):
        """
        Get ages of tracked UIDs
        
        Returns:
            numpy.ndarray: Ages of tracked UIDs in years
        """
        if not hasattr(self.sim, 'people') or len(self.track_uids) == 0:
            return np.array([])
            
        ages = []
        for uid in self.track_uids:
            if uid < len(self.sim.people.age):
                ages.append(self.sim.people.age[uid])
            else:
                ages.append(0.0)  # Default age for invalid UID
                
        return np.array(ages)
    
    def to_df(self):
        """Convert results to dataframe"""
        df = self.results.to_df()
        
        if df is None:
            return None
            
        # Remove duplicate timevec columns
        indexes_to_drop = df.columns.get_indexer_for(["timevec"])
        if len(indexes_to_drop) > 1:
            df.drop(columns=df.columns[indexes_to_drop[1:]], inplace=True)
            
        return df

    def plot_field_heatmap(self, strain_name, field_name='rel_sus', max_age=10, 
                          age_bin_years=1, figsize=(12, 8), cmap='viridis_r', 
                          save_path=None, use_age_bins=None):
        """
        Generate heat map visualization of tracked field for specified strain
        
        This function creates a heat map showing how the tracked field varies
        over time. When UidTracker data is available, it shows individual UIDs on the x-axis
        with their ages in labels. Otherwise, it falls back to age bins for aggregated data.
        
        Args:
            strain_name (str): Name of strain to plot (e.g., 'G1P8')
            field_name (str): Name of field to plot (must be in track_fields)
            max_age (float): Maximum age in years to include (default: 10) - only used for fallback modes
            age_bin_years (float): Age bin size in years (default: 1) - only used for fallback modes  
            figsize (tuple): Figure size (width, height)
            cmap (str): Colormap for heat map (default: 'viridis_r' - dark=high values)
            save_path (str, optional): Path to save figure
            use_age_bins (bool, optional): Force use of age bins even with UidTracker data (default: None=auto)
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
            
        Example:
            # After running simulation with UidTracker
            uid_tracker = sim.analyzers['uidtracker'] 
            fig, ax = uid_tracker.plot_field_heatmap('G1P8', 'rel_sus')
            plt.show()
        """
        import matplotlib.pyplot as plt
        
        # Validate inputs
        if field_name not in self.track_fields:
            raise ValueError(f"Field {field_name} not tracked. Available: {self.track_fields}")
        
        if strain_name not in self.strain_names:
            raise ValueError(f"Strain {strain_name} not found. Available: {self.strain_names}")
        
        # Find the specified strain disease for fallback mode
        strain_disease = None
        for disease in self.sim.diseases.values():
            if hasattr(disease, "G") and hasattr(disease, "P"):
                if f"G{disease.G}P{disease.P}" == strain_name:
                    strain_disease = disease
                    break
        
        if strain_disease is None:
            raise ValueError(f"Strain {strain_name} not found in simulation diseases")
        
        # Get field data from UidTracker
        field_data = None
        uid_ages = None
        tracked_uids = None
        data_source = "unknown"
        use_uid_mode = False
        
        if len(self.track_uids) > 0 and use_age_bins != True:
            try:
                field_matrix = self.get_field_matrix(strain_name, field_name)
                if field_matrix.size > 0:
                    field_data = field_matrix  # Shape: (timesteps, uids)
                    uid_ages = self.get_uid_ages()
                    tracked_uids = self.track_uids
                    data_source = "UidTracker"
                    use_uid_mode = True
                    n_timesteps, n_agents = field_data.shape
                    print(f"Using UidTracker {field_name} data: {n_timesteps} timesteps, {n_agents} tracked UIDs")
            except (ValueError, KeyError) as e:
                print(f"Could not access UidTracker data: {e}")
        
        # Fallback: use current state snapshot from strain disease
        if field_data is None:
            print(f"No time series {field_name} data found. Using current state snapshot.")
            if not hasattr(strain_disease, field_name):
                raise ValueError(f"No {field_name} data available for strain {strain_name}")
            
            # Create single timepoint data from current state
            current_field_data = getattr(strain_disease, field_name)
            if hasattr(current_field_data, 'values'):
                current_values = current_field_data.values  # Get numpy array of current values
            else:
                current_values = np.array([current_field_data])  # Scalar case
                
            field_data = current_values.reshape(1, -1)  # Shape: (1, n_agents)
            uid_ages = self.sim.people.age.values  # All agent ages
            data_source = "current state snapshot"
            n_timesteps, n_agents = field_data.shape
            print(f"Using current state snapshot: {n_agents} agents")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        if use_uid_mode:
            # Transpose data: (timesteps, uids) -> (uids, timesteps)
            # Display with time steps on x-axis, UIDs on y-axis
            heatmap_data = field_data.T  # Transpose to get (uids, timesteps)
            n_uids, n_timesteps = heatmap_data.shape
            
            # Determine appropriate vmin/vmax for the field
            vmin, vmax = 0, 1
            if field_name != 'rel_sus':
                # For other fields, use data range
                data_min, data_max = np.nanmin(heatmap_data), np.nanmax(heatmap_data)
                if not np.isnan(data_min) and not np.isnan(data_max):
                    vmin, vmax = data_min, data_max
            
            # Create heat map with time steps on x-axis, UIDs on y-axis
            im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, origin='lower',
                           vmin=vmin, vmax=vmax, interpolation='nearest')
            
            # Set axis labels
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Agent (UID : Age)', fontsize=12)
            title_suffix = f" ({n_timesteps} timesteps, {n_uids} UIDs)"
            ax.set_title(f'{field_name.replace("_", " ").title()} Heat Map - {strain_name}{title_suffix}',
                        fontsize=14, fontweight='bold')
            
            # Set x-tick labels (time steps)
            if n_timesteps > 1:
                x_tick_positions = np.arange(0, n_timesteps, max(1, n_timesteps//10))  # Show ~10 ticks
                ax.set_xticks(x_tick_positions)
                ax.set_xticklabels([f'{t}' for t in x_tick_positions])
            else:
                ax.set_xticks([0])
                ax.set_xticklabels(['Final'])
            
            # Set y-tick labels to show UID and age
            y_tick_positions = np.arange(n_uids)
            if n_uids <= 20:
                # Show all UIDs if not too many
                y_tick_labels = [f'{uid}:{age:.1f}' for uid, age in zip(tracked_uids, uid_ages)]
                ax.set_yticks(y_tick_positions)
                ax.set_yticklabels(y_tick_labels)
            else:
                # Show subset of UIDs if too many
                tick_step = max(1, n_uids // 15)  # Show ~15 ticks max
                show_positions = y_tick_positions[::tick_step]
                show_labels = [f'{tracked_uids[i]}:{uid_ages[i]:.1f}' for i in show_positions]
                ax.set_yticks(show_positions)
                ax.set_yticklabels(show_labels)
                
        else:
            # Use age bins (fallback behavior)
            age_bins = np.arange(0, max_age + age_bin_years, age_bin_years)
            n_age_bins = len(age_bins) - 1
            
            # Initialize heat map matrix: rows=age bins, cols=time points
            n_timesteps = field_data.shape[0]
            heatmap_data = np.full((n_age_bins, n_timesteps), np.nan)
            
            # For each timestep, bin agents by age and calculate mean field value
            for t in range(n_timesteps):
                # Calculate current ages (assumes timestep = 1 year for aging, adjust if needed)
                dt_years = 1.0 if n_timesteps > 1 else 0.0
                current_ages = uid_ages + (t * dt_years)
                
                # Get field values for this timestep
                field_t = field_data[t, :]
                
                # Bin by age and calculate mean field value for each age bin
                for i in range(n_age_bins):
                    age_mask = (current_ages >= age_bins[i]) & (current_ages < age_bins[i+1])
                    if np.any(age_mask):
                        heatmap_data[i, t] = np.mean(field_t[age_mask])
            
            # Determine appropriate vmin/vmax for the field
            vmin, vmax = 0, 1
            if field_name != 'rel_sus':
                # For other fields, use data range
                data_min, data_max = np.nanmin(heatmap_data), np.nanmax(heatmap_data)
                if not np.isnan(data_min) and not np.isnan(data_max):
                    vmin, vmax = data_min, data_max
            
            # Create heat map (flip y-axis so age 0 is at bottom)
            im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, origin='lower',
                           vmin=vmin, vmax=vmax, interpolation='nearest')
            
            # Set axis labels and ticks
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Age (years)', fontsize=12)
            title_suffix = f" (snapshot)" if n_timesteps == 1 else f" ({n_timesteps} timesteps)"
            ax.set_title(f'{field_name.replace("_", " ").title()} Heat Map - {strain_name}{title_suffix}',
                        fontsize=14, fontweight='bold')
            
            # Set y-tick labels to show age ranges
            y_tick_positions = np.arange(n_age_bins)
            y_tick_labels = [f'{age_bins[i]:.0f}-{age_bins[i+1]:.0f}' for i in range(n_age_bins)]
            ax.set_yticks(y_tick_positions)
            ax.set_yticklabels(y_tick_labels)
            
            # Set x-tick labels
            if n_timesteps > 1:
                x_tick_positions = np.arange(0, n_timesteps, max(1, n_timesteps//10))  # Show ~10 ticks
                ax.set_xticks(x_tick_positions)
                ax.set_xticklabels([f'{t}' for t in x_tick_positions])
            else:
                # Single timepoint
                ax.set_xticks([0])
                ax.set_xticklabels(['Final'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(field_name.replace('_', ' ').title(), fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add data source annotation
        mode_info = " (UID mode)" if use_uid_mode else " (age bin mode)"
        ax.text(0.02, 0.98, f'Data source: {data_source}{mode_info}', transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heat map saved to: {save_path}")
        
        return fig, ax


class PersonTimeByAge(ss.Analyzer):
    """Accumulate true person-time per age bin within a calibration window.

    Incidence rate denominators must be person-time *integrated over the
    observation window* -- the sum over timesteps of (headcount in bin) x dt.
    A common shortcut snapshots the population once (e.g. at sim end) and
    multiplies the headcount by the window length. That is only correct for a
    stationary population; when births != deaths the population size drifts and
    the snapshot systematically mis-states the denominator (a growing
    population inflates an end-of-sim snapshot, biasing IR low).

    This analyzer does it exactly: at every step whose simulation time falls
    inside ``calibration_window`` (years from sim start), it adds
    ``count_in_bin * dt_years`` to a running per-bin total. The result is the
    unbiased person-time regardless of demographic change.

    Args:
        bins_months: dict of ``label -> (low_month, high_month)``, half-open
            ``[low, high)``. Defaults to the MAL-ED age bins.
        calibration_window: ``(start_year, stop_year)`` measured in years from
            sim start; only steps within this window are accumulated. ``None``
            accumulates over the whole run.

    Example:
        pt = PersonTimeByAge(calibration_window=(5.0, 10.0))
        sim = ss.Sim(..., analyzers=[pt]); sim.run()
        pm = sim.analyzers['persontimebyage'].person_months  # {bin: person-months}
    """

    DEFAULT_BINS_MONTHS = {
        '<6 m':    (0.0, 6.0),
        '6-11 m':  (6.0, 12.0),
        '12-23 m': (12.0, 24.0),
        '24-35 m': (24.0, 36.0),
    }

    def __init__(self, bins_months=None, calibration_window=None, **kwargs):
        super().__init__(**kwargs)
        self.bins_months = dict(bins_months) if bins_months is not None else dict(self.DEFAULT_BINS_MONTHS)
        self.calibration_window = calibration_window
        # Accumulated person-YEARS per bin (converted to months on read).
        self.person_years_acc = {label: 0.0 for label in self.bins_months}

    def step(self):
        if not hasattr(self.sim, "people") or not hasattr(self.sim.people, "age"):
            return
        dt_years = self.dt.years
        # Window membership uses calendar time (relvec), matching the convention
        # used for infection CollectionTime, so denominators and case windows align.
        year_from_start = self.sim.t.relvec[self.sim.ti].years
        if self.calibration_window is not None:
            lo, hi = self.calibration_window
            if not (lo <= year_from_start < hi):
                return
        ages = self.sim.people.age.values  # years
        alive = self.sim.people.alive.values  # dead agents contribute no observation time
        for label, (lo_m, hi_m) in self.bins_months.items():
            lo_y, hi_y = lo_m / 12.0, hi_m / 12.0
            count = int(((ages >= lo_y) & (ages < hi_y) & alive).sum())
            self.person_years_acc[label] += count * dt_years

    @property
    def person_months(self):
        """Accumulated person-MONTHS per age bin (the IR denominator)."""
        return {label: py * 12.0 for label, py in self.person_years_acc.items()}


class MALEDTargets(ss.Analyzer):
    """Compute MAL-ED calibration targets in-step, emitting only tiny summaries.

    A memory-bounded alternative to logging every infection event with
    ``InfectedStrainStats`` and post-processing into targets. The full event log
    is O(infections), which blows up at high prevalence and over long runs (and
    multiplies across parallel workers). This analyzer folds the detection and
    age-binning into the ``step`` loop and retains only:

      - one integer case counter per MAL-ED age bin (symptomatic events <=36m),
      - accumulated person-time per bin (the IR denominator),
      - one first-detected infection age per agent,
      - an in-window prevalence series,

    so memory is O(agents), independent of prevalence and duration.

    It reproduces ``process_incidence_maled.process_model`` exactly (validated):
      - symptomatic prob = logistic(beta0 + beta1*(a-12) + beta2*(a-12)^2), age a
        in months capped at 60 (the 'age_only' symptom model);
      - optional reporting filter reporting_rate*severity (no-op at 1.0*1.0);
      - symptomatic IR per bin = symptomatic events <=36m / person-months * 100;
      - first-DETECTED age per agent: detected = symptomatic OR (asymptomatic AND
        uniform < p_asymp_detect), earliest per agent, <=36m.

    Window membership and infection time use calendar time (relvec), matching the
    convention used for infection CollectionTime.

    Args:
        calibration_window: (start_year, stop_year) in years from sim start.
        beta0, beta1, beta2: age-symptom logistic parameters.
        reporting_rate, constant_severity: reporting filter (1.0/1.0 = no-op).
        p_asymp_detect: detection prob for asymptomatic infections.
        censor_at_months: drop infections above this age (MAL-ED follow-up).
        seed: RNG seed for the symptomatic/detection draws.

    Example:
        a = MALEDTargets(calibration_window=(5,10), beta0=-1, beta1=-0.1, beta2=-0.05)
        sim = ss.Sim(..., analyzers=[a]); sim.run()
        out = sim.analyzers['maledtargets'].results_dict()  # ir, first_infection, prev
    """

    LABELS = ['<6 m', '6-11 m', '12-23 m', '24-35 m']
    EDGES_M = np.array([0.0, 6.0, 12.0, 24.0, 36.0])
    BINS_M = {'<6 m': (0, 6), '6-11 m': (6, 12), '12-23 m': (12, 24), '24-35 m': (24, 36)}

    def __init__(self, calibration_window, beta0, beta1, beta2,
                 reporting_rate=1.0, constant_severity=1.0,
                 p_asymp_detect=0.4, censor_at_months=36.0, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.window = calibration_window
        self.beta0, self.beta1, self.beta2 = beta0, beta1, beta2
        self.reporting_rate = reporting_rate
        self.constant_severity = constant_severity
        self.p_asymp = p_asymp_detect
        self.censor = censor_at_months
        self.rng = np.random.default_rng(seed)
        self.cases = {b: 0 for b in self.LABELS}
        self.person_years = {b: 0.0 for b in self.LABELS}
        self.first_detected = {}     # uid -> age_months (first detected <=36m)
        self.prev = []               # in-window prevalence series

    def init_pre(self, sim, force=False):
        super().init_pre(sim, force)
        self._dty = self.dt.years

    def init_results(self):
        super().init_results()
        self._diseases = [d for d in self.sim.diseases.values() if hasattr(d, 'G')]
        self._prev_infected = {d.name: d.infected.uids for d in self._diseases}

    def _symp_prob(self, age_m):
        ac = np.minimum(age_m, 60.0) - 12.0
        lp = self.beta0 + self.beta1 * ac + self.beta2 * ac * ac
        return 1.0 / (1.0 + np.exp(-lp))

    def step(self):
        sim = self.sim
        yr = sim.t.relvec[sim.ti].years
        in_window = (self.window[0] <= yr < self.window[1])
        ages_y = sim.people.age.values
        alive = sim.people.alive.values

        if in_window:
            for b, (lo_m, hi_m) in self.BINS_M.items():
                lo, hi = lo_m / 12.0, hi_m / 12.0
                self.person_years[b] += int(((ages_y >= lo) & (ages_y < hi) & alive).sum()) * self._dty
            inf = np.zeros(len(ages_y), dtype=bool)
            for d in self._diseases:
                inf |= d.infected.values
            n = int(alive.sum())
            if n > 0:
                self.prev.append(float((inf & alive).sum() / n))

        for d in self._diseases:
            cur = d.infected.uids
            if in_window:
                new = cur - self._prev_infected[d.name]
                if len(new):
                    new_arr = np.asarray(new)
                    age_m = np.asarray(sim.people.age[new]) * 12.0
                    m36 = age_m <= self.censor
                    if m36.any():
                        am = age_m[m36]
                        uu = new_arr[m36]
                        is_symp = self.rng.random(len(am)) < self._symp_prob(am)
                        if self.reporting_rate is not None:
                            rep_keep = self.rng.random(len(am)) < (self.reporting_rate * self.constant_severity)
                            is_symp = is_symp & rep_keep
                        if is_symp.any():
                            idx = np.digitize(am[is_symp], self.EDGES_M) - 1
                            for k in range(4):
                                self.cases[self.LABELS[k]] += int((idx == k).sum())
                        is_det = is_symp | (~is_symp & (self.rng.random(len(am)) < self.p_asymp))
                        for u, a, det in zip(uu, am, is_det):
                            if det and int(u) not in self.first_detected:
                                self.first_detected[int(u)] = float(a)
            self._prev_infected[d.name] = cur

    def results_dict(self):
        """Tiny summary: IR by age bin, first-infection quartiles, prevalence."""
        person_months = {b: self.person_years[b] * 12.0 for b in self.LABELS}
        ir = {b: (self.cases[b] / person_months[b] * 100.0 if person_months[b] > 0 else 0.0)
              for b in self.LABELS}
        ages = np.array(list(self.first_detected.values()), dtype=float)
        ages = ages[ages <= self.censor]
        if len(ages):
            q25, med, q75 = (float(x) for x in np.quantile(ages, [0.25, 0.5, 0.75]))
        else:
            q25 = med = q75 = float('nan')
        prev_mean = float(np.mean(self.prev)) if self.prev else float('nan')
        half = len(self.prev) // 2
        prev_drift = (float(np.mean(self.prev[half:])) - float(np.mean(self.prev[:half]))) if half else float('nan')
        return dict(ir=ir, cases=dict(self.cases), person_months=person_months,
                    fi_q25=q25, fi_median=med, fi_q75=q75, fi_n=int(len(ages)),
                    prev_mean=prev_mean, prev_drift=prev_drift)


# Make importable from package root
__all__ = ["StrainStats", "StrainStatistics", "EventStats", "AgeStats", "InfectedStrainStats", "UidTracker", "PersonTimeByAge", "MALEDTargets"]
