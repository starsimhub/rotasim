"""
V2 analyzers for Rotasim - Maintains compatibility with v1 output formats

These analyzers work with the new v2 architecture where each strain is a separate
Rotavirus disease, but provide the same output format as v1 for backwards compatibility
with existing analysis scripts and data processing workflows.
"""
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
            if hasattr(disease, 'G') and hasattr(disease, 'P'):
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
                f'{strain_name} proportion', 
                dtype=float, 
                scale=False, 
                module='strainstats', 
                shape=self.timevec.shape, 
                timevec=self.timevec
            )
            
            # Create count result  
            self.results += ss.Result(
                f'{strain_name} count', 
                dtype=float, 
                scale=True,
                module='strainstats', 
                shape=self.timevec.shape, 
                timevec=self.timevec
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
            count = len(disease.infected.uids)  # Number of infected agents
            strain_counts[strain_name] = count
            total_count += count
        
        # Update results - matching v1 logic exactly
        if total_count > 0:
            for strain_name, count in strain_counts.items():
                # Calculate proportion 
                proportion = count / total_count
                
                # Store results using v1 format
                self.results[f'{strain_name} proportion'][self.sim.ti] = proportion
                self.results[f'{strain_name} count'][self.sim.ti] = count
        else:
            # No infections - set all to zero
            for strain_name in self._strain_names:
                self.results[f'{strain_name} proportion'][self.sim.ti] = 0.0
                self.results[f'{strain_name} count'][self.sim.ti] = 0.0
                
    def to_df(self):
        """Convert results to dataframe - matches v1 format exactly"""
        df = self.results.to_df()
        
        # Handle case where results.to_df() returns None
        if df is None:
            if self.sim.pars.verbose:
                print("Warning: StrainStats results.to_df() returned None - no data collected")
            return None
        
        # Remove duplicate timevec columns (same logic as v1)
        indexes_to_drop = df.columns.get_indexer_for(['timevec'])
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
            return {'total_strains': 0, 'strain_stats': {}}
        
        summary = {
            'total_strains': len(self._rotavirus_diseases),
            'strain_stats': {}
        }
        
        # Calculate summary statistics for each strain
        for strain_name in self._strain_names:
            count_key = f'{strain_name} count'
            prop_key = f'{strain_name} proportion'
            
            if count_key in self.results and prop_key in self.results:
                counts = self.results[count_key].values
                proportions = self.results[prop_key].values
                
                # Remove NaN values for statistics
                valid_counts = counts[~np.isnan(counts)]
                valid_props = proportions[~np.isnan(proportions)]
                
                summary['strain_stats'][strain_name] = {
                    'max_count': float(np.max(valid_counts)) if len(valid_counts) > 0 else 0.0,
                    'mean_count': float(np.mean(valid_counts)) if len(valid_counts) > 0 else 0.0,
                    'max_proportion': float(np.max(valid_props)) if len(valid_props) > 0 else 0.0,
                    'mean_proportion': float(np.mean(valid_props)) if len(valid_props) > 0 else 0.0,
                    'total_timesteps_active': int(np.sum(valid_counts > 0)) if len(valid_counts) > 0 else 0,
                }
        
        return summary


class EventStats(ss.Analyzer):
    """
    V2 analyzer to track simulation events - compatible with v1 event_counts_*.csv format
    
    This analyzer tracks key simulation events per timestep matching the v1 format:
    - births: Population births
    - deaths: Population deaths  
    - recoveries: Disease recoveries across all strains
    - contacts: Transmission events across all strains
    - wanings: Immunity waning events
    - reassortments: Genetic reassortment events
    
    The output format matches v1 event_counts_*.csv for backwards compatibility.
    
    Example usage:
        analyzer = EventStats()
        sim = Sim(initial_strains=[(1,8), (2,4)], analyzers=[analyzer])
        sim.run()
        df = analyzer.to_df()  # Same format as v1 event_counts
    """
    
    def __init__(self, **kwargs):
        """Initialize event statistics analyzer"""
        super().__init__(**kwargs)
        
    def init_results(self):
        """Initialize results storage for event tracking"""
        super().init_results()
        
        # Create results for each event type - matching v1 format exactly
        event_types = [
            'births',
            'deaths', 
            'recoveries',
            'contacts',
            'wanings',
            'reassortments',
            'total_infected',
            'coinfected_agents'
        ]
        
        for event_type in event_types:
            self.results += ss.Result(
                event_type,
                dtype=int,
                scale=True,
                module='eventstats',
                shape=self.timevec.shape,
                timevec=self.timevec
            )
            
        if self.sim.pars.verbose:
            print(f"EventStats: Tracking {len(event_types)} event types")
            print(f"  Events: {', '.join(event_types)}")
        
    def step(self):
        """Collect event statistics at each timestep"""
        
        # Initialize all events to 0 for this timestep
        events = {
            'births': 0,
            'deaths': 0,
            'recoveries': 0, 
            'contacts': 0,
            'wanings': 0,
            'reassortments': 0
        }
        
        # Get population changes (births/deaths) from demographics modules
        for module in self.sim.modules:
            if hasattr(module, 'results'):
                #todo replace with people births/deaths logic after upgrading to SS v3.
                # Check for births
                if hasattr(module.results, 'new_births'):
                    events['births'] += getattr(module.results.new_births, self.sim.ti, 0)
                # Check for deaths  
                if hasattr(module.results, 'new_deaths'):
                    events['deaths'] += getattr(module.results.new_deaths, self.sim.ti, 0)
        
        # Count recoveries and new infections across all Rotavirus diseases
        for disease in self.sim.diseases.values():
            if hasattr(disease, 'G') and hasattr(disease, 'P'):  # Is Rotavirus
                # Count agents who recovered this timestep
                if hasattr(disease.results, 'new_recovered'):
                    events['recoveries'] += disease.results.new_recovered[self.sim.ti]
                
                # Count new infections this timestep (built into ss.Infection)
                if hasattr(disease.results, 'new_infections'):
                    events['contacts'] += disease.results.new_infections[self.sim.ti]
        
        # Count immunity waning events
        immunity_connector = self.sim.get_connector_by_type('RotaImmunityConnector', warn_if_multiple=False)
        if immunity_connector:
            events['wanings'] = immunity_connector.results.n_waned[self.sim.ti]
        
        # Count reassortment events from reassortment connector
        reassortment_connector = self.sim.get_connector_by_type('RotaReassortmentConnector', warn_if_multiple=False)
        if reassortment_connector:
            events['reassortments'] = reassortment_connector.results.n_reassortments[self.sim.ti]

        # Count total infected agents and coinfected agents
        infection_counts = np.zeros(len(self.sim.people), dtype=int)
        for disease in self.sim.diseases.values():
            if hasattr(disease, 'G') and hasattr(disease, 'P'):  # Is Rotavirus
                infection_counts += disease.infected[:].astype(int)
        
        events['total_infected'] = int(np.sum(infection_counts > 0))  # Agents infected with any strain
        events['coinfected_agents'] = int(np.sum(infection_counts > 1))  # Agents infected with >1 strain

        if self.sim.pars.verbose > 1:
            print(events)

        # Store results
        for event_type, count in events.items():
            self.results[event_type][self.sim.ti] = count
            
    def to_df(self):
        """Convert results to dataframe - matches v1 format exactly"""
        df = self.results.to_df()
        
        # Remove duplicate timevec columns (same logic as v1)
        indexes_to_drop = df.columns.get_indexer_for(['timevec'])
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
        
        # V1 age bins and labels (from v1_legacy code)
        self.age_bins = [2/12, 4/12, 6/12, 12/12, 24/12, 36/12, 48/12, 60/12, 100]
        self.age_labels = ["0-2", "2-4", "4-6", "6-12", "12-24", "24-36", "36-48", "48-60", "60+"]
        
    def init_results(self):
        """Initialize results storage for age distribution tracking"""
        super().init_results()
        
        # Create results for each age bin - matching v1 format exactly
        for age_label in self.age_labels:
            self.results += ss.Result(
                age_label,
                dtype=int,
                scale=True,
                module='agestats',
                shape=self.timevec.shape,
                timevec=self.timevec
            )
            
        if self.sim.pars.verbose:
            print(f"AgeStats: Tracking {len(self.age_labels)} age bins")
            print(f"  Age bins: {self.age_labels}")
        
    def step(self):
        """Collect age distribution statistics at each timestep"""
        
        if not hasattr(self.sim, 'people') or not hasattr(self.sim.people, 'age'):
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
        indexes_to_drop = df.columns.get_indexer_for(['timevec'])
        if len(indexes_to_drop) > 1:
            df.drop(columns=df.columns[indexes_to_drop[1:]], inplace=True)
        
        return df


# Legacy aliases for backwards compatibility
StrainStatistics = StrainStats  # In case v1 scripts use different name


# Make importable from package root
__all__ = ['StrainStats', 'StrainStatistics', 'EventStats', 'AgeStats']