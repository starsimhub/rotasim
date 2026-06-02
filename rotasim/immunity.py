"""
High-performance immunity connector for Rotavirus v2 architecture
Uses bitmask vectorization for cross-strain immunity calculations
"""

# Standard library imports
# (none needed for this module)

# Third-party imports
import numpy as np
import starsim as ss

# Local imports
from .rotavirus import Rotavirus


class PathogenMatch:
    """Define whether pathogens are completely heterotypic, partially heterotypic, or homotypic"""

    COMPLETE_HETERO = 1
    PARTIAL_HETERO = 2
    HOMOTYPIC = 3


class RotaImmunityConnector(ss.Connector):
    """
    High-performance cross-strain immunity connector using bitmask vectorization

    This connector automatically detects all Rotavirus disease instances in the
    simulation and manages cross-strain immunity using efficient bitwise operations
    on entire population arrays, avoiding all UID loops.
    """

    def __init__(self, pars=None, **kwargs):
        """
        Initialize immunity connector with default parameters

        Args:
            pars (dict, optional): Parameters dict to override defaults
            **kwargs: Additional parameters

        Default parameters:
            homotypic_immunity_efficacy (0.9): Protection from same G,P strain
            partial_heterotypic_immunity_efficacy (0.5): Protection from shared G or P
            complete_heterotypic_immunity_efficacy (0.3): Protection from different G,P
            naive_immunity_efficacy (0.0): Baseline immunity for naive individuals
            # full_waning_rate: Immunity waning rate (~273 days mean, 365/273 per year)
            immunity_waning_delay (0 days): Delay before immunity decay starts
            cotransmission_prob (2%): Probability of co-transmitting multiple strains
        """
        super().__init__()

        # Define immunity parameters
        self.define_pars(
            homotypic_immunity_efficacy=1.0,  # Protection from same G,P strain
            partial_heterotypic_immunity_efficacy=0.5,  # Protection from shared G or P
            complete_heterotypic_immunity_efficacy=0.3,  # Protection from different G,P (or no prior exposure to any strain)
            naive_immunity_efficacy=0.0,  # Baseline immunity for naive individuals (0.0 = fully susceptible)
            # immunity_waning_delay=ss.days(14),  # Time delay before immunity decay starts (90 days = ~3 months)
            # Maternal immunity parameters (passive immunity from mother).
            # Default efficacy=0.0 means maternal immunity is OFF; set >0 to enable.
            # Protection at age a (years): efficacy * exp(-ln(2) * a / half_life_years).
            # Combined with acquired immunity via element-wise max in rel_sus calc.
            maternal_immunity_efficacy=0.0,
            maternal_immunity_half_life=ss.days(90),
            # Age-dependent baseline immunity parameters
            # adult_baseline_immunity=0.0,  # Baseline immunity for adults (e.g., from childhood exposure history). Calibration parameter.
            # adult_age_threshold=5.0,  # Age threshold (in years) for applying adult baseline immunity (default 5 years)
            # Long-term immunity parameters (permanent immunity after multiple infections)
            long_term_immunity_prob_after_1=0.39,  # Probability of long-term immunity after 1st infection (39%)
            long_term_immunity_prob_after_2=0.52,  # Probability of long-term immunity after 2nd infection (52%)
            long_term_immunity_prob_after_3=0.67,  # Probability of long-term immunity after 3rd infection (67%)
            long_term_immunity_prob_after_4=1.00,  # Probability of long-term immunity after 4th+ infections (100%)
            # infection_history_susceptibility_factors = {0: 1, 1: 1, 2: 1, 3: 1},  # Susceptibility scaling based on total infection history. We may want to remove this feature later.
            cotransmission_prob=ss.bernoulli(
                p=0.02
            ),  # Probability of transmitting all strains instead of dominant strain selection (2%). We may want to remove this feature later.
            immunity_init_dist = ss.randint(),
            baseline_immunity_exponential_rate = 0.1,  # Used for exponential model only
            # Fixed susceptibility model (alternative to exponential model)
            use_fixed_susceptibility=False,  # Use fixed susceptibility values based on infection count
            sus_after_1=0.67,  # Susceptibility after 1 infection (33% protection)
            sus_after_2=0.50,  # Susceptibility after 2 infections (50% protection)
            sus_after_3plus=0.36,  # Susceptibility after 3+ infections (64% protection)
        )

        # Update with user parameters
        self.update_pars(pars=pars, **kwargs)

        # Define immunity state arrays

        #  name=name, dtype=ss_int, nan=int_nan, **kwargs)
        self.define_states(
            ss.Arr(
                "exposed_GP_bitmask", dtype=np.int64, nan=ss.dtypes.int_nan, default=0
            ),  # Bitmask of exposed (G,P) pairs
            ss.Arr("exposed_G_bitmask", dtype=np.int64, nan=ss.dtypes.int_nan, default=0),  # Bitmask of exposed G types
            ss.Arr("exposed_P_bitmask", dtype=np.int64, nan=ss.dtypes.int_nan, default=0),  # Bitmask of exposed P types
            ss.FloatArr("oldest_infection", default=np.nan),  # Time of first infection (for waning)
            ss.BoolArr("has_immunity", default=False),  # Whether agent has any immunity
            ss.BoolArr("long_term_immune", default=False),  # Whether agent has permanent long-term immunity
            ss.FloatArr("long_term_immune_age", default=np.nan),  # Age (in years) when agent developed long-term immunity
            ss.FloatArr(
                "num_recovered_infections", default=0.0
            ),  # Total number of prior infections (for scaling susceptibility)
            ss.FloatArr(
                "num_current_infections", default=0.0
            ),  # Current number of active infections (for coinfection logic)
            ss.FloatArr(
                "homotypic_immunity_decay_factor", default=0.0
            ),  # Decay factor for immunity over time (1.0 = full immunity, 0.0 = none)
            ss.FloatArr(
                "partial_match_immunity_decay_factor", default=0.0
            ),  # Decay factor for partial matches (G, P, or G and P not homotypic)
            ss.FloatArr(
                "final_decayed_immunity_factor", default=0.0
            ),  # Reusable array for final decay factor calculations (to reduce allocations)
            ss.FloatArr("baseline_immunity", default=0.0),  # Permanent baseline immunity (e.g., for adults with childhood exposure history)
        )

        # Will be populated during init_post
        self.rota_diseases = []
        self.G_to_bit = {}
        self.P_to_bit = {}
        self.GP_to_bit = {}
        self.disease_G_masks = {}
        self.disease_P_masks = {}
        self.disease_GP_masks = {}

    def init_results(self):
        """Initialize results storage for immunity-related outputs"""
        super().init_results()

        return

    def init_pre(self, sim, force=False):
        """Auto-detect Rotavirus diseases and create states before initialization"""
        # Auto-detect all Rotavirus disease instances
        self.rota_diseases = [d for d in sim.diseases.values() if isinstance(d, Rotavirus)]

        if len(self.rota_diseases) == 0:
            if sim.pars.verbose:
                print("Warning: No Rotavirus diseases found in simulation")
            super().init_pre(sim)
            return

        if sim.pars.verbose:
            print(f"RotaImmunityConnector: Found {len(self.rota_diseases)} Rotavirus strains")

        # Calculate unique genotypes once and store for use in init_post()
        self.unique_G = sorted(set(d.G for d in self.rota_diseases))
        self.unique_P = sorted(set(d.P for d in self.rota_diseases))
        self.unique_GP = sorted(set((d.G, d.P) for d in self.rota_diseases))

        GP_states = []
        for gp in self.unique_GP:
            state_name = f"G{gp[0]}P{gp[1]}_decayed_immunity_factor"
            state = ss.FloatArr(state_name, default=0.0)
            GP_states.append(state)

        self.define_states(*GP_states)

        # NOW call super().init_pre() with states already defined
        super().init_pre(sim)

    def init_post(self):
        """Create bitmask mappings after initialization"""
        super().init_post()

        if len(self.rota_diseases) == 0:
            return

        # Use the genotypes calculated in init_pre() - no duplication!

        # Ensure we don't exceed bitwise limits based on actual datatype
        max_bits = np.iinfo(self.exposed_GP_bitmask.dtype).bits
        if len(self.unique_G) > max_bits or len(self.unique_P) > max_bits or len(self.unique_GP) > max_bits:
            raise ValueError(
                f"Too many unique genotypes: {len(self.unique_G)} G types, {len(self.unique_P)} P types, {len(self.unique_GP)} GP pairs. Max {max_bits} each. Either increase bitmask dtype size or reduce number of strains."
            )

        # Create mappings: genotype -> bit position
        self.G_to_bit = {g: i for i, g in enumerate(self.unique_G)}
        self.P_to_bit = {p: i for i, p in enumerate(self.unique_P)}
        self.GP_to_bit = {gp: i for i, gp in enumerate(self.unique_GP)}

        if self.sim.pars.verbose > 1:
            print(f"  - G genotypes: {self.unique_G} -> bits {list(self.G_to_bit.values())}")
            print(f"  - P genotypes: {self.unique_P} -> bits {list(self.P_to_bit.values())}")
            print(f"  - GP pairs: {self.unique_GP} -> bits {list(self.GP_to_bit.values())}")

        # Pre-compute disease-specific bitmasks for fast lookup
        self.disease_G_masks = {}
        self.disease_P_masks = {}
        self.disease_GP_masks = {}
        for disease in self.rota_diseases:
            self.disease_G_masks[disease.name] = 1 << self.G_to_bit[disease.G]
            self.disease_P_masks[disease.name] = 1 << self.P_to_bit[disease.P]
            self.disease_GP_masks[disease.name] = 1 << self.GP_to_bit[(disease.G, disease.P)]

        if self.sim.pars.verbose > 1:
            print(f"  - Pre-computed bitmasks for {len(self.rota_diseases)} diseases")

        # Init baseline immunity
        # NOTE: adult_baseline_immunity parameter has been removed
        # Baseline immunity is now initialized via initialize_immunity() method using baseline_immunity_exponential_rate
        # if self.pars['adult_baseline_immunity'] > 0:
        #     adults = self.sim.people.age >= self.pars['adult_age_threshold']
        #     self.baseline_immunity[adults] = self.pars['adult_baseline_immunity']

    def step(self):
        """Main connector step: apply waning and update cross-immunity"""
        if len(self.rota_diseases) == 0:
            return

        # Update age-dependent baseline immunity BEFORE calculating cross-immunity
        # This ensures all adults maintain the correct baseline immunity every timestep
        # if self.pars['adult_baseline_immunity'] > 0:
        #     adults = self.sim.people.age >= self.pars['adult_age_threshold']
        #     self.baseline_immunity[adults] = self.pars['adult_baseline_immunity']

        # Update cross-immunity protection for all diseases
        self._update_cross_immunity()

    def _update_cross_immunity(self):
        """Fully vectorized cross-immunity using bitwise operations - NO UID LOOPS"""
        # Reset decay factors and update immunity for all diseases
        self._reset_decay_factors()
        # self._update_immunity_decay_factors()
        self._calculate_disease_susceptibilities()

    def _reset_decay_factors(self):
        """Reset partial immunity factors to 0.0 each timestep for fresh calculation"""
        self.final_decayed_immunity_factor[:] = 0.0
        self.partial_match_immunity_decay_factor[:] = 0.0
        self.homotypic_immunity_decay_factor[:] = 0.0

    def _update_immunity_decay_factors(self):
        """Update immunity decay factors for all diseases based on recovery times"""
        for disease in self.rota_diseases:
            # Update max decay factors for agents recovered from this specific strain
            recovered_from_strain = (disease.infected == False) & (disease.n_infections > 0)
            recovered_uids = recovered_from_strain.uids
            infected_uids = (disease.infected == True).uids
            #
            # if recovered_from_strain.any():
            #     # Calculate time since recovery in days
            #     time_since_recovery = (disease.ti - disease.ti_recovered[recovered_from_strain]) * disease.dt.days
            #
            #     # Apply delayed exponential decay
            #     waning_started = time_since_recovery > disease.pars.waning_delay.days
            #
            #     if waning_started.any():
            #         waning_started_uids = recovered_uids[waning_started]
            #         # Calculate decay factor for agents past the delay period
            #         decay_time = time_since_recovery[waning_started] - self.pars.immunity_waning_delay.days
            #         # Use pre-computed decay rates stored when agents recovered
            #         decay_rate = disease.waning_rate[waning_started_uids]
            #         decay_factor = np.exp(
            #             -decay_rate * decay_time
            #         )  # decay_rate (1/days) * decay_time (days) = dimensionless
            #
            #         # Update per-strain decay factor (for homotypic immunity)
            #
            #         # todo these overwrite each other for each disease in the loop
            #         # self.homotypic_immunity_decay_factor[waning_started_uids] = decay_factor
            #         # self.homotypic_immunity_decay_factor[infected_uids] = 1
            #         # self[f"G{disease.G}P{disease.P}_decayed_immunity_factor"][waning_started_uids] = decay_factor

    def _calculate_disease_susceptibilities(self):
        """Calculate disease susceptibilities based on immunity matching and decay factors"""

        # Calculate baseline immunity using either fixed or exponential model
        if self.pars.use_fixed_susceptibility:
            # Fixed susceptibility model: discrete values based on infection count
            # Protection = 1 - susceptibility
            baseline_immunity = np.zeros_like(self.num_recovered_infections)
            baseline_immunity[self.num_recovered_infections == 1] = 1 - self.pars.sus_after_1  # 33% protection
            baseline_immunity[self.num_recovered_infections == 2] = 1 - self.pars.sus_after_2  # 50% protection
            baseline_immunity[self.num_recovered_infections >= 3] = 1 - self.pars.sus_after_3plus  # 64% protection
            # baseline_immunity[self.num_recovered_infections == 0] remains 0 (naive, no protection)
        else:
            # Exponential saturation model (original): continuous exponential growth approaching 1.0
            # OLD CODE (kept for reference):
            # long term baseline immunity increases with number of exposures
            # todo: add a waning term based on most recent exposure
            baseline_immunity = (1- np.exp(
                -self.pars.baseline_immunity_exponential_rate * self.num_recovered_infections
            ))

        combined_immunity_factor = baseline_immunity


        for disease in self.rota_diseases:
            # Reset final_decayed_immunity_factor to 0.0 at the start of each disease loop
            # This ensures stale values from previous disease calculations don't persist
            # self.final_decayed_immunity_factor[:] = 0.0

            # store a list of all partial hetero matches for this strain
            disease_partial_matches = []
            # for gp in self.unique_GP:
            for match_disease in self.rota_diseases:
                if (match_disease.G == disease.G) ^ (match_disease.P == disease.P):
                    disease_partial_matches.append(match_disease)

            disease_G_mask = self.disease_G_masks[disease.name]
            disease_P_mask = self.disease_P_masks[disease.name]
            disease_GP_mask = self.disease_GP_masks[disease.name]

            # Extract raw numpy arrays for bitwise operations
            G_bits = self.exposed_G_bitmask.values
            P_bits = self.exposed_P_bitmask.values
            GP_bits = self.exposed_GP_bitmask.values

            # Vectorized matching using bitwise operations
            has_exact_match = (GP_bits & disease_GP_mask) != 0
            has_G_match = ((G_bits & disease_G_mask) != 0) & ~has_exact_match
            has_P_match = ((P_bits & disease_P_mask) != 0) & ~has_exact_match

            # Determine immunity type and assign protection levels
            has_partial_match = has_G_match | has_P_match
            # has_immunity_mask = self.has_immunity[:]  # Agents with any prior immunity

            # Separate naive agents (no prior immunity) from true heterotypic matches
            has_complete_hetero = ~has_partial_match & ~has_exact_match & self.has_immunity


            # combined_immunity_factor = np.zeros(has_exact_match.shape, dtype=float)

            # strain match immunity efficacy reduces rel_sus by this factor.
            strain_match_immunity_efficacy = np.zeros(has_exact_match.shape, dtype=float)
            strain_match_immunity_efficacy[has_exact_match] = self.pars.homotypic_immunity_efficacy
            strain_match_immunity_efficacy[has_partial_match] = self.pars.partial_heterotypic_immunity_efficacy
            strain_match_immunity_efficacy[has_complete_hetero] = self.pars.complete_heterotypic_immunity_efficacy
            strain_match_immunity_efficacy[~self.has_immunity] = self.pars.naive_immunity_efficacy

            # Homotypic: use per-strain decay
            # self.final_decayed_immunity_factor[has_exact_match] = np.maximum(self.final_decayed_immunity_factor[has_exact_match], disease.waned_immunity_efficacy[has_exact_match])
            combined_immunity_factor[has_exact_match] = np.maximum(combined_immunity_factor[has_exact_match], disease.waned_immunity_efficacy[has_exact_match])

            # self.final_decayed_immunity_factor[has_exact_match] = np.minimum(self.final_decayed_immunity_factor[has_exact_match], disease.waned_immunity_efficacy[has_exact_match])



            # self.final_decayed_immunity_factor[has_exact_match] = strain_match_immunity_efficacy
            # test_immunity_factor = disease.waned_immunity_efficacy[has_exact_match]

            for partial_match_disease in disease_partial_matches:
                # gp_decay = self[f"G{gp[0]}P{gp[1]}_decayed_immunity_factor"][has_partial_match]
                #
                # Update partial match decay factor to the maximum of any matching G or P type
                # self.final_decayed_immunity_factor[has_partial_match] = np.maximum(
                #     self.final_decayed_immunity_factor[has_partial_match], gp_decay
                # )
                combined_immunity_factor[has_partial_match] = np.maximum(combined_immunity_factor[has_partial_match], partial_match_disease.waned_immunity_efficacy[has_partial_match])

            # Note: Complete heterotypic and naive agents keep decay_factor=0.0 (from reset above)
            # This works correctly because:
            # - For naive: efficacy=0.0, so protection = 0.0 * 0.0 = 0.0 (no protection, full susceptibility)
            # - For complete hetero: efficacy>0, but decay_factor=0.0 means immunity has fully decayed (no protection)

            # Apply protection with appropriate decay factor
            # People without immunity have full susceptibility (rel_sus = 1.0)

            # The full rel_sus calculation combines strain match efficacy, decayed immunity factor, and infection history scaling.
            # * strain_match_immunity_efficacy is the base protection level based on match type (0.0 to 1.0)
            # * final_decay_factor reduces this protection over time since last infection (0.0 to 1.0). In the case of a partial match, it uses the max decay from either G or P.
            # * infection_history_susceptibility_factor scales susceptibility based on total prior infections. It does not decay over time.

            # Calculate acquired immunity protection
            # acquired_immunity_protection = strain_match_immunity_efficacy * self.final_decayed_immunity_factor
            # acquired_immunity_protection = np.maximum(strain_match_immunity_efficacy, self.final_decayed_immunity_factor.values)

            # Acquired immunity from prior infections (strain-match-weighted).
            acquired_protection = combined_immunity_factor * strain_match_immunity_efficacy

            # Maternal immunity: protection = efficacy * exp(-ln(2) * age_years / half_life_years).
            # Applied to all agents but only meaningful for infants -- decays to ~0 by age ~1y
            # with default 90-day half-life. Combined with acquired via element-wise max.
            mat_eff = float(self.pars.maternal_immunity_efficacy)
            if mat_eff > 0:
                mat_hl_years = self.pars.maternal_immunity_half_life.years
                agent_ages_years = self.sim.people.age.values
                maternal_decay = np.exp(-np.log(2) * agent_ages_years / mat_hl_years)
                maternal_protection = mat_eff * maternal_decay
                total_protection = np.maximum(acquired_protection, maternal_protection)
            else:
                total_protection = acquired_protection

            disease.rel_sus[:] = 1 - total_protection

            # Override susceptibility for long-term immune agents (cannot be reinfected)
            # DISABLED: Removed LTI mechanism per user request to implement simple SIRS model
            # disease.rel_sus[self.long_term_immune[:]] = 0.0

    def record_infection(self, disease, new_infected_uids):
        self.num_current_infections[new_infected_uids] += 1.0

    def record_recovery(self, disease, recovered_uids):
        """
        Update bitmasks when people recover from infections

        This method should be called by Rotavirus instances when infections resolve.

        Args:
            disease: Rotavirus disease instance
            recovered_uids: Array of UIDs who recovered from this disease
        """
        if len(recovered_uids) == 0:
            return

        if not isinstance(disease, Rotavirus):
            return  # Only handle Rotavirus diseases

        # Get bit positions for this disease's G,P genotypes and combination
        G_bit = 1 << self.G_to_bit[disease.G]
        P_bit = 1 << self.P_to_bit[disease.P]
        GP_bit = 1 << self.GP_to_bit[(disease.G, disease.P)]

        # Update bitmasks using IntArr bitwise ops
        current_G = self.exposed_G_bitmask[recovered_uids]
        current_P = self.exposed_P_bitmask[recovered_uids]
        current_GP = self.exposed_GP_bitmask[recovered_uids]

        self.exposed_G_bitmask[recovered_uids] = current_G | G_bit
        self.exposed_P_bitmask[recovered_uids] = current_P | P_bit
        self.exposed_GP_bitmask[recovered_uids] = current_GP | GP_bit

        # Mark as having immunity
        self.has_immunity[recovered_uids] = True

        # Decrement current infection count
        self.num_current_infections[recovered_uids] -= 1.0

        # Only increment recovered infections when ALL concurrent infections have resolved
        # This ensures co-infections or sequential infections count as ONE episode
        completed_episode = self.num_current_infections[recovered_uids] == 0
        episode_complete_uids = recovered_uids[completed_episode]

        if len(episode_complete_uids) > 0:
            self.num_recovered_infections[episode_complete_uids] += 1.0

            # Track oldest infection time (only set if first infection episode)
            first_infections = np.isnan(self.oldest_infection[episode_complete_uids])
            self.oldest_infection[episode_complete_uids[first_infections]] = self.sim.ti

        # DISABLED: Removed LTI mechanism per user request to implement simple SIRS model
        # Probabilistically assign long-term immunity based on infection EPISODES (not individual strains)
        # Only check for LTI when an infection episode completes
        # if len(episode_complete_uids) > 0:
        #     infection_counts = self.num_recovered_infections[episode_complete_uids]
        #
        #     # Determine probability for each agent based on their total infection episode count
        #     probs = np.zeros(len(episode_complete_uids))
        #     probs[infection_counts == 1] = self.pars.long_term_immunity_prob_after_1
        #     probs[infection_counts == 2] = self.pars.long_term_immunity_prob_after_2
        #     probs[infection_counts == 3] = self.pars.long_term_immunity_prob_after_3
        #     probs[infection_counts >= 4] = self.pars.long_term_immunity_prob_after_4
        #
        #     # Randomly assign long-term immunity
        #     develops_long_term = np.random.rand(len(episode_complete_uids)) < probs
        #     newly_immune_uids = episode_complete_uids[develops_long_term]
        #     self.long_term_immune[newly_immune_uids] = True
        #
        #     # Store the age at which agents developed long-term immunity
        #     if len(newly_immune_uids) > 0:
        #         self.long_term_immune_age[newly_immune_uids] = self.sim.people.age[newly_immune_uids]

    def initialize_immunity(self, min_age, max_age, min_exposures, max_exposures, exposures_per_year = 1.5):
        """
            Initialize immunity reflecting prior infections

            Most adults have experienced multiple rotavirus infections in childhood
            and have built up cumulative immunity. This function initializes immunity reflecting prior infections

            NOTE: This currently assigns the same number of exposures to all circulating strains. This is not realistic
            in environments with many strains circulating.


            Args:
                min_age: Minimum age of population to initialize immunity reflecting prior infections
                max_age: Maximum age of population to initialize immunity reflecting prior infections
                min_exposures: Minimum potential prior infections
                max_exposures: Maximum potential prior infections
            """
        import numpy as np

        eligible_uids = ((self.sim.people.age >= min_age) & (self.sim.people.age < max_age)).uids
        n_uids = len(eligible_uids)

        if n_uids == 0:
            return  # No pop to initialize

        self.pars.immunity_init_dist.set(low=min_exposures, high=max_exposures)

        # Record infection history for tracking purposes
        # self.num_recovered_infections[eligible_uids] = self.pars.immunity_init_dist.rvs(eligible_uids)

        # num_exposures_per_strain = self.pars.immunity_init_dist.rvs(eligible_uids)
        # exposure history is now based on age and expected number of exposures per year, divided by the number of strains.
        num_exposures_per_strain = np.round(exposures_per_year * self.sim.people.age[eligible_uids], 0)/len(self.rota_diseases)
        self.has_immunity[eligible_uids] = True

        # For each eligible agent, distribute the number of recovered infections across the different strains

        # Set bitmasks indicating prior exposure to circulating strains
        for disease in self.sim.diseases.values():
            if isinstance(disease, Rotavirus):
                G_bit = 1 << self.G_to_bit[disease.G]
                P_bit = 1 << self.P_to_bit[disease.P]
                GP_bit = 1 << self.GP_to_bit[(disease.G, disease.P)]

                # Update bitmasks using IntArr bitwise ops
                # self.exposed_G_bitmask[eligible_uids] = self.exposed_G_bitmask[eligible_uids] | G_bit
                # self.exposed_P_bitmask[eligible_uids] = self.exposed_P_bitmask[eligible_uids] | P_bit
                # self.exposed_GP_bitmask[eligible_uids] = self.exposed_GP_bitmask[eligible_uids] | GP_bit
                #
                # disease.n_infections[eligible_uids] = num_exposures_per_strain
                self.num_recovered_infections[eligible_uids] += num_exposures_per_strain


        # Calculate baseline immunity based on number of prior exposures
        # Each exposure provides incremental immunity (like acquired immunity but permanent)
        # Use homotypic efficacy as the per-exposure protection (cumulative effect)
        # for uid in eligible_uids:
        #     num_exp = self.num_recovered_infections[uid]
            # Cumulative protection: each exposure adds homotypic_immunity_efficacy
            # Capped at adult_baseline_immunity as the maximum achievable protection
            # cumulative_protection = min(
            #     self.pars.adult_baseline_immunity,
            #     num_exp * self.pars.homotypic_immunity_efficacy
            # )
            # self.baseline_immunity[uid] = cumulative_protection
            # self.baseline_immunity[uid] =

        # self.baseline_immunity[eligible_uids] = self.pars.adult_baseline_immunity

        # if self.sim.pars.verbose:
            # print(f"\n✓ Initialized {n_uids} agents with baseline immunity:")
            # print(f"  Prior infections: {min_exposures}-{max_exposures}")
            # print(f"  Cumulative protection: {self.pars.adult_baseline_immunity * 100:.1f}% (from repeated prior exposures)")
            # print(f"  Note: This baseline doesn't wane - new infections add temporary immunity on top")

        # Update rel_sus immediately after setting baseline_immunity
        # This ensures immunity is applied before the simulation starts
        self._calculate_disease_susceptibilities()
