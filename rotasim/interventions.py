"""
Rotavirus vaccination intervention for multi-strain simulations
"""
# Standard library imports
from typing import Union, List, Optional, Dict

# Third-party imports
import numpy as np
import starsim as ss
from .rotavirus import Rotavirus


class RotaVaccination(ss.Intervention):
    """
    Rotavirus vaccination intervention with multi-dose schedule and strain-specific protection
    
    This intervention implements rotavirus vaccination with the following features:
    - Multi-dose vaccination schedule (1-3 doses typical)
    - Age-based eligibility criteria
    - Probability-based uptake
    - G and P antigen-specific protection with cross-protection
    - Dose-specific vaccine effectiveness
    - Cross-strain protection (homotypic, partial heterotypic, complete heterotypic)
    - Immunity waning reset for protected strains
    
    The vaccination works by modifying the rel_sus parameter for covered strains, providing
    protection that wanes over time. The intervention tracks its own waning protection factor
    and applies it to rel_sus only when it provides better protection than existing immunity.
    
    Cross-protection efficacies are precomputed during initialization for optimal performance.
    
    Args:
        start_date (str or ss.date): Date to start vaccination program
        end_date (str or ss.date, optional): Date to end vaccination program (None = continue indefinitely)
        n_doses (int): Number of doses in vaccination schedule (default: 2)
        dose_interval (int or ss.days): Time between doses in days (default: ss.days(28))
        G_antigens (list): List of G genotypes covered by vaccine (default: [1])
        P_antigens (list): List of P genotypes covered by vaccine (default: [8])
        dose_effectiveness (list or dict): Effectiveness by dose number (default: [0.6, 0.8] for 2 doses)
        min_age (int or ss.days): Minimum age for vaccination (default: ss.days(42) = 6 weeks)
        max_age (int or ss.days): Maximum age for vaccination (default: ss.days(365) = 1 year)
        uptake_prob (float): Probability that eligible agents receive vaccine (default: 0.8)
        waning_rate_dist (ss.Dist): Distribution for vaccine waning time (default: ss.lognorm_ex(mean=365))
        homotypic_efficacy (float): Efficacy multiplier for exact G+P matches (default: 1.0)
        partial_heterotypic_efficacy (float): Efficacy multiplier for shared G or P (default: 0.6)
        complete_heterotypic_efficacy (float): Efficacy multiplier for no shared G,P (default: 0.3)
        verbose (bool): Print vaccination events (default: False)
    
    Examples:
        # Simple 2-dose G1P8 vaccination
        vax = RotaVaccination(
            start_date='2025-01-01',
            G_antigens=[1],
            P_antigens=[8]
        )
        
        # Multi-strain vaccine (pentavalent-like)
        vax = RotaVaccination(
            start_date='2025-01-01',
            n_doses=3,
            dose_interval=ss.days(28),
            G_antigens=[1, 2, 3, 4],
            P_antigens=[8, 4, 6],
            dose_effectiveness=[0.5, 0.7, 0.85],
            uptake_prob=0.9
        )
        
        # Limited-time vaccination campaign
        vax = RotaVaccination(
            start_date='2025-01-01',
            end_date='2027-12-31',
            G_antigens=[1, 2],
            P_antigens=[8, 4],
            uptake_prob=0.6
        )
    """
    
    def __init__(self, start_date, end_date=None, n_doses=2, dose_interval=None,
                 G_antigens=None, P_antigens=None, dose_effectiveness=None,
                 min_age=None, max_age=None, uptake_prob=0.8,
                 waning_rate_dist=None, waning_delay=None, homotypic_efficacy=1.0,
                 partial_heterotypic_efficacy=0.6, complete_heterotypic_efficacy=0.3,
                 verbose=False, **kwargs):
        
        super().__init__(**kwargs)
        
        # Set defaults
        if dose_interval is None:
            dose_interval = ss.days(28)  # 4 weeks between doses
        if G_antigens is None:
            G_antigens = [1]  # G1 by default
        if P_antigens is None:
            P_antigens = [8]  # P8 by default
        if min_age is None:
            min_age = ss.days(42)  # 6 weeks
        if max_age is None:
            max_age = ss.days(365)  # 1 year
        if dose_effectiveness is None:
            if n_doses == 1:
                dose_effectiveness = [0.7]
            elif n_doses == 2:
                dose_effectiveness = [0.6, 0.8]
            elif n_doses == 3:
                dose_effectiveness = [0.5, 0.7, 0.85]
            else:
                # Linear increase for more doses
                dose_effectiveness = [0.4 + (0.4 * i / (n_doses - 1)) for i in range(n_doses)]
        
        # Store parameters
        self.start_date = start_date
        self.end_date = end_date
        self.n_doses = int(n_doses)
        self.dose_interval = dose_interval
        self.G_antigens = list(G_antigens)
        self.P_antigens = list(P_antigens)
        self.dose_effectiveness = list(dose_effectiveness)
        self.min_age = min_age
        self.max_age = max_age
        self.uptake_prob = ss.bernoulli(p = uptake_prob)
        self.verbose = bool(verbose)
        
        # Validation
        if len(self.dose_effectiveness) != self.n_doses:
            raise ValueError(f"dose_effectiveness must have {self.n_doses} values, got {len(self.dose_effectiveness)}")
        
        if not (0 <= uptake_prob <= 1):
            raise ValueError(f"uptake_prob must be between 0 and 1, got {uptake_prob}")
            
        if self.n_doses < 1 or self.n_doses > 10:
            raise ValueError(f"n_doses must be between 1 and 10, got {self.n_doses}")
        
        # Cross-protection efficacy parameters (matching immunity.py pattern)
        self.homotypic_efficacy = float(homotypic_efficacy)
        self.partial_heterotypic_efficacy = float(partial_heterotypic_efficacy)
        self.complete_heterotypic_efficacy = float(complete_heterotypic_efficacy)
        
        # Validation for cross-protection parameters
        if not (0 <= self.homotypic_efficacy <= 1):
            raise ValueError(f"homotypic_efficacy must be between 0 and 1, got {self.homotypic_efficacy}")
        if not (0 <= self.partial_heterotypic_efficacy <= 1):
            raise ValueError(f"partial_heterotypic_efficacy must be between 0 and 1, got {self.partial_heterotypic_efficacy}")
        if not (0 <= self.complete_heterotypic_efficacy <= 1):
            raise ValueError(f"complete_heterotypic_efficacy must be between 0 and 1, got {self.complete_heterotypic_efficacy}")
        
        # Vaccine waning parameters
        if waning_rate_dist is None:
            self.waning_rate_dist = ss.lognorm_ex(mean=365)  # Default: 1 year mean waning time
        elif isinstance(waning_rate_dist, ss.Dist):
            self.waning_rate_dist = waning_rate_dist
        elif np.isscalar(waning_rate_dist):
            self.waning_rate_dist = ss.constant(waning_rate_dist)
        else:
            raise ValueError("waning_rate_dist must be an ss.Dist or a scalar value")

        if waning_delay is None:
            self.waning_delay = 0
        
        # Define states for vaccination tracking
        self.define_states(
            ss.IntArr('doses_received', default=0),  # Number of doses received
            ss.IntArr('doses_eligible', default=0),  # Number of doses agent has been eligible for
            ss.FloatArr('last_dose_time', default=-np.inf),  # Time of last dose
            ss.FloatArr('next_dose_due', default=-np.inf),  # When next dose is due
            ss.BoolArr('completed_schedule', default=False),  # Whether completed all doses
        )
        
    def init_pre(self, sim):
        """Initialize vaccination state tracking"""
        # Convert dates to simulation time indices
        start_date_obj = ss.date(self.start_date)
        # Find first timevec index greater than start date
        self.start_ti = np.argmax(sim.t.yearvec > start_date_obj.years)
        
        if self.end_date is not None:
            end_date_obj = ss.date(self.end_date)
            # Find last timevec index less than or equal to end date
            valid_indices = np.where(sim.t.yearvec <= end_date_obj.years)[0]
            self.end_ti = valid_indices[-1] if len(valid_indices) > 0 else 0
        else:
            self.end_ti = np.inf
        
        # Find rotavirus diseases
        self.rotavirus_diseases = []
        for disease in sim.diseases.values():
            if isinstance(disease, Rotavirus):
                self.rotavirus_diseases.append(disease)
                
        if not self.rotavirus_diseases:
            raise RuntimeError("No Rotavirus diseases found in simulation")
        
        # With cross-protection, ALL rotavirus diseases are covered (with different efficacy levels)
        self.covered_diseases = self.rotavirus_diseases.copy()
        self.vaccine_protections = {}  # Maps disease name to protection level state
        self.vaccine_waning_delays = {}      # Maps disease name to waning time state
        
        # Precompute match efficacies for all diseases (performance optimization). These are based on the antigen match types.
        self.disease_match_efficacies = {}
        for disease in self.covered_diseases:
            self.disease_match_efficacies[disease.name] = self._compute_match_efficacy(disease)
        
        # Create dynamic states for covered diseases BEFORE calling super().init_pre()
        dynamic_states = []
        
        for disease in self.covered_diseases:
                
                # Create individual states for this disease's vaccine protection
                protection_state_name = f'vax_protection_{disease.name}'
                # waning_state_name = f'vax_waning_{disease.name}'
                
                # Create state objects
                protection_state = ss.FloatArr(protection_state_name, default=0.0)
                # waning_state = ss.FloatArr(waning_state_name, default=-np.inf)
                
                # Store references for easy access
                self.vaccine_protection_states[disease.name] = protection_state
                # self.vaccine_waning_delays[disease.name] = waning_state
                
                # Add to dynamic states list
                dynamic_states.append(protection_state)
                dynamic_states.append(waning_state)
        
        # Define all dynamic states at once
        if dynamic_states:
            self.define_states(*dynamic_states)
        
        # NOW call super().init_pre() with states already defined
        super().init_pre(sim)
        
        # Print initialization summary if verbose
        if self.verbose:
            print(f"RotaVaccination initialized:")
            print(f"  Start: {self.start_date} (ti={self.start_ti})")
            print(f"  End: {self.end_date} (ti={self.end_ti})")
            print(f"  Doses: {self.n_doses}")
            print(f"  Interval: {self.dose_interval} days")
            print(f"  G antigens: {self.G_antigens}")
            print(f"  P antigens: {self.P_antigens}")
            print(f"  Effectiveness: {self.dose_effectiveness}")
            print(f"  Age range: {self.min_age}-{self.max_age} days")
            print(f"  Uptake: {self.uptake_prob}")
            print(f"  Covered diseases: {len(self.covered_diseases)}")
            if self.covered_diseases:
                covered_strains = [(d.G, d.P) for d in self.covered_diseases]
                print(f"  Covered strains: {covered_strains}")
                
    def check_eligibility(self):
        """
        Check which agents are eligible for vaccination
        
        Returns:
            np.array: Boolean array of agent eligibility
        """
        sim = self.sim
        # Check if intervention is active
        if sim.ti < self.start_ti or sim.ti > self.end_ti:
            return np.array([False] * len(sim.people))
            
        people = sim.people
        
        # Age eligibility
        age_eligible = (people.age >= self.min_age.value) & (people.age <= self.max_age.value)

        # Exclude those who completed the schedule
        age_eligible = age_eligible & ~self.completed_schedule
        
        # For multi-dose: check if next dose is due
        next_dose_due = (self.doses_received < self.n_doses) & (self.ti >= self.next_dose_due)
        first_dose = (self.doses_received == 0)
        
        age_eligible = age_eligible & (first_dose | next_dose_due)
        
        return age_eligible
        
    def step(self):
        """Apply vaccination and update vaccine protection at current timestep"""
        # First, update vaccine protection for all agents (waning)
        self._update_vaccine_protection()
        
        # Then, check for new vaccinations
        eligible_agents = self.check_eligibility()
        eligible_uids = self.sim.people.uid[eligible_agents]
        
        if len(eligible_uids) > 0:
            # Track eligibility: increment doses_eligible and update next_dose_due for all eligible agents
            self.doses_eligible[eligible_uids] += 1
            
            # Update next_dose_due for agents who will need more doses
            still_need_doses = self.doses_eligible[eligible_uids] < self.n_doses
            self.next_dose_due[eligible_uids] = np.where(
                still_need_doses,
                self.ti + self.dose_interval.value,
                self.next_dose_due[eligible_uids]  # Keep existing value if no more doses needed
            )
            
            # Random uptake
            # uptake = self.uptake_prob.rvs(eligible_uids)
            # vaccinated_uids = eligible_uids[uptake]
            vaccinated_uids = self.uptake_prob.filter(eligible_uids)
            
            if len(vaccinated_uids) > 0:
                # Apply vaccination (vectorized)
                self._vaccinate_agents(vaccinated_uids)
                
                if self.verbose:
                    total_eligible = np.sum(eligible_agents)
                    print(f"Day {self.sim.ti}: Vaccinated {len(vaccinated_uids)}/{total_eligible} eligible agents")
            
    def _vaccinate_agents(self, uids):
        """Vaccinate multiple agents at once (vectorized)"""
        if len(uids) == 0:
            return

        sim = self.sim
            
        # Get current dose numbers for all agents being vaccinated
        current_doses = self.doses_received[uids]
        
        # Update vaccination tracking (vectorized)
        self.doses_received[uids] += 1
        self.last_dose_time[uids] = self.ti
        
        # Mark completed schedules
        completed_mask = self.doses_received[uids] >= self.n_doses
        self.completed_schedule[uids] = np.where(completed_mask, True, self.completed_schedule[uids])
        
        # Apply vaccine protection to covered diseases
        self._apply_vaccine_protection(sim, uids, current_doses)
            
        if self.verbose:
            # Group by dose number for cleaner output
            dose_counts = {}
            for i, uid in enumerate(uids):
                dose_num = current_doses[i]  # Get dose number for this agent
                dose_display = dose_num + 1  # 1-indexed for display
                effectiveness = self.dose_effectiveness[dose_num]
                if dose_display not in dose_counts:
                    dose_counts[dose_display] = {'count': 0, 'effectiveness': effectiveness}
                dose_counts[dose_display]['count'] += 1
            
            for dose_display, info in sorted(dose_counts.items()):
                print(f"  Dose {dose_display}/{self.n_doses}: {info['count']} agents (effectiveness={info['effectiveness']:.1%})")
            
    def _is_homotypic_match(self, disease):
        """Check if disease strain has exact G+P match with vaccine"""
        return disease.G in self.G_antigens and disease.P in self.P_antigens
    
    def _is_partial_heterotypic_match(self, disease):
        """Check if disease strain has partial match (shared G or P) with vaccine"""
        return (disease.G in self.G_antigens or disease.P in self.P_antigens) and not self._is_homotypic_match(disease)
    
    def _is_complete_heterotypic_match(self, disease):
        """Check if disease strain has no match with vaccine"""
        return not (disease.G in self.G_antigens or disease.P in self.P_antigens)
    
    def _compute_match_efficacy(self, disease):
        """Compute and return match efficacy for a disease (called once during initialization)"""
        if self._is_homotypic_match(disease):
            return self.homotypic_efficacy
        elif self._is_partial_heterotypic_match(disease):
            return self.partial_heterotypic_efficacy
        else:  # complete heterotypic
            return self.complete_heterotypic_efficacy

    def _apply_vaccine_protection(self, uids, current_doses):
        """
        Apply vaccine protection by updating vaccine states (vectorized)
        
        Updates the vaccine protection level and waning time for each covered disease.
        The protection will later be applied to rel_sus in _update_vaccine_protection.
        """
        if len(uids) == 0:
            return
            
        # Get effectiveness values for each agent based on their current dose number
        dose_efficacy = self.dose_effectiveness[current_doses]
        
        # Sample waning times for each agent
        if hasattr(self.waning_rate_dist, 'rvs'): # use the disease's regular waning rate by default
            waning_rate_denoms = self.waning_rate_dist.rvs(uids)
        # else:
        #     waning_rate_denoms = np.full(len(uids), 365)  # Default fallback
        new_waning_times = self.ti + waning_rate_denoms
        
        # Update vaccine protection states for each covered disease
        for disease in self.covered_diseases:
            protection_state = self.vaccine_protection_states[disease.name]
            waning_state = self.vaccine_waning_delays[disease.name]
            
            # Use precomputed match efficacy for this disease (performance optimization)
            match_efficacy = self.disease_match_efficacies[disease.name]
            
            # Calculate disease-specific effectiveness (dose effectiveness * match efficacy)
            disease_effectiveness = dose_efficacy * match_efficacy
            
            # Update protection level (takes maximum of current and new protection)
            current_protection = protection_state[ss.uids(uids)]
            new_protection = np.maximum(current_protection, disease_effectiveness)
            protection_state[ss.uids(uids)] = new_protection
            
            # Update waning time (set to latest waning time)
            waning_state[ss.uids(uids)] = new_waning_times
                        
        if self.verbose:
            covered_strains = [(d.G, d.P) for d in self.covered_diseases]
            print(f"    Applied protection to {len(uids)} agents against strains: {covered_strains}")
            
    def _update_vaccine_protection(self):
        """
        Update vaccine protection levels due to waning and apply to rel_sus parameters
        
        This method:
        1. Updates waned vaccine protection levels
        2. Compares with existing rel_sus values
        3. Applies the most protective value to rel_sus
        """
        if len(self.covered_diseases) == 0:
            return
            
        # Update waned protection for each covered disease
        for disease in self.covered_diseases:
            protection_state = self.vaccine_protection_states[disease.name]
            waning_state = self.vaccine_waning_delays[disease.name]
            
            # Calculate current protection level based on waning
            time_since_vaccination = self.ti - waning_state
            waned_protection = np.where(
                waning_state > self.ti,  # Protection hasn't started waning yet
                protection_state,
                np.maximum(0.0, protection_state * np.exp(-time_since_vaccination / 365))  # Exponential decay
            )
            
            # Update the protection state with waned values
            protection_state[:] = waned_protection
            
            # Apply to disease rel_sus if vaccine protection is better

            # Calculate vaccine susceptibility (inverse of protection)
            vaccine_sus = 1.0 - waned_protection

            # Use the minimum susceptibility (most protective)
            current_sus = disease.rel_sus[:]
            new_sus = np.minimum(current_sus, vaccine_sus)
            disease.rel_sus[:] = new_sus
            

    def get_vaccination_summary(self):
        """Get summary of vaccination program status"""
        total_agents = len(self.doses_received)
        
        summary = {
            'total_agents': total_agents,
            'doses_eligible': np.count_nonzero(self.doses_eligible > 0),
            'received_any_dose': np.count_nonzero(self.doses_received > 0),
            'completed_schedule': np.count_nonzero(self.completed_schedule),
            'doses_by_number': {},
            'mean_doses': np.mean(self.doses_received[self.doses_received > 0]) if np.any(self.doses_received > 0) else 0
        }
        
        # Count agents by dose number
        for dose_num in range(0, self.n_doses + 1):
            summary['doses_by_number'][dose_num] = np.count_nonzero(self.doses_received == dose_num)
            
        return summary
        
    def print_vaccination_summary(self):
        """Print vaccination program summary"""
        summary = self.get_vaccination_summary()
        
        print(f"\n=== RotaVaccination Summary ===")
        print(f"Total agents: {summary['total_agents']:,}")
        print(f"Ever eligible: {summary['doses_eligible']:,} ({100*summary['doses_eligible']/summary['total_agents']:.1f}%)")
        print(f"Received any dose: {summary['received_any_dose']:,} ({100*summary['received_any_dose']/summary['total_agents']:.1f}%)")
        print(f"Completed schedule: {summary['completed_schedule']:,} ({100*summary['completed_schedule']/summary['total_agents']:.1f}%)")
        print(f"Mean doses (among vaccinated): {summary['mean_doses']:.2f}")
        
        print(f"\nDoses received:")
        for dose_num in range(1, self.n_doses + 1):
            count = summary['doses_by_number'][dose_num]
            pct = 100 * count / summary['total_agents']
            print(f"  Dose {dose_num}: {count:,} ({pct:.1f}%)")
            
        print(f"\nVaccine coverage:")
        print(f"  G antigens: {self.G_antigens}")
        print(f"  P antigens: {self.P_antigens}")
        print(f"  Effectiveness by dose: {self.dose_effectiveness}")


# Legacy alias for backward compatibility
RotaVax = RotaVaccination