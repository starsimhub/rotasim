"""
Rotavirus vaccination intervention for multi-strain simulations
"""

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
        uptake_dist (ss.Dist): Distribution that eligible agents receive vaccine (ss.bernoulli(p=0.8))
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
            uptake_dist=ss.bernoulli(0.9)
        )

        # Limited-time vaccination campaign
        vax = RotaVaccination(
            start_date='2025-01-01',
            end_date='2027-12-31',
            G_antigens=[1, 2],
            P_antigens=[8, 4],
            uptake_dist=ss.bernoulli(0.6)
        )
    """

    def __init__(
        self,
        start_date,
        end_date=None,
        n_doses=2,
        dose_interval=None,
        G_antigens=[1],
        P_antigens=[8],
        dose_effectiveness=None,
        min_age=ss.days(42),
        max_age=ss.days(365),
        uptake_dist=ss.bernoulli(0.8),
        waning_rate_dist=ss.lognorm_ex(mean=365),  # Default: 1 year mean waning time
        waning_delay=ss.days(0),
        homotypic_efficacy=1.0,
        partial_heterotypic_efficacy=0.6,
        complete_heterotypic_efficacy=0.3,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.define_pars(
            start_date=start_date,
            end_date=end_date,
            n_doses=n_doses,
            dose_interval=dose_interval,
            G_antigens=G_antigens,
            P_antigens=P_antigens,
            dose_effectiveness=dose_effectiveness,
            min_age=min_age,
            max_age=max_age,
            uptake_dist=uptake_dist,
            waning_rate_dist=waning_rate_dist,
            waning_delay=waning_delay,
            homotypic_efficacy=homotypic_efficacy,
            partial_heterotypic_efficacy=partial_heterotypic_efficacy,
            complete_heterotypic_efficacy=complete_heterotypic_efficacy,
            verbose=verbose,
        )

        self.update_pars(**kwargs)

        # Set defaults
        if dose_interval is None:
            self.pars.dose_interval = ss.days(28)  # 4 weeks between doses

        if dose_effectiveness is None:
            if n_doses == 1:
                self.pars.dose_effectiveness = [0.7]
            elif n_doses == 2:
                self.pars.dose_effectiveness = [0.6, 0.8]
            elif n_doses == 3:
                self.pars.dose_effectiveness = [0.5, 0.7, 0.85]
            else:
                # Linear increase for more doses
                self.pars.dose_effectiveness = [0.4 + (0.4 * i / (n_doses - 1)) for i in range(n_doses)]

        # Store parameters
        if self.pars.uptake_dist is None or np.isscalar(self.pars.uptake_dist):
            uptake_prob = float(self.pars.uptake_dist) if self.pars.uptake_dist is not None else 0.8
            self.pars.uptake_dist = ss.bernoulli(p=uptake_prob)

        # Validation
        if len(self.pars.dose_effectiveness) != self.pars.n_doses:
            raise ValueError(
                f"dose_effectiveness must have {self.pars.n_doses} values, got {len(self.pars.dose_effectiveness)}"
            )

        if self.pars.n_doses < 1:
            raise ValueError(f"n_doses must be between >= 1, got {self.pars.n_doses}")

        # Validation for cross-protection parameters
        if not (0 <= self.pars.homotypic_efficacy <= 1):
            raise ValueError(f"homotypic_efficacy must be between 0 and 1, got {self.pars.homotypic_efficacy}")
        if not (0 <= self.pars.partial_heterotypic_efficacy <= 1):
            raise ValueError(
                f"partial_heterotypic_efficacy must be between 0 and 1, got {self.pars.partial_heterotypic_efficacy}"
            )
        if not (0 <= self.pars.complete_heterotypic_efficacy <= 1):
            raise ValueError(
                f"complete_heterotypic_efficacy must be between 0 and 1, got {self.pars.complete_heterotypic_efficacy}"
            )

        # Vaccine waning parameters
        if not isinstance(waning_rate_dist, ss.Dist):
            raise ValueError("waning_rate_dist must be an ss.Dist")

        # Define states for vaccination tracking
        self.define_states(
            ss.IntArr("doses_received", default=0),  # Number of doses received
            ss.IntArr("doses_eligible", default=0),  # Number of doses agent has been eligible for
            ss.FloatArr("last_dose_ti", default=-np.inf),  # Time of last dose
            ss.FloatArr("next_dose_due", default=-np.inf),  # When next dose is due
            ss.BoolArr(
                "completed_schedule",
                default=False,
            ),  # Whether completed all doses
            ss.FloatArr("waning_rate", default=0.0),
            ss.FloatArr("waning_delay", default=0.0),
        )

    def init_pre(self, sim):
        """Initialize vaccination state tracking"""
        # Convert dates to simulation time indices
        start_date_obj = ss.date(self.pars.start_date)
        # Find first timevec index greater than start date
        self.start_ti = np.argmax(sim.t.yearvec > start_date_obj.years)

        if self.pars.end_date is not None:
            end_date_obj = ss.date(self.pars.end_date)
            # Find last timevec index less than or equal to end date
            valid_indices = np.where(sim.t.yearvec <= end_date_obj.years)[0]
            self.end_ti = valid_indices[-1] if len(valid_indices) > 0 else 0
        else:
            self.end_ti = np.inf

        # Find rotavirus diseases
        # With cross-protection, ALL rotavirus diseases are covered (with different efficacy levels)
        self.covered_diseases = []
        for disease in sim.diseases.values():
            if isinstance(disease, Rotavirus):
                self.covered_diseases.append(disease)

        if not self.covered_diseases:
            raise RuntimeError("No Rotavirus diseases found in simulation")

        # Precompute match efficacies for all diseases (performance optimization). These are based on the antigen match types.
        self.disease_match_efficacies = {}
        for disease in self.covered_diseases:
            self.disease_match_efficacies[disease.name] = self._compute_match_efficacy(disease)

        # NOW call super().init_pre() with states already defined
        super().init_pre(sim)

        # Print initialization summary if verbose
        if self.pars.verbose:
            print(f"RotaVaccination initialized:")
            print(f"  Start: {self.pars.start_date} (ti={self.start_ti})")
            print(f"  End: {self.pars.end_date} (ti={self.end_ti})")
            print(f"  Doses: {self.pars.n_doses}")
            print(f"  Interval: {self.pars.dose_interval} days")
            print(f"  G antigens: {self.pars.G_antigens}")
            print(f"  P antigens: {self.pars.P_antigens}")
            print(f"  Effectiveness: {self.pars.dose_effectiveness}")
            print(f"  Age range: {self.pars.min_age}-{self.pars.max_age} days")
            print(f"  Uptake: {self.pars.uptake_dist}")
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

        people = sim.people

        # Age eligibility
        age_eligible = (people.age >= self.pars.min_age.years) & (people.age <= self.pars.max_age.years)

        # Exclude those who completed the schedule
        age_eligible = age_eligible & ~self.completed_schedule

        # For multi-dose: check if next dose is due
        next_dose_due = (self.doses_received < self.pars.n_doses) & (self.ti >= self.next_dose_due)
        first_dose = self.doses_received == 0

        age_eligible = age_eligible & (first_dose | next_dose_due)

        return age_eligible

    def step(self):
        """Apply vaccination and update vaccine protection at current timestep"""
        # First, update vaccine protection for all agents (waning)
        # This must happen before the check for active intervention because vaccinations will continue to affect rel_sus even after the program ends
        self._update_vaccine_protection()

        if self.ti < self.start_ti or self.ti > self.end_ti:
            return

        # Then, check for new vaccinations
        eligible_agents = self.check_eligibility()
        eligible_uids = eligible_agents.uids

        if len(eligible_uids) > 0:
            # Track eligibility: increment doses_eligible and update next_dose_due for all eligible agents
            self.doses_eligible[eligible_uids] += 1

            # Update next_dose_due for agents who will need more doses
            still_need_doses = self.doses_eligible[eligible_uids] < self.pars.n_doses
            self.next_dose_due[eligible_uids] = np.where(
                still_need_doses,
                self.ti + self.pars.dose_interval,
                self.next_dose_due[eligible_uids],  # Keep existing value if no more doses needed
            )

            # Random uptake
            vaccinated_uids = self.pars.uptake_dist.filter(eligible_uids)

            if len(vaccinated_uids) > 0:
                # Apply vaccination (vectorized)
                self._vaccinate_agents(vaccinated_uids)

                if self.pars.verbose:
                    total_eligible = len(eligible_uids)
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
        self.last_dose_ti[uids] = self.ti

        # Mark completed schedules
        completed_mask = self.doses_received[uids] >= self.pars.n_doses
        completed_uids = uids[completed_mask]
        if len(completed_uids) > 0:
            self.completed_schedule[completed_uids] = True

        # Apply vaccine protection to covered diseases
        self._apply_vaccine_protection(uids, current_doses)

        if self.pars.verbose:
            # Group by dose number for cleaner output
            dose_counts = {}
            for i, uid in enumerate(uids):
                dose_num = current_doses[i]  # Get dose number for this agent
                dose_display = dose_num + 1  # 1-indexed for display
                effectiveness = self.pars.dose_effectiveness[dose_num]
                if dose_display not in dose_counts:
                    dose_counts[dose_display] = {
                        "count": 0,
                        "effectiveness": effectiveness,
                    }
                dose_counts[dose_display]["count"] += 1

            for dose_display, info in sorted(dose_counts.items()):
                print(
                    f"  Dose {dose_display}/{self.pars.n_doses}: {info['count']} agents (effectiveness={info['effectiveness']:.1%})"
                )

    def _is_homotypic_match(self, disease):
        """Check if disease strain has exact G+P match with vaccine"""
        return disease.G in self.pars.G_antigens and disease.P in self.pars.P_antigens

    def _is_partial_heterotypic_match(self, disease):
        """Check if disease strain has partial match (shared G or P) with vaccine"""
        return (
            disease.G in self.pars.G_antigens or disease.P in self.pars.P_antigens
        ) and not self._is_homotypic_match(disease)

    def _is_complete_heterotypic_match(self, disease):
        """Check if disease strain has no match with vaccine"""
        return not (disease.G in self.pars.G_antigens or disease.P in self.pars.P_antigens)

    def _compute_match_efficacy(self, disease):
        """Compute and return match efficacy for a disease (called once during initialization)"""
        if self._is_homotypic_match(disease):
            return self.pars.homotypic_efficacy
        elif self._is_partial_heterotypic_match(disease):
            return self.pars.partial_heterotypic_efficacy
        else:  # complete heterotypic
            return self.pars.complete_heterotypic_efficacy

    def _apply_vaccine_protection(self, uids, current_doses):
        """
        Apply vaccine protection by updating vaccine states (vectorized)

        Updates the vaccine protection level and waning time for each covered disease.
        The protection will later be applied to rel_sus in _update_vaccine_protection.
        """
        if len(uids) == 0:
            return

        # Get effectiveness values for each agent based on their current dose number
        # dose_efficacy = self.pars.dose_effectiveness[current_doses]

        # Sample waning times for each agent
        if hasattr(self.pars.waning_rate_dist, "rvs"):
            self.waning_rate[uids] = 1.0 / self.pars.waning_rate_dist.rvs(uids)
        self.waning_delay[uids] = self.pars.waning_delay

        if self.pars.verbose:
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

            # Only apply to agents who have received at least one dose
        vaccinated_mask = self.doses_received > 0
        if not np.any(vaccinated_mask):
            return  # No one is vaccinated yet

        # Calculate time since last dose for vaccinated agents

        days_since_vaccination = np.maximum(0, (self.ti - self.last_dose_ti) * self.dt.days)
        days_since_waning = np.maximum(0, days_since_vaccination - self.waning_delay)

        waned_effectiveness_factor = np.exp(-self.waning_rate * days_since_waning)

        # Update waned protection for each covered disease
        for disease in self.covered_diseases:
            dose_effectiveness = np.zeros_like(waned_effectiveness_factor)
            dose_effectiveness[vaccinated_mask] = np.array(self.pars.dose_effectiveness)[
                self.doses_received[vaccinated_mask] - 1
            ]
            vx_eff = waned_effectiveness_factor * dose_effectiveness * self.disease_match_efficacies[disease.name]

            # Apply to disease rel_sus if vaccine protection is better
            # Calculate vaccine susceptibility (inverse of protection)
            vx_sus = 1.0 - vx_eff

            # Use the minimum susceptibility (most protective)
            current_sus = disease.rel_sus[:]
            new_sus = np.minimum(current_sus, vx_sus)
            disease.rel_sus[:] = new_sus

    def get_vaccination_summary(self):
        """Get summary of vaccination program status"""
        total_agents = len(self.doses_received)

        summary = {
            "total_agents": total_agents,
            "doses_eligible": np.count_nonzero(self.doses_eligible > 0),
            "received_any_dose": np.count_nonzero(self.doses_received > 0),
            "completed_schedule": np.count_nonzero(self.completed_schedule),
            "doses_by_number": {},
            "mean_doses": (
                np.mean(self.doses_received[self.doses_received > 0]) if np.any(self.doses_received > 0) else 0
            ),
        }

        # Count agents by dose number
        for dose_num in range(0, self.pars.n_doses + 1):
            summary["doses_by_number"][dose_num] = np.count_nonzero(self.doses_received == dose_num)

        return summary

    def print_vaccination_summary(self):
        """Print vaccination program summary"""
        summary = self.get_vaccination_summary()

        print(f"\n=== RotaVaccination Summary ===")
        print(f"Total agents: {summary['total_agents']:,}")
        print(
            f"Ever eligible: {summary['doses_eligible']:,} ({100 * summary['doses_eligible'] / summary['total_agents']:.1f}%)"
        )
        print(
            f"Received any dose: {summary['received_any_dose']:,} ({100 * summary['received_any_dose'] / summary['total_agents']:.1f}%)"
        )
        print(
            f"Completed schedule: {summary['completed_schedule']:,} ({100 * summary['completed_schedule'] / summary['total_agents']:.1f}%)"
        )
        print(f"Mean doses (among vaccinated): {summary['mean_doses']:.2f}")

        print(f"\nDoses received:")
        for dose_num in range(1, self.pars.n_doses + 1):
            count = summary["doses_by_number"][dose_num]
            pct = 100 * count / summary["total_agents"]
            print(f"  Dose {dose_num}: {count:,} ({pct:.1f}%)")

        print(f"\nVaccine coverage:")
        print(f"  G antigens: {self.pars.G_antigens}")
        print(f"  P antigens: {self.pars.P_antigens}")
        print(f"  Effectiveness by dose: {self.pars.dose_effectiveness}")


# Legacy alias for backward compatibility
RotaVax = RotaVaccination

#
# class InitializeChildImmunity(ss.Intervention):
#     """
#     Initialize young children with prior infection history at simulation start.
#
#     This intervention sets the infection count for children under a specified age
#     to reflect realistic pre-existing immunity. This is important for calibration
#     because it ensures young children start with some infection history, which
#     affects severity-based reporting rates.
#
#     Args:
#         max_age_years (float): Maximum age in years for initialization (default: 3.0 = 36 months)
#         min_infections (int): Minimum number of prior infections to assign (default: 1)
#         max_infections (int): Maximum number of prior infections to assign (default: 1)
#         verbose (bool): Print initialization details (default: False)
#
#     Examples:
#         # Initialize all children <36 months with 1 prior infection
#         init_immunity = InitializeChildImmunity(max_age_years=3.0, min_infections=1)
#
#         # Initialize all children <24 months with 1-2 prior infections
#         init_immunity = InitializeChildImmunity(
#             max_age_years=2.0,
#             min_infections=1,
#             max_infections=2
#         )
#     """
#
#     def __init__(self, max_age_years=3.0, min_infections=1, max_infections=1, verbose=False, **kwargs):
#         super().__init__(**kwargs)
#         self.max_age_years = max_age_years
#         self.min_infections = min_infections
#         self.max_infections = max_infections
#         self.verbose = verbose
#         self.n_infections_dist = ss.randint(self.min_infections, self.max_infections+1)
#
#     def step(self):
#         """Initialize infection counts for young children at t=0"""
#         sim = self.sim
#         # Only run once at initialization
#         if sim.ti != 0:
#             return
#
#         # Get all Rotavirus disease instances
#         rota_diseases = [d for d in sim.diseases.values() if isinstance(d, Rotavirus)]
#
#         if len(rota_diseases) == 0:
#             if self.verbose:
#                 print("InitializeChildImmunity: No Rotavirus diseases found, skipping")
#             return
#
#         # Find children under max_age_years
#         child_uids = (sim.people.age < self.max_age_years).uids
#
#         if len(child_uids) == 0:
#             if self.verbose:
#                 print(f"InitializeChildImmunity: No children <{self.max_age_years} years found")
#             return
#
#         # Set infection counts for each child
#         # For simplicity, use the first disease's n_infections state
#         # (all diseases share the same n_infections counter per person)
#         # todo: the above isn't true. each disease has its own n_infections counter, but leaving for now
#         disease = rota_diseases[0]
#         disease.n_infections[child_uids] = self.n_infections_dist.rvs(child_uids)
#
#         # Report what was done
#         if self.verbose:
#             print(f"\nInitializeChildImmunity:")
#             print(f"  Initialized {len(child_uids)} children <{self.max_age_years} years")
#             print(f"  Prior infections: {self.min_infections}-{self.max_infections}")
#             if self.min_infections == self.max_infections:
#                 print(f"  All children assigned {self.min_infections} prior infection(s)")
#             else:
#                 mean_prior = disease.n_infections[child_uids].mean()
#                 print(f"  Mean prior infections: {mean_prior:.2f}")