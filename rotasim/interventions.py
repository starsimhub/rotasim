from collections import defaultdict


import math

import numpy as np
import rotasim.rotasim_genetics as rsg
import sciris as sc
import starsim as ss


__all__ = ['RotaVax', 'RotaVaxProg']


class RotaVax(ss.Vx):
    """
    RotaVax is a product class for administering Rota virus vaccinations.

    Args:
        vx_types (string/list): a list of vaccine G and P types, e.g. ['G1', 'G2', 'P1']
        mean_dur_protection (list): List of mean durations of protection for each dose, e.g. [39 weeks, 78 weeks].
        waning_delay (int | dur): Delay before waning starts. If an int, it is interpreted as timesteps, if a dur, it is interpreted as a starsim duration.

        **kwargs: Additional keyword arguments.
    """
    def __init__(self, vx_strains=None, mean_dur_protection=[ss.dur(39, 'weeks'), ss.dur(78, 'weeks')],
                    waning_delay=ss.dur(0, unit='week'), **kwargs):
        super().__init__(**kwargs)
        self.segments='gpab'
        self.g = []
        self.p = []

        self.define_pars(
            mean_dur_protection=mean_dur_protection,  # Mean duration of protection for each dose
            waning_delay = waning_delay, # Delay before waning starts
            ve_i_to_ve_s_ratio = 0.5,
            vaccine_efficacy_dose_factor=[0.8, 1],  # Efficacy factor per dose of the vaccine [e.g. 0.8 for first dose, 1 for second dose means the first dose provides 80% of the efficacy of the second dose]
            # vaccine_efficacy_d2 determines the immunity provided by the vaccines.
            # To simulate scenarios with different vaccine behaviors such as a vaccine that only mounts immunity against homotypic strains, set the efficacy of the other two types to zero.
            vaccine_efficacy_match_factor={
                rsg.PathogenMatch.HOMOTYPIC: 0.65,  # 0.65,  # 0.95,  # 0.8,
                rsg.PathogenMatch.PARTIAL_HETERO: 0.45,  # 0.45,  # 0.9,  # 0.65,
                rsg.PathogenMatch.COMPLETE_HETERO: 0.25,  # 0.25,  # 0.75,  # 0.35,
            },
        )

        self.define_states(
            ss.FloatArr('waning_rate', default=0, label='Rate of waning immunity'),
        )

        vx_strains = sc.promotetolist(vx_strains)  # Ensure vx_types is a list

        self.g = [int(vx_type[1]) for vx_type in vx_strains if vx_type[0].upper() == 'G' ]
        self.p = [int(vx_type[1]) for vx_type in vx_strains if vx_type[0].upper() == 'P' ]

        self.vaccine_efficacy_i = {}
        self.vaccine_efficacy_s = {}

        return


    def init_post(self):
        super().init_post()

        for dose, dose_factor in enumerate(self.pars.vaccine_efficacy_dose_factor):
            self.vaccine_efficacy_i[dose+1] = {}
            self.vaccine_efficacy_s[dose+1] = {}
            for match, match_factor in self.pars.vaccine_efficacy_match_factor.items():
                (ve_i, ve_s) = self.breakdown_vaccine_efficacy(
                    match_factor*dose_factor, self.pars.ve_i_to_ve_s_ratio
                )
                self.vaccine_efficacy_i[dose+1][match] = ve_i
                self.vaccine_efficacy_s[dose+1][match] = ve_s

        if self.sim.pars.verbose > 0:
            print("VE_i d1: ", self.vaccine_efficacy_i[1])
            print("VE_s d1: ", self.vaccine_efficacy_s[1])
            print("VE_i d2: ", self.vaccine_efficacy_i[2])
            print("VE_s d2: ", self.vaccine_efficacy_s[2])


    def breakdown_vaccine_efficacy(self, ve, x):
        """
        Break down the vaccine efficacy into its components
        """
        (r1, r2) = self.solve_quadratic(x, -(1 + x), ve)
        if self.sim.pars.verbose > 0:
            print(r1, r2)
        if r1 >= 0 and r1 <= 1:
            ve_s = r1
        elif r2 >= 0 and r2 <= 1:
            ve_s = r2
        else:
            raise RuntimeError(
                "No valid solution to the equation: x: %d, ve: %d. Solutions: %f %f"
                % (x, ve, r1, r2)
            )
        ve_i = x * ve_s
        return (ve_i, ve_s)


    @staticmethod
    def solve_quadratic(a, b, c):
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            root1 = (-b + discriminant**0.5) / (2 * a)
            root2 = (-b - discriminant**0.5) / (2 * a)
            return tuple(sorted([root1, root2]))
        else:
            return "No real roots"


    def is_match(self, infecting_strain):
        """ Check if the infecting strain matches the vaccine strains.
        Args:
            infecting_strain (string|tuple): An strain having at least the first two (antigenic) types representing the infecting strain.
        Returns:
            rsg.PathogenMatch: The type of match between the infecting strain and the vaccine strains.
        """
        # check only antigenic segments (e.g. G and P)
        matches = [False, False]
        if infecting_strain[0] in self.g:
            matches[0] = True
        if infecting_strain[1] in self.p:
            matches[1] = True

        if np.all(matches):
            return rsg.PathogenMatch.HOMOTYPIC
        elif np.any(matches):
            return rsg.PathogenMatch.PARTIAL_HETERO
        else:
            return rsg.PathogenMatch.COMPLETE_HETERO



class RotaVaxProg(ss.BaseVaccination):
    """
    RotaVaxProg is a vaccination program for Rota virus.

    Args:
        prob (float): Probability of vaccination. Assumes single test per agent, so .8 means 80% chance an eligible agent will be vaccinated.
        eligibility (function): Function that returns a list of uids of agents eligible for vaccination.
        start_date (int | date): Start date of the vaccination program.
        end_date (int | date): End date of the vaccination program.
        vx_strains (list): List of vaccine strains to be used in the program, e.g. ['G1', 'P2'].
        mean_dur_protection (list): List of mean durations of protection for each dose, e.g. [39 weeks, 78 weeks].
        waning_delay (int | dur): Delay before waning starts. If an int, it is interpreted as timesteps, if a dur, it is interpreted as a starsim duration.

        **kwargs: Additional keyword arguments.
    """

    def __init__(self, pars=None, prob=None, eligibility=None, start_date=None, end_date=None, vx_strains=None, mean_dur_protection=[ss.dur(39, 'weeks'), ss.dur(78, 'weeks')],
                 waning_delay=ss.dur(0, unit='week'), **kwargs):

        # Separate out the product parameters:
        product_pars = dict(vx_strains=vx_strains, mean_dur_protection=mean_dur_protection, waning_delay=waning_delay,)
        product = RotaVax(**product_pars)

        self.start_date = start_date
        self.end_date = end_date

        if eligibility is None:
            eligibility = self.eligible  # Define eligibility function

        if prob is None:
            # Vaccination coverage is derived based on the following formula
            # we are going to set a target vaccine second dose coverage (e.g 80%) and use that value to decide how much of the population needs to get the first dose.
            # we assume that the same proportion of people who get the first dose will get the second dose (0.89 * 0.89 = 0.8).
            # Therefore we set the first dose coverage to the square root of second dose coverage
            prob = 0.89  # Default probability of vaccination

        self.define_states(
            ss.FloatArr('n_doses', default=0, label='Number of doses received'),
            ss.FloatArr('waned_effectiveness', default=1.0, label='current base waned effectiveness'),
            ss.FloatArr('waning_rate', default=0, label='Rate of waning immunity'),
            ss.BoolArr("to_vx", default=False, label="Vaccination flag, if true the person is eligible for vaccination"),
        )

        super().__init__(product=product, prob=prob, eligibility=eligibility)
        self.define_pars(
            vx_age_min=ss.dur(4.55, 'week'),
            vx_age_max=ss.dur(6.5, 'week'),
            vx_spacing=ss.dur(6, unit='week'),  # Spacing between doses
            max_doses=2,
        )

        self.update_pars(pars=pars, **kwargs)

        return


    def init_pre(self, sim):
        super().init_pre(sim)

        # convert the start and end dates to date objects if they are not already
        if self.start_date is None:
            self.start_date = ss.date(sim.t.timevec[0])
        if self.end_date is None:
            self.end_date = ss.date(sim.t.timevec[-1])

        if not isinstance(self.start_date, ss.date):
            self.start_date = ss.date(self.start_date)
        if not isinstance(self.end_date, ss.date):
            self.end_date = ss.date(self.end_date)

        start_year = self.start_date.to_year()
        end_year = self.end_date.to_year()

        self.timepoints = [index for index, tval in enumerate(sim.t.timevec) if tval >= start_year and tval < end_year]
        if isinstance(self.prob, (int, float)) or len(self.prob) == 1:
            probs = [self.prob] * len(self.timepoints)
            self.prob = probs  # Assign the probability to the timepoints


    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('new_vaccinated_first_dose', dtype=int, label='Number of people vaccinated with first dose'),
                            ss.Result('new_vaccinated_second_dose', dtype=int, label='Number of people vaccinated with second dose'),)


    def eligible(self, sim):
        """ Determine which agents are eligible for vaccination """
        # ppl = sim.people
        eligible = self.to_vx == True
        return eligible.uids


    def step(self):
        ppl = self.sim.people
        if self.sim.ti in self.timepoints:
            if self.sim.ti == self.timepoints[0]:
                # if there is no strain associated with the product, select the most prevalent strain
                if self.product.g == [] and self.product.p == []:
                    total_strain_counts = defaultdict(int)
                    for strain, count in self.sim.connectors.rota.strain_count.items():
                        total_strain_counts[strain[: self.sim.connectors.rota.pars.numAgSegments]] += (
                            count
                        )

                    strain = sorted(
                        list(total_strain_counts.keys()),
                        key=lambda x: total_strain_counts[x],
                    )[-1]

                    self.product.g = [strain[0]]
                    self.product.p = [strain[1]]

                # this is the first time step, so all still need an opportunity to get vaccinated

                to_vx = (ppl.age >= self.pars.vx_age_min) & (ppl.age < self.pars.vx_age_max)
                self.to_vx[to_vx] = True
            else:
                # if this is not the first time step, we only vaccinate those who are eligible:
                #   * those who have aged into the vaccination age range
                #   * those ready for their next dose

                to_vx_age = (ppl.age >= self.pars.vx_age_min) & (ppl.age < (self.pars.vx_age_min + self.t.dt_year))
                to_vx_next_dose = (self.n_doses < self.pars.max_doses) & \
                                  (self.ti - self.ti_vaccinated >= self.pars.vx_spacing.values) & \
                                  (self.ti - self.ti_vaccinated < self.pars.vx_spacing.values + 1)  # Ensure they have been vaccinated before
                self.to_vx[to_vx_age] = True
                self.to_vx[to_vx_next_dose] = True

        vx_uids = super().step()

        self.results.new_vaccinated_first_dose[self.ti] = np.count_nonzero(self.n_doses[vx_uids] == 1)
        self.results.new_vaccinated_second_dose[self.ti] = np.count_nonzero(self.n_doses[vx_uids] == 2)

        # update the waning rate for all new vaccinations based on the total number of doses
        self.product.waning_rate[vx_uids] = [self.product.pars.mean_dur_protection[int(self.n_doses[vx_uid]) - 1].values for vx_uid in vx_uids]

        self.update_immunity()

        self.to_vx[:] = False # Reset the vaccination flag for all individuals after processing


    def update_immunity(self):
        # Update immunity based on the vaccine's waning delay and mean duration
        vxed = self.n_doses > 0  # Vaccinated individuals
        vxuids = vxed.uids

        waning_started = ((self.ti - self.ti_vaccinated[vxed] - self.product.pars.waning_delay.values) > 0)
        waning_uids = vxuids[waning_started]
        nonwaning_uids = vxuids[~waning_started]

        # waning not yet started:
        self.waned_effectiveness[nonwaning_uids] = 1
        self.waned_effectiveness[waning_uids] = np.exp( (-1/self.product.waning_rate[waning_uids]) *
                                           (self.ti - (np.round(self.product.pars.waning_delay) + self.ti_vaccinated[waning_uids])))

        return
