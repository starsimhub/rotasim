import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['RotaVax', 'RotaVaxProg']

class RotaVax(ss.Vx):
    """
    RotaVax is a product class for administering Rota virus vaccinations.

    Args:
        vx_types (string/list): a list of vaccine G and P types, e.g. ['G1', 'G2', 'P1']

        **kwargs: Additional keyword arguments.
    """
    def __init__(self, vx_types=None, fixed_protection=0, mean_dur_protection=[ss.dur(39, 'weeks'), ss.dur(78, 'weeks')], **kwargs):
        super().__init__(**kwargs)
        self.segments='gpab'
        self.g = []
        self.p = []

        self.fixed_protection = fixed_protection
        self.mean_dur_protection = mean_dur_protection

        vx_types = sc.promotetolist(vx_types)  # Ensure vx_types is a list
        self.vx_types = vx_types

        self.g = [vx_type[1] for vx_type in vx_types if vx_type[0].upper() == 'G' ]
        self.p = [vx_type[1] for vx_type in vx_types if vx_type[0].upper() == 'P' ]

        return


    def administer(self, people, inds):
        """
        Administer the RotaVax vaccination to eligible individuals.

        Args:
            people (People): The population to administer the vaccine to.
            inds (array-like): Indices of individuals to vaccinate.

        Returns:
            None
        """
        # Adjust immunity and severe rates based on number of doses received
        return



class RotaVaxProg(ss.routine_vx):
    """
    RotaVaxProg is a vaccination program for Rota virus.

    Args:
        pars (dict): Parameters for the vaccination program.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, pars=None, product=None, prob=None, eligibility=None, **kwargs):
        if product is None:
            # product = RotaVax(pars=pars, **kwargs)
            raise ValueError("A product must be specified for RotaVaxProg.")
        if eligibility is None:
            eligibility = self.eligible  # Define eligibility function
        if prob is None:
            prob = 0.89  # Default probability of vaccination


        super().__init__(pars=pars, product=product, prob=prob, eligibility=eligibility,  **kwargs)
        self.define_pars(
            vx_age_min=ss.dur(4.55, 'week'),
            vx_age_max=ss.dur(6.5, 'week'),
            vx_spacing=ss.dur(6, unit='week'),  # Spacing between doses
            waning_rate = [ss.dur(2, unit='week'),  # Rate of waning for the first dose
                           ss.dur(78, unit='week')],  # Rate of waning for subsequent doses
            waning_delay = ss.dur(0, unit='week'), #, parent_dt = self.t.dt, parent_unit=self.t.unit), # Delay before waning starts
            max_doses=2,
        )


        self.update_pars(pars=pars, **kwargs)

        self.define_states(
            ss.FloatArr('n_doses', default=0, label='Number of doses received'),
            ss.FloatArr('vx_e_i', default=0, label='Vaccine efficacy against infection'),
            ss.FloatArr('vx_e_s', default=0, label='Vaccine efficacy against severe disease'),
            ss.FloatArr('waning_rate', default=0, label='Rate of waning immunity'),
        )
        return

    def eligible(self, sim):
        """ Determine which agents are eligible for vaccination """
        ppl = sim.people
        eligible = ((ppl.age >= self.pars.vx_age_min) &
                    (ppl.age < self.pars.vx_age_max) &
                    (self.n_doses == 0 | ((self.n_doses < self.pars.max_doses) & (self.ti - self.ti_vaccinated >= self.pars.vx_spacing )))
                    )
        return eligible.uids

    def step(self):
        vx_uids = super().step()
        self.waning_rate[vx_uids] = [self.pars.waning_rate[int(self.n_doses[vx_uid]) - 1].values for vx_uid in vx_uids]
        self.update_immunity()

    def update_immunity(self):
        """
        Update the immunity of vaccinated individuals.

        Args:
            people (People): The population to update.
            inds (array-like): Indices of individuals whose immunity is updated.

        Returns:
            None
        """
        # Update immunity based on the vaccine's fixed protection and mean duration
        vxed = self.n_doses > 0  # Vaccinated individuals
        vxuids = vxed.uids

        waning_started = ((self.ti - self.ti_vaccinated[vxed] - self.pars.waning_delay.values) > 0)
        waning_uids = vxuids[waning_started]
        nonwaning_uids = vxuids[~waning_started]

        # waning not yet started:
        self.vx_e_i[nonwaning_uids] = 1
        self.vx_e_i[waning_uids] = np.exp( (-1/self.waning_rate[waning_uids]) * (self.ti - (np.round(self.pars.waning_delay) + self.ti_vaccinated[waning_uids])))

        return
