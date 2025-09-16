"""
Rotavirus disease class for v2 architecture
Individual strain-specific disease instances following traditional Starsim patterns
"""
import starsim as ss


class Rotavirus(ss.Infection):
    """
    Individual rotavirus strain as a separate disease instance
    
    Each Rotavirus instance represents a specific G,P combination that behaves
    as an independent disease in the simulation, with cross-strain interactions
    handled by connector classes.
    """
    
    def __init__(self, G, P, pars=None, **kwargs):
        """
        Initialize Rotavirus strain
        
        Args:
            G (int): G genotype (antigenic segment)
            P (int): P genotype (antigenic segment) 
            pars (dict, optional): Parameters dict
            **kwargs: Standard ss.Infection parameters (init_prev, beta, name, etc.)
        """
        # Store G,P genotypes as attributes
        self.G = G
        self.P = P
        
        # Auto-generate name if not provided
        if 'name' not in kwargs:
            kwargs['name'] = f"G{G}P{P}"
            
        super().__init__()
        
        self.define_pars(
            init_prev = ss.bernoulli(p=0.01),     # Initial prevalence
            beta = ss.perday(0.1),               # Transmission rate (will be modified by fitness)
            dur_inf = ss.lognorm_ex(mean=7),      # Duration of infection (~7 days)
            dur_waning = ss.poisson(lam=90), # Duration of waning immunity (~100 days)
            waning_delay = ss.days(0)
        )


        
        # Define additional disease states (base ss.Infection already provides susceptible, infected, rel_sus, rel_trans, ti_infected)
        self.define_states(
            ss.BoolState('recovered', label='Recovered'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_waned', label='Time of waned immunity'),
            ss.FloatArr('n_infections', default=0, label='Total number of infections'),
        )
        
        self.update_pars(pars=pars, **kwargs)

        self.pars.dur_inf.dt_jump_size = 15000 # TODO check if this is reasonable

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('new_recovered', label='New recoveries this timestep', dtype=int, scale=True))

        return

        
    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses for agents who become infected with this rotavirus strain
        
        This method is called when agents transition from susceptible to infected.
        It handles the state transitions and determines recovery timing.
        
        Args:
            uids (array): UIDs of agents becoming infected
            sources (array, optional): UIDs of agents who infected them (for contact tracing)
        """
        # Call parent method for logging and other standard setup
        super().set_prognoses(uids, sources)
        
        # Get current timestep (use self.t.ti pattern from SIR example)
        ti = self.t.ti
        
        # Update agent states: susceptible → infected
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.recovered[uids] = False
        self.ti_infected[uids] = ti
        
        # Increment infection count for each agent
        self.n_infections[uids] += 1
        
        # Sample duration of infection for each agent
        # dur_inf is typically ss.lognorm_ex(mean=7) for ~7 days
        dur_inf = self.pars.dur_inf.rvs(uids)
        
        # Set recovery time: current time + infection duration
        self.ti_recovered[uids] = ti + dur_inf
        self.sim.connectors['rotaimmunityconnector'].record_infection(self, uids)
        
        return

    def step_state(self):
        """
        Update disease states each timestep
        
        This method handles state transitions:
        - infected → recovered (when ti_recovered is reached)
        """
        # Progress infected -> recovered (following SIR example pattern)
        sim = self.sim
        recovering = (self.infected & (self.ti_recovered <= sim.ti)).uids
        self.infected[recovering] = False
        self.recovered[recovering] = True
        self.susceptible[recovering] = True # When recovered, become susceptible again (SIRS), but with modified susceptibility via connector
        self.ti_waned[recovering] = sim.ti + self.pars.dur_waning.rvs(recovering)

        self.results['new_recovered'][self.ti] = len(recovering)

        self.sim.connectors['rotaimmunityconnector'].record_recovery(self, recovering)
        
        return

    @property
    def strain(self):
        """Return strain tuple for compatibility with existing code"""
        return (self.G, self.P)