import numpy as np
import rotasim as rs
import sciris as sc
import starsim as ss
import calibration.process_incidence as cpi

def test_calibration():
    # Test running sims
    sc.heading('Test Rotasim calibration')
    sim = rs.Sim(N=10_000, timelimit=7)
    events = sim.run()
    out = cpi.process_model(sim.connectors['rota'].df)
    print(out)

def test_calibration2():
    """
    This test is a basic example of how to use Starsim's built-in calibration tools.
    It does not use real data, and the generated data is nonsensical, but it shows the
    workflow.

    In this example, we are calibrating two parameters: rel_beta and initial_immunity_rate.
    The "data" it optimizes is the total number of infections vs a constant value of 100.
    """
    n_agents = 2000
    debug = False

    def make_sim():
        rota = rs.Rota(
            rel_beta=1,
            initial_immunity_rate=0.1,
        )

        analyzer = rs.StrainStats()

        sim = rs.Sim(
            n_agents=n_agents,
            start=sc.date('1960-01-01'),
            timelimit=10,
            connectors=rota,
            analyzers=analyzer,
            verbose=0,
        )

        return sim

    calib_pars = dict(
        rel_beta=dict(low=0.1, high=1, guess=0.9, suggest_type='suggest_float', log=True),  # Note the log scale
        initial_immunity_rate = dict(low=0.01, high=0.2, guess=0.1),  # Default type is suggest_float, no need to re-specify
    )

    def build_sim(sim, calib_pars, **kwargs):
        """
        Modify the base simulation by applying calib_pars. The result can be a
        single simulation or multiple simulations if n_reps>1. Note that here we are
        simply building the simulation by modifying the base sim. Running the sims
        and extracting results will be done by the calibration function.
        """

        rota =  sim.connectors.rota  # There is only one disease in this simulation and it is a SIR

        for k, pars in calib_pars.items():  # Loop over the calibration parameters
            if k == 'rand_seed':
                sim.pars.rand_seed = v
                continue

            # Each item in calib_pars is a dictionary with keys like 'low', 'high',
            # 'guess', 'suggest_type', and importantly 'value'. The 'value' key is
            # the one we want to use as that's the one selected by the algorithm
            v = pars['value']
            if k == 'rel_beta':
                rota.pars.rel_beta = v
            elif k == 'initial_immunity_rate':
                rota.pars.initial_immunity_rate = v
            else:
                raise NotImplementedError(f'Parameter {k} not recognized')

        # If just one simulation per parameter set, return the single simulation

        return sim

    sc.heading('Beginning calibration')

    # Make the sim and data
    sim = make_sim()
    sim.init()

    def eval(sim, expected):

        ret = 0
        daily_counts=np.zeros(len(sim.results.strainstats.timevec))
        results = sim.connectors.rota.to_df().groupby('CollectionTime').count()['Strain']

        for k, v in sim.results.strainstats.items():
            if 'count' in k:
                daily_counts += v.values


        ret += sum( (daily_counts - expected)**2 )
        return ret

    # Make the calibration
    calib = ss.Calibration(
        calib_pars=calib_pars,
        sim=sim,
        build_fn=build_sim,
        build_kw=dict(n_reps=3),  # Run 3 replicates for each parameter set
        reseed=True,  # If true, a different random seed will be provided to each configuration
        eval_fn= eval,
        eval_kw=dict(expected=np.full(len(sim.timevec), 100),),
        total_trials=10,
        n_workers=None,  # None indicates to use all available CPUs
        die=True,
        debug=debug,  # Run in serial if True
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate();