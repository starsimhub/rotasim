import csv
import itertools
import math

import numpy as np
import random as rnd
import sciris as sc
import starsim as ss

from . import arrays as rsa

__all__ = ['Rota']

# Define age bins and labels
age_bins = [2/12, 4/12, 6/12, 12/12, 24/12, 36/12, 48/12, 60/12, 100]
age_distribution = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.84]                  # needs to be changed to fit the site-specific population
age_labels = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60+']


### Pathogen classes

class PathogenMatch:
    """ Define whether pathogens are completely heterotypic, partially heterotypic, or homotypic """
    COMPLETE_HETERO = 1
    PARTIAL_HETERO = 2
    HOMOTYPIC = 3

class RotaPathogen(sc.quickobj):
    """
    Pathogen dynamics
    """

    def __init__(self, sim, is_reassortant, creation_time, is_severe=False, host_uid=None, strain=None):
        self.sim = sim
        self.host_uid = host_uid
        self.creation_time = creation_time
        self.is_reassortant = is_reassortant
        self.strain = strain
        self.is_severe = is_severe
        self.fitness_hypothesis = self.sim.pars.rota.fitness_hypothesis

        self.fitness_map = {
            1: {'default': 1},
            2: {'default': 0.90, (1, 1): 0.93, (2, 2): 0.93, (3, 3): 0.93, (4, 4): 0.93},
            3: {'default': 0.87, (1, 1): 0.93, (2, 2): 0.93, (3, 3): 0.90, (4, 4): 0.90},
            4: {'default': 1, (1, 1): 1, (2, 2): 0.2},
            5: {'default': 0.2, (1, 1): 1, (2, 1): 0.5, (1, 3): 0.5},
            6: {'default': 0.05, (1, 8): 1, (2, 4): 0.2, (3, 8): 0.4, (4, 8): 0.5},
            7: {'default': 0.05, (1, 8): 1, (2, 4): 0.3, (3, 8): 0.7, (4, 8): 0.6},
            8: {'default': 0.05, (1, 8): 1, (2, 4): 0.4, (3, 8): 0.9, (4, 8): 0.8},
            9: {'default': 0.2, (1, 8): 1, (2, 4): 0.5, (3, 8): 0.9, (4, 8): 0.8},
            10: {'default': 0.4, (1, 8): 1, (2, 4): 0.6, (3, 8): 0.9, (4, 8): 0.9},
            11: {'default': 0.5, (1, 8): 0.98, (2, 4): 0.7, (3, 8): 0.8, (4, 8): 0.8},
            12: {'default': 0.5, (1, 8): 0.98, (2, 4): 0.8, (3, 8): 0.9, (4, 8): 0.9},
            13: {'default': 0.7, (1, 8): 0.98, (2, 4): 0.8, (3, 8): 0.9, (4, 8): 0.9},
            14: {'default': 0.05, (1, 8): 0.98, (2, 4): 0.4, (3, 8): 0.7, (4, 8): 0.6, (9, 8): 0.7, (12, 8): 0.75,
                 (9, 6): 0.58, (11, 8): 0.2},
            15: {'default': 0.4, (1, 8): 1, (2, 4): 0.7, (3, 8): 0.93, (4, 8): 0.93, (9, 8): 0.95, (12, 8): 0.94,
                 (9, 6): 0.3, (11, 8): 0.35},
            16: {'default': 0.4, (1, 8): 1, (2, 4): 0.7, (3, 8): 0.85, (4, 8): 0.88, (9, 8): 0.95, (12, 8): 0.93,
                 (9, 6): 0.85, (12, 6): 0.90, (9, 4): 0.90, (1, 6): 0.6, (2, 8): 0.6, (2, 6): 0.6},
            17: {'default': 0.7, (1, 8): 1, (2, 4): 0.85, (3, 8): 0.85, (4, 8): 0.88, (9, 8): 0.95, (12, 8): 0.93,
                 (9, 6): 0.83, (12, 6): 0.90, (9, 4): 0.90, (1, 6): 0.8, (2, 8): 0.8, (2, 6): 0.8},
            18: {'default': 0.65, (1, 8): 1, (2, 4): 0.92, (3, 8): 0.79, (4, 8): 0.81, (9, 8): 0.95, (12, 8): 0.89,
                 (9, 6): 0.80, (12, 6): 0.86, (9, 4): 0.83, (1, 6): 0.75, (2, 8): 0.75, (2, 6): 0.75},
            19: {'default': 0.4, (1, 8): 1, (2, 4): 0.5, (3, 8): 0.55, (4, 8): 0.55, (9, 8): 0.6},
        }
        return

    # compares two strains
    # if they both have the same antigenic segments we return homotypic
    def match(self, strainIn):
        numAgSegments = self.sim.numAgSegments
        if strainIn[:numAgSegments] == self.strain[:numAgSegments]:
            return PathogenMatch.HOMOTYPIC

        strains_match = False
        for i in range(numAgSegments):
            if strainIn[i] == self.strain[i]:
                strains_match = True

        if strains_match:
            return PathogenMatch.PARTIAL_HETERO
        else:
            return PathogenMatch.COMPLETE_HETERO

    def get_fitness(self):
        """ Get the fitness based on the fitness hypothesis and the two strains """
        fitness_hypothesis = self.fitness_hypothesis
        key = (self.strain[0], self.strain[1])

        if fitness_hypothesis == 1:
            return 1

        elif fitness_hypothesis == 2:
            default = 0.90
            mapping = {
                (1, 1): 0.93,
                (2, 2): 0.93,
                (3, 3): 0.93,
                (4, 4): 0.93,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 3:
            default = 0.87
            mapping = {
                (1, 1): 0.93,
                (2, 2): 0.93,
                (3, 3): 0.90,
                (4, 4): 0.90,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 4:
            default = 1
            mapping = {
                (1, 1): 1,
                (2, 2): 0.2,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 5:
            default = 0.2
            mapping = {
                (1, 1): 1,
                (2, 1): 0.5,
                (1, 3): 0.5,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 6:
            default = 0.05
            mapping = {
                (1, 8): 1,
                (2, 4): 0.2,
                (3, 8): 0.4,
                (4, 8): 0.5,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 7:
            default = 0.05
            mapping = {
                (1, 8): 1,
                (2, 4): 0.3,
                (3, 8): 0.7,
                (4, 8): 0.6,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 8:
            default = 0.05
            mapping = {
                (1, 8): 1,
                (2, 4): 0.4,
                (3, 8): 0.9,
                (4, 8): 0.8,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 9:
            default = 0.2
            mapping = {
                (1, 8): 1,
                (2, 4): 0.6,
                (3, 8): 0.9,
                (4, 8): 0.9,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 10:
            default = 0.4
            mapping = {
                (1, 8): 1,
                (2, 4): 0.6,
                (3, 8): 0.9,
                (4, 8): 0.9,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 11:
            default = 0.5
            mapping = {
                (1, 8): 0.98,
                (2, 4): 0.7,
                (3, 8): 0.8,
                (4, 8): 0.8,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 12:
            default = 0.5
            mapping = {
                (1, 8): 0.98,
                (2, 4): 0.7,
                (3, 8): 0.9,
                (4, 8): 0.9,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 13:
            default = 0.7
            mapping = {
                (1, 8): 0.98,
                (2, 4): 0.8,
                (3, 8): 0.9,
                (4, 8): 0.9,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 14:
            default = 0.05
            mapping = {
                (1, 8): 0.98,
                (2, 4): 0.4,
                (3, 8): 0.7,
                (12, 8): 0.75,
                (9, 6): 0.58,
                (11, 8): 0.2,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 15:
            default = 0.4
            mapping = {
                (1, 8): 1,
                (2, 4): 0.7,
                (3, 8): 0.93,
                (4, 8): 0.93,
                (9, 8): 0.95,
                (12, 8): 0.94,
                (9, 6): 0.3,
                (11, 8): 0.35,
            }
            return mapping.get(key, default)


        elif fitness_hypothesis == 16:
            default = 0.4
            mapping = {
                (1, 8): 1,
                (2, 4): 0.7,
                (3, 8): 0.85,
                (4, 8): 0.88,
                (9, 8): 0.95,
                (12, 8): 0.93,
                (9, 6): 0.85,
                (12, 6): 0.90,
                (9, 4): 0.90,
                (1, 6): 0.6,
                (2, 8): 0.6,
                (2, 6): 0.6,
            }
            return mapping.get(key, default)

        elif fitness_hypothesis == 17:
            default = 0.7
            mapping = {
                (1, 8): 1,
                (2, 4): 0.85,
                (3, 8): 0.85,
                (4, 8): 0.88,
                (9, 8): 0.95,
                (12, 8): 0.93,
                (9, 6): 0.83,
                (12, 6): 0.90,
                (9, 4): 0.90,
                (1, 6): 0.8,
                (2, 8): 0.8,
                (2, 6): 0.8,
            }
            return mapping.get(key, default)

        # below fitness hypo. 18 was used in the analysis for the high baseline diversity setting in the report
        elif fitness_hypothesis == 18:
            default = 0.65
            mapping = {
                (1, 8): 1,
                (2, 4): 0.92,
                (3, 8): 0.79,
                (4, 8): 0.81,
                (9, 8): 0.95,
                (12, 8): 0.89,
                (9, 6): 0.80,
                (12, 6): 0.86,
                (9, 4): 0.83,
                (1, 6): 0.75,
                (2, 8): 0.75,
                (2, 6): 0.75,
            }
            return mapping.get(key, default)

        # below fitness hypo 19 was used for the low baseline diversity setting analysis in the report
        elif fitness_hypothesis == 19:
            default = 0.4
            mapping = {
                (1, 8): 1,
                (2, 4): 0.5,
                (3, 8): 0.55,
                (4, 8): 0.55,
                (9, 8): 0.6,
            }
            return mapping.get(key, default)

        else:
            print("Invalid fitness_hypothesis: ", fitness_hypothesis)
            exit(-1)


    def get_strain_name(self):
        G, P, A, B = [str(self.strain[i]) for i in range(4)]
        return f'G{G}P{P}A{A}B{B}'

    def __str__(self):
        return "Strain: " + self.get_strain_name() + " Severe: " + str(self.is_severe) + " Host: " + str(
            self.host.id) + str(self.creation_time)

class Vaccine(sc.prettyobj):
    """
    A rotavirus vaccine
    """
    def __init__(self, strain, time, dose):
        self.strain = strain  # list of strains
        self.time = time  # time of vaccination
        self.dose = dose  # number of doses
        return

class Rota(ss.Module):
    """
    Pathogen dynamics
    """
    def __init__(self, to_csv=True, timelimit=2, **kwargs):
        super().__init__()
        self.T = sc.timer()

        numSegments = 4
        numNoneAgSegments = 2
        numAgSegments = numSegments - numNoneAgSegments

        self.define_pars(
            immunity_hypothesis = 1,
            reassortment_rate = 0.1,
            fitness_hypothesis = 2,
            vaccine_hypothesis = 1,
            waning_hypothesis = 1,
            initial_immunity = 0,
            ve_i_to_ve_s_ratio = 0.5,
            experiment_number = 1,
            rel_beta = 1.0,
            fitness_map = {
                1: {'default': 1},
                2: {'default': 0.90, (1, 1): 0.93, (2, 2): 0.93, (3, 3): 0.93, (4, 4): 0.93},
                3: {'default': 0.87, (1, 1): 0.93, (2, 2): 0.93, (3, 3): 0.90, (4, 4): 0.90},
                4: {'default': 1, (1, 1): 1, (2, 2): 0.2},
                5: {'default': 0.2, (1, 1): 1, (2, 1): 0.5, (1, 3): 0.5},
                6: {'default': 0.05, (1, 8): 1, (2, 4): 0.2, (3, 8): 0.4, (4, 8): 0.5},
                7: {'default': 0.05, (1, 8): 1, (2, 4): 0.3, (3, 8): 0.7, (4, 8): 0.6},
                8: {'default': 0.05, (1, 8): 1, (2, 4): 0.4, (3, 8): 0.9, (4, 8): 0.8},
                9: {'default': 0.2, (1, 8): 1, (2, 4): 0.5, (3, 8): 0.9, (4, 8): 0.8},
                10: {'default': 0.4, (1, 8): 1, (2, 4): 0.6, (3, 8): 0.9, (4, 8): 0.9},
                11: {'default': 0.5, (1, 8): 0.98, (2, 4): 0.7, (3, 8): 0.8, (4, 8): 0.8},
                12: {'default': 0.5, (1, 8): 0.98, (2, 4): 0.8, (3, 8): 0.9, (4, 8): 0.9},
                13: {'default': 0.7, (1, 8): 0.98, (2, 4): 0.8, (3, 8): 0.9, (4, 8): 0.9},
                14: {'default': 0.05, (1, 8): 0.98, (2, 4): 0.4, (3, 8): 0.7, (4, 8): 0.6, (9, 8): 0.7, (12, 8): 0.75,
                     (9, 6): 0.58, (11, 8): 0.2},
                15: {'default': 0.4, (1, 8): 1, (2, 4): 0.7, (3, 8): 0.93, (4, 8): 0.93, (9, 8): 0.95, (12, 8): 0.94,
                     (9, 6): 0.3, (11, 8): 0.35},
                16: {'default': 0.4, (1, 8): 1, (2, 4): 0.7, (3, 8): 0.85, (4, 8): 0.88, (9, 8): 0.95, (12, 8): 0.93,
                     (9, 6): 0.85, (12, 6): 0.90, (9, 4): 0.90, (1, 6): 0.6, (2, 8): 0.6, (2, 6): 0.6},
                17: {'default': 0.7, (1, 8): 1, (2, 4): 0.85, (3, 8): 0.85, (4, 8): 0.88, (9, 8): 0.95, (12, 8): 0.93,
                     (9, 6): 0.83, (12, 6): 0.90, (9, 4): 0.90, (1, 6): 0.8, (2, 8): 0.8, (2, 6): 0.8},
                18: {'default': 0.65, (1, 8): 1, (2, 4): 0.92, (3, 8): 0.79, (4, 8): 0.81, (9, 8): 0.95, (12, 8): 0.89,
                     (9, 6): 0.80, (12, 6): 0.86, (9, 4): 0.83, (1, 6): 0.75, (2, 8): 0.75, (2, 6): 0.75},
                19: {'default': 0.4, (1, 8): 1, (2, 4): 0.5, (3, 8): 0.55, (4, 8): 0.55, (9, 8): 0.6},
            },
            numAgSegments = numAgSegments,
            to_csv = to_csv,  # whether to write files
            mu = 1.0 / 70.0,  # average life span is 70 years
            gamma = 365 / 7,  # 1/average infectious period (1/gamma =7 days)
            omega = None,
            birth_rate = None,
            contact_rate = 365 / 1,
            reassortmentRate_GP = None,

            vaccination_time = 20,

            # Efficacy of the vaccine first dose
            vaccine_efficacy_d1 = {
                PathogenMatch.HOMOTYPIC: 0.6,
                PathogenMatch.PARTIAL_HETERO: 0.45,
                PathogenMatch.COMPLETE_HETERO: 0.15,
            },
        # Efficacy of the vaccine second dose
            vaccine_efficacy_d2 = {
                PathogenMatch.HOMOTYPIC: 0.8,
                PathogenMatch.PARTIAL_HETERO: 0.65,
                PathogenMatch.COMPLETE_HETERO: 0.35,
            },

            vaccination_single_dose_waning_rate = 365 / 273,  # 365/1273
            vaccination_double_dose_waning_rate = 365 / 546,  # 365/2600
            # vaccination_waning_lower_bound = 20 * 7 / 365.0,

            # Tau leap parametes
            tau = 1 / 365.0,

            # if initialization starts with a proportion of immune agents:
            num_initial_immune = 10000,
        )

        self.define_states(
            ss.BoolArr('is_reassortant'),  # is the pathogen a reassortant
            ss.BoolArr('is_severe'),  # is the pathogen severe
            ss.BoolArr('is_infected'),
            ss.FloatArr('strain'),
            #ss.Arr('infecting_pathogen', default=[]),  # list of pathogens infecting the host
            rsa.MultiList('infecting_pathogen', default=[]),  # Ages at time of live births
            rsa.MultiList('possibleCombinations', default=[]),  # list of possible reassortant combinations
            rsa.MultiList('immunity', default={}), # set of strains the host is immune to
            rsa.MultiList('vaccine', default = []),
            ss.FloatArr('prior_infections', default=0),
            rsa.MultiList('prior_vaccinations', default = []),
            ss.Arr('vaccine', dtype=Vaccine, default = None),
            rsa.MultiList('infections_with_vaccination', default = []),
            rsa.MultiList('infections_without_vaccination' , default = []),
            ss.BoolArr('is_immune_flag', default = False),
            ss.FloatArr('oldest_infection', default = np.nan),
        )

        # Set filenames
        name_suffix = '%r_%r_%r_%r_%r_%r_%r_%r' % (
        self.pars.immunity_hypothesis, self.pars.reassortment_rate, self.pars.fitness_hypothesis, self.pars.vaccine_hypothesis,
        self.pars.waning_hypothesis, self.pars.initial_immunity, self.pars.ve_i_to_ve_s_ratio, self.pars.experiment_number)
        self.files = sc.objdict()
        self.files.outputfilename = './results/rota_strain_count_%s.csv' % (name_suffix)
        self.files.vaccinations_outputfilename = './results/rota_vaccinecount_%s.csv' % (name_suffix)
        self.files.sample_outputfilename = './results/rota_strains_sampled_%s.csv' % (name_suffix)
        self.files.infected_all_outputfilename = './results/rota_strains_infected_all_%s.csv' % (name_suffix)
        self.files.age_outputfilename = './results/rota_agecount_%s.csv' % (name_suffix)
        self.files.vaccine_efficacy_output_filename = './results/rota_vaccine_efficacy_%s.csv' % (name_suffix)
        self.files.sample_vaccine_efficacy_output_filename = './results/rota_sample_vaccine_efficacy_%s.csv' % (
            name_suffix)


        return

    def init_post(self):
        super().init_post()

        # Reset the seed
        rnd.seed(self.sim.pars.rand_seed)
        np.random.seed(self.sim.pars.rand_seed)

        if self.pars.waning_hypothesis == 1:
            omega = 365 / 273  # duration of immunity by infection= 39 weeks
        elif self.pars.waning_hypothesis == 2:
            omega = 365 / 50
        elif self.pars.waning_hypothesis == 3:
            omega = 365 / 100
        self.pars.omega = omega
        self.pars.birth_rate = self.pars.mu * 2  # Adjust birth rate to be more in line with Bangladesh
        self.reassortmentRate_GP = self.pars.reassortment_rate

        # Final initialization
        self.immunity_counts = 0
        self.reassortment_count = 0
        self.tau_steps = 0
        self.rota_results = sc.objdict(
            columns=["id", "Strain", "CollectionTime", "Age", "Severity", "InfectionTime", "PopulationSize"],
            infected_all=[],
        )

        # relative protection for infection from natural immunity
        immunity_hypothesis = self.pars.immunity_hypothesis

        # Define the mapping of immunity_hypothesis to their corresponding rates using tuples
        immunity_rates = {
            1: (0, 0, 0),
            2: (0, 1, 0),
            3: (0, 0.5, 0),
            4: (0, 0.95, 0.9),
            5: (0, 0.95, 0.90),
            7: (0.95, 0.90, 0.2),
            8: (0.9, 0.5, 0),
            9: (0.9, 0.45, 0.35),
            10: (0.8, 0.45, 0.35),
        }

        # Get the rates for the given immunity_hypothesis
        rates = immunity_rates.get(immunity_hypothesis)

        if rates is None:
            raise NotImplementedError(f"No partial cross immunity rate for immunity hypothesis: {immunity_hypothesis}")

        # Unpack the tuple into the corresponding variables
        homotypic_immunity_rate, partial_cross_immunity_rate, complete_heterotypic_immunity_rate = rates

        self.homotypic_immunity_rate = homotypic_immunity_rate
        self.partial_cross_immunity_rate = partial_cross_immunity_rate
        self.complete_heterotypic_immunity_rate = complete_heterotypic_immunity_rate

        self.done_vaccinated = False

        self.vaccine_efficacy_i_d1 = {}
        self.vaccine_efficacy_s_d1 = {}
        self.vaccine_efficacy_i_d2 = {}
        self.vaccine_efficacy_s_d2 = {}
        for (k, v) in self.pars.vaccine_efficacy_d1.items():
            (ve_i, ve_s) = self.breakdown_vaccine_efficacy(v, self.pars.ve_i_to_ve_s_ratio)
            self.vaccine_efficacy_i_d1[k] = ve_i
            self.vaccine_efficacy_s_d1[k] = ve_s
        for (k, v) in self.pars.vaccine_efficacy_d2.items():
            (ve_i, ve_s) = self.breakdown_vaccine_efficacy(v, self.pars.ve_i_to_ve_s_ratio)
            self.vaccine_efficacy_i_d2[k] = ve_i
            self.vaccine_efficacy_s_d2[k] = ve_s

        if self.sim.pars.verbose: print("VE_i: ", self.vaccine_efficacy_i_d1)
        if self.sim.pars.verbose: print("VE_s: ", self.vaccine_efficacy_s_d1)

        # Vaccination rates are derived based on the following formula
        self.vaccine_second_dose_rate = 0.8
        self.vaccine_first_dose_rate = math.sqrt(self.vaccine_second_dose_rate)
        if self.sim.pars.verbose: print("Vaccination - first dose rate: %s, second dose rate %s" % (
        self.vaccine_first_dose_rate, self.vaccine_second_dose_rate))

        self.total_strain_counts_vaccine = {}

        # segmentVariants = [[i for i in range(1, 3)], [i for i in range(1, 3)], [i for i in range(1, 2)], [i for i in range(1, 2)]]     ## creating variats for the segments
        segmentVariants = [[1, 2, 3, 4, 9, 11, 12], [8, 4, 6], [i for i in range(1, 2)], [i for i in range(1, 2)]]
        # segmentVariants for the Low baseline diversity setting
        # segmentVariants = [[1,2,3,4,9], [8,4], [i for i in range(1, 2)], [i for i in range(1, 2)]]
        segment_combinations = [tuple(i) for i in itertools.product(
            *segmentVariants)]  # getting all possible combinations from a list of list
        rnd.shuffle(segment_combinations)
        number_all_strains = len(segment_combinations)
        n_init_seg = 100
        initial_segment_combinations = {
            (1, 8, 1, 1): n_init_seg,
            (2, 4, 1, 1): n_init_seg,
            (9, 8, 1, 1): n_init_seg,
            (4, 8, 1, 1): n_init_seg,
            (3, 8, 1, 1): n_init_seg,
            (12, 8, 1, 1): n_init_seg,
            (12, 6, 1, 1): n_init_seg,
            (9, 4, 1, 1): n_init_seg,
            (9, 6, 1, 1): n_init_seg,
            (1, 6, 1, 1): n_init_seg,
            (2, 8, 1, 1): n_init_seg,
            (2, 6, 1, 1): n_init_seg,
            (11, 8, 1, 1): n_init_seg,
            (11, 6, 1, 1): n_init_seg,
            (1, 4, 1, 1): n_init_seg,
            (12, 4, 1, 1): n_init_seg,
        }
        # initial strains for the Low baseline diversity setting
        # initial_segment_combinations = {(1,8,1,1): 100, (2,4,1,1): 100} #, (9,8,1,1): 100} #, (4,8,1,1): 100}

        # Track the number of immune hosts(immunity_counts) in the host population
        infected_uids = []
        pathogens_uids = []

        # for each strain track the number of hosts infected with it at current time: strain_count
        strain_count = {}

        self.to_be_vaccinated_uids = []
        self.single_dose_vaccinated_uids = []

        # Store these for later
        self.infected_uids = infected_uids
        self.pathogens_uids = pathogens_uids
        self.strain_count = strain_count

        for i in range(number_all_strains):
            self.strain_count[segment_combinations[i]] = 0

        # if initial immunity is true
        if self.sim.pars.verbose:
            if self.pars.initial_immunity:
                print("Initial immunity is set to True")
            else:
                print("Initial immunity is set to False")

        ### infecting the initial infecteds
        for (initial_strain, num_infected) in initial_segment_combinations.items():
            if self.pars.initial_immunity:
                for j in range(self.pars.num_initial_immune):
                    h = rnd.choice(self.people)
                    h.immunity[initial_strain] = self.t
                    self.immunity_counts += 1
                    h.is_immune_flag = True

            # This does NOT guarantee num_infected will be the starting number of infected hosts. Need a random choice
            # without replacement for that.
            for j in range(num_infected):
                h = int(rnd.choice(self.sim.people.uid))
                if not self.isInfected(h):
                    infected_uids.append(h)
                p = RotaPathogen(self.sim, False, self.t.abstvec[self.ti], host_uid=h, strain=initial_strain)
                pathogens_uids.append(p)
                self.infecting_pathogen[h].append(p)
                strain_count[p.strain] += 1
        if self.sim.pars.verbose: print(strain_count)

        if self.pars.to_csv:
            self.initialize_files(strain_count)

        self.tau_steps = 0
        self.last_data_colllected = 0
        self.data_collection_rate = 0.1

        for strain, count in strain_count.items():
            if strain[:self.pars.numAgSegments] in self.total_strain_counts_vaccine:
                self.total_strain_counts_vaccine[strain[:self.pars.numAgSegments]] += count
            else:
                self.total_strain_counts_vaccine[strain[:self.pars.numAgSegments]] = count


    # Initialize all the output files
    def initialize_files(self, strain_count):
        print('Initializing files')
        sc.makefilepath('./results/', makedirs=True) # Ensure results folder exists
        files = self.files
        with open(files.outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["time"] + list(strain_count.keys()) + ["reassortment_count"])  # header for the csv file
            write.writerow([self.t] + list(strain_count.values()) + [self.reassortment_count])  # first row of the csv file will be the initial state

        with open(files.sample_outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["id", "Strain", "CollectionTime", "Age", "Severity", "InfectionTime", "PopulationSize"])
        with open(files.infected_all_outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["id", "Strain", "CollectionTime", "Age", "Severity", "InfectionTime", "PopulationSize"])
        with open(files.vaccinations_outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["id", "VaccineStrain", "CollectionTime", "Age", "Dose"])  # header for the csv file

        for outfile in [files.vaccine_efficacy_output_filename, files.sample_vaccine_efficacy_output_filename]:
            with open(outfile, "w+", newline='') as outputfile:
                write = csv.writer(outputfile)
                write.writerow(["CollectionTime", "Vaccinated", "Unvaccinated", "VaccinatedInfected", "VaccinatedSevere", "UnVaccinatedInfected", "UnVaccinatedSevere",
                                "VaccinatedHomotypic", "VaccinatedHomotypicSevere", "VaccinatedpartialHetero", "VaccinatedpartialHeteroSevere", "VaccinatedFullHetero", "VaccinatedFullHeteroSevere"])

        with open(files.age_outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["time"] + list(age_labels))


    def step(self):
        """
                Perform the actual integration loop
                """
        self.event_dict = sc.objdict(
            births=0,
            deaths=0,
            recoveries=0,
            contacts=0,
            wanings=0,
            reassortments=0,
            vaccine_dose_1_wanings=0,
            vaccine_dose_2_wanings=0,
        )

        if self.tau_steps % 10 == 0:
            if self.sim.pars.verbose is not False: print(
                f"Year: {self.t.abstvec[self.ti]}; step: {self.tau_steps}; hosts: {len(self.sim.people)}; elapsed: {self.T.total} s")
            if self.sim.pars.verbose: print(self.strain_count)

        ### Every 100 steps, write the age distribution of the population to a file
        if self.tau_steps % 100 == 0:
            age_dict = {}

            binned_ages = np.digitize(self.sim.people.age, age_bins)
            bin_counts = np.bincount(binned_ages, minlength=len(age_bins)+1)
            for i in np.arange(len(age_labels)):
                age_dict[age_labels[i]] = bin_counts[i]

            if self.sim.pars.verbose: print("Ages: ", age_dict)
            if self.pars.to_csv:
                with open(self.files.age_outputfilename, "a", newline='') as outputfile:
                    write = csv.writer(outputfile)
                    write.writerow(["{:.2}".format(self.t.timevec[self.ti])] + list(age_dict.values()))

        # Count the number of hosts with 1 or 2 vaccinations
        single_dose_hosts = []
        double_dose_hosts = []

        vaccinated_inds = [index for index, value in enumerate(self.vaccine.values) if value is not None]
        if len(vaccinated_inds):
            single_dose_hosts = (self.vaccine.raw[vaccinated_inds][2]==1).auids
            double_dose_hosts = (self.vaccine.raw[vaccinated_inds][2]==2).auids

        # Get the number of events in a single tau step
        events = self.get_event_counts(len(self.sim.people), len(self.infected_uids), self.immunity_counts, self.pars.tau,
                                       self.reassortmentRate_GP, len(single_dose_hosts), len(double_dose_hosts))
        births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings, vaccine_dose_2_wanings = events
        if self.sim.pars.verbose: print(
            "t={}, births={}, deaths={}, recoveries={}, contacts={}, wanings={}, reassortments={}, waning_vaccine_d1={}, waning_vaccine_d2={}".format(
                self.t.abstvec[self.ti], births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings,
                vaccine_dose_2_wanings))

        # Parse into dict
        self.event_dict[:] += events

        # perform the events for the obtained counts
        self.birth_events(births)
        self.reassortment_event(self.infected_uids, reassortments)  # calling the function
        counter = self.contact_event(contacts, self.infected_uids, self.strain_count)
        self.death_event(deaths, self.infected_uids, self.strain_count)
        self.recovery_event(recoveries, self.infected_uids, self.strain_count)
        self.waning_event(wanings)
        self.waning_vaccinations_first_dose(single_dose_hosts, vaccine_dose_1_wanings)
        self.waning_vaccinations_second_dose(double_dose_hosts, vaccine_dose_2_wanings)

        # Collect the total counts of strains at each time step to determine the most prevalent strain for vaccination
        if not self.done_vaccinated:
            for strain, count in self.strain_count.items():
                self.total_strain_counts_vaccine[strain[:self.pars.numAgSegments]] += count

        # Administer the first dose of the vaccine
        # Vaccination strain is the most prevalent strain in the population before the vaccination starts
        vaccinated_strain = sorted(list(self.total_strain_counts_vaccine.keys()), key=lambda x: self.total_strain_counts_vaccine[x])[-1]
        if (self.pars.vaccine_hypothesis != 0) and (not self.done_vaccinated) and (self.t.abstvec[self.ti] >= self.pars.vaccination_time):
            # Sort the strains by the number of hosts infected with it in the past
            # Pick the last one from the sorted list as the most prevalent strain

            # Select hosts under 6.5 weeks and over 4.55 weeks of age for vaccinate
            # child_host_pop = [h for h in host_pop if self.age <= 0.13 and self.age >= 0.09]
            child_host_uids = (self.sim.people.age <= 0.13 and self.sim.people.age >= 0.9).uids
            # Use the vaccination rate to determine the number of hosts to vaccinate
            vaccination_count = int(len(child_host_uids) * self.vaccine_first_dose_rate)
            sample_population_uids = rnd.sample(child_host_uids, vaccination_count)
            if self.sim.pars.verbose: print("Vaccinating with strain: ", vaccinated_strain, vaccination_count)
            if self.sim.pars.verbose: print(
                "Number of people vaccinated: {} Number of people under 6 weeks: {}".format(len(sample_population_uids),
                                                                                            len(child_host_uids)))
            for uid in sample_population_uids:
                self.vaccinate(uid, vaccinated_strain)
                self.single_dose_vaccinated_uids.append(uid)
            self.done_vaccinated = True
        elif self.done_vaccinated:
            for child_uid in self.to_be_vaccinated_uids:
                if self.sim.people.age[child_uid] >= 0.11:
                    self.vaccinate(child_uid, vaccinated_strain)
                    self.to_be_vaccinated_uids.remove(child_uid)
                    self.single_dose_vaccinated_uids.append(child_uid)

        # Administer the second dose of the vaccine if first dose has already been administered.
        # The second dose is administered 6 weeks after the first dose with probability vaccine_second_dose_rate
        if self.done_vaccinated:
            while len(self.single_dose_vaccinated_uids) > 0:
                # If the first dose of the vaccine is older than 6 weeks then administer the second dose
                if self.t.abstvec[self.ti] - self.vaccine[self.single_dose_vaccinated_uids[0]][1] >= 0.11:
                    child_uid = self.single_dose_vaccinated_uids.pop(0)
                    if rnd.random() < self.pars.vaccine_second_dose_rate:
                        self.vaccinate(child_uid, vaccinated_strain)
                else:
                    break

        f = self.files
        if self.t.abstvec[self.ti] >= self.last_data_colllected:
            self.collect_and_write_data(f.sample_outputfilename, f.vaccinations_outputfilename,
                                        f.sample_vaccine_efficacy_output_filename, sample=True)
            self.collect_and_write_data(f.infected_all_outputfilename, f.vaccinations_outputfilename,
                                        f.vaccine_efficacy_output_filename, sample=False)
            self.last_data_colllected += self.data_collection_rate

        if self.pars.to_csv:
            with open(f.outputfilename, "a", newline='') as outputfile:
                write = csv.writer(outputfile)
                write.writerow([self.t] + list(self.strain_count.values()) + [self.reassortment_count])

        self.tau_steps += 1
        # self.t += self.tau

        # if self.sim.pars.verbose > 0:
        #     self.T.toc()
        #     print(self.event_dict)
        # return self.event_dict

    # compares two strains
    # if they both have the same antigenic segments we return homotypic
    def match(self, strainIn):
        numAgSegments = self.pars.numAgSegments
        if strainIn[:numAgSegments] == self.strain[:numAgSegments]:
            return PathogenMatch.HOMOTYPIC

        strains_match = False
        for i in range(numAgSegments):
            if strainIn[i] == self.strain[i]:
                strains_match = True

        if strains_match:
            return PathogenMatch.PARTIAL_HETERO
        else:
            return PathogenMatch.COMPLETE_HETERO


    def get_strain_name(self):
        G,P,A,B = [str(self.strain[i]) for i in range(4)]
        return f'G{G}P{P}A{A}B{B}'

    def get_probability_of_severe(self, pathogen_in, vaccine, immunity_count):  # TEMP: refactor and include above
        if immunity_count >= 3:
            severity_probability = 0.18
        elif immunity_count == 2:
            severity_probability = 0.24
        elif immunity_count == 1:
            severity_probability = 0.23
        elif immunity_count == 0:
            severity_probability = 0.17

        if vaccine is not None:
            # Probability of severity also depends on the strain (homotypic/heterltypic/etc.)
            pathogen_strain_type = pathogen_in.match(vaccine[0][0])
            # Effectiveness of the vaccination depends on the number of doses
            if vaccine[2] == 1:
                ve_s = self.pars.vaccine_efficacy_s_d1[pathogen_strain_type]
            elif vaccine[2] == 2:
                ve_s = self.pars.vaccine_efficacy_s_d2[pathogen_strain_type]
            else:
                raise NotImplementedError(f"Unsupported vaccine dose: {vaccine[2]}")
            return severity_probability * (1 - ve_s)
        else:
            return severity_probability

        ############# tau-Function to calculate event counts ############################

    def get_event_counts(self, N, I, R, tau, RR_GP, single_dose_count, double_dose_count):
        births = np.random.poisson(size=1, lam=tau * N * self.pars.birth_rate)[0]
        deaths = np.random.poisson(size=1, lam=tau * N * self.pars.mu)[0]
        recoveries = np.random.poisson(size=1, lam=tau * self.pars.gamma * I)[0]
        contacts = np.random.poisson(size=1, lam=tau * self.pars.contact_rate * I)[0]
        wanings = np.random.poisson(size=1, lam=tau * self.pars.omega * R)[0]
        reassortments = np.random.poisson(size=1, lam=tau * RR_GP * I)[0]
        vaccination_wanings_one_dose = \
        np.random.poisson(size=1, lam=tau * self.pars.vaccination_single_dose_waning_rate * single_dose_count)[0]
        vaccination_wanings_two_dose = \
        np.random.poisson(size=1, lam=tau * self.pars.vaccination_double_dose_waning_rate * double_dose_count)[0]
        return (births, deaths, recoveries, contacts, wanings, reassortments, vaccination_wanings_one_dose,
                vaccination_wanings_two_dose)

    def coInfected_contacts(self, h1_uid, h2_uid, strain_counts):
        h2existing_pathogens = list(self.infecting_pathogen[h2_uid])
        randomnumber = rnd.random()
        if randomnumber < 0.02:  # giving all the possible strains
            for path in self.infecting_pathogen[h1_uid]:
                if self.can_variant_infect_host(h2_uid, path.strain):
                    self.infect_with_pathogen(h2_uid, path, strain_counts)
        else:  # give only one strain depending on fitness
            host1paths = list(self.infecting_pathogen[h1_uid])
            # Sort by fitness first and randomize the ones with the same fitness
            host1paths.sort(key=lambda path: (path.get_fitness(), rnd.random()), reverse=True)
            for path in host1paths:
                if self.can_variant_infect_host(h2_uid, path.strain):
                    infected = self.infect_with_pathogen(h2_uid, path, strain_counts)
                    if infected:
                        break

    def isInfected(self, uid):
        return len(self.infecting_pathogen[uid]) != 0

    def get_weights_by_age(self):
        # bdays = np.array(self.host_pop.bdays)
        # weights = self.t - bdays
        weights = self.sim.people.age.values
        total_w = np.sum(weights)
        weights = weights / total_w
        return weights

    def birth_events(self, birth_count):
        for _ in range(birth_count):
            # self.pop_id += 1
            # new_host = Host(self.pop_id, sim=self)
            # new_host.bday = self.t
            # self.host_pop.append(new_host)
            new_uids = self.sim.people.grow(birth_count) # add more people!
            self.sim.people.age[new_uids] = 0

            if self.pars.vaccine_hypothesis != 0 and self.done_vaccinated:
                if rnd.random() < self.vaccine_first_dose_rate:
                    self.to_be_vaccinated_uids.append(new_uids)

    def death_event(self, num_deaths, infected_uids, strain_count):
        # host_list = np.arange(len(self.host_pop))
        p = self.get_weights_by_age()
        dying_uids = np.random.choice(self.sim.people.alive.uids, p=p, size=num_deaths, replace=False)
        # dying_hosts = [self.host_pop[ind] for uid in uids]
        for uid in dying_uids:
            if self.isInfected(uid):
                infected_uids.remove(uid)
                for path in self.infecting_pathogen[uid]:
                    if not path.is_reassortant:
                        self.strain_count[path.strain] -= 1
            if self.is_immune_flag[uid]:
                self.immunity_counts -= 1
            self.sim.people.request_death(uid)  # remove the host from the simulation
        return

    def recovery_event(self, num_recovered, infected_uids, strain_count):
        weights = np.array([self.get_oldest_current_infection(x) for x in infected_uids])
        # If there is no one with an infection older than 0 return without recovery
        if (sum(weights) == 0):
            return
        # weights_e = np.exp(weights)
        total_w = np.sum(weights)
        weights = weights / total_w

        recovering_hosts_uids = np.random.choice(infected_uids, p=weights, size=num_recovered, replace=False)
        # for host in recovering_hosts:
        #     if not host.is_immune_flag:
        #         self.immunity_counts += 1
        #     host.recover(strain_count)
        #     infected_uids.remove(host)
        #
        not_immune = recovering_hosts_uids[self.is_immune_flag[ss.uids(recovering_hosts_uids)] == False]
        self.immunity_counts += len(not_immune)

        self.recover(recovering_hosts_uids, strain_count)
        for recovering_host_uid in recovering_hosts_uids:
            infected_uids.remove(recovering_host_uid)

    def contact_event(self, contacts, infected_uids, strain_count):
        if len(infected_uids) == 0:
            print("[Warning] No infected hosts in a contact event. Skipping")
            return

        h1_uids = np.random.choice(infected_uids, size=contacts)
        h2_uids = np.random.choice(self.sim.people.alive.uids, size=contacts)
        rnd_nums = np.random.random(size=contacts)
        counter = 0

        # based on prior infections and current infections, the relative risk of subsequent infections
        infecting_probability_map = {
            0: 1,
            1: 0.61,
            2: 0.48,
            3: 0.33,
        }

        for h1_uid, h2_uid, rnd_num in zip(h1_uids, h2_uids, rnd_nums):
            # h1 = infected_pop[h1_ind]
            # h2 = self.host_pop[h2_ind]

            # If the contact is the same as the infected host, pick another host at random
            while h1_uid == h2_uid:
                h2_uid = rnd.choice(self.sim.people.alive.uids)

            infecting_probability = infecting_probability_map.get(self.prior_infections[h2_uid], 0)
            infecting_probability *= self.pars.rel_beta  # Scale by this calibration parameter

            # No infection occurs
            if rnd_num > infecting_probability:
                continue  # CK: was "return", which was a bug!
            else:
                counter += 1
                h2_previously_infected = self.isInfected(uid=h2_uid)

                if len(self.infecting_pathogen[h1_uid]) == 1:
                    if self.can_variant_infect_host(h2_uid, self.infecting_pathogen[h1_uid][0].strain):
                        self.infect_with_pathogen(h2_uid, self.infecting_pathogen[h1_uid][0], strain_count)
                    # else:
                    #     print('Unclear what should happen here')
                else:
                    self.coInfected_contacts(h1_uid, h2_uid, strain_count)

                # in this case h2 was not infected before but is infected now
                if not h2_previously_infected and self.isInfected(h2_uid):
                    infected_uids.append(h2_uid)
        return counter

    def reassortment_event(self, infected_uids, reassortment_count):
        coinfectedhosts = []
        for i in infected_uids:
            if len(self.infecting_pathogen[i]) >= 2:
                coinfectedhosts.append(i)
        rnd.shuffle(coinfectedhosts)  # TODO: maybe replace this

        for i in range(min(len(coinfectedhosts), reassortment_count)):
            #parentalstrains = [path.strain for path in coinfectedhosts[i].infecting_pathogen]
            parentalstrains = [path.strain for path in self.infecting_pathogen[coinfectedhosts[i]]]
            #possible_reassortants = [path for path in coinfectedhosts[i].compute_combinations() if
            #                         path not in parentalstrains]
            possible_reassortants = [path for path in self.compute_combinations(coinfectedhosts[i]) if
                                     path not in parentalstrains]
            for path in possible_reassortants:
                self.infect_with_reassortant(coinfectedhosts[i], path)

    def waning_event(self, wanings):
        # Get all the hosts in the population that has an immunity
        h_immune_uids = (self.is_immune_flag).uids
        #h_immune = [h for h in self.host_pop if h.is_immune_flag]
        # oldest = np.array([h.oldest_infection for h in h_immune])
        oldest_infections = self.oldest_infection[h_immune_uids]
        # oldest += 1e-6*np.random.random(len(oldest)) # For tiebreaking -- not needed
        order = np.argsort(oldest_infections, stable=True)

        # For the selected hosts set the immunity to be None
        for i in order[:wanings]:  # range(min(len(hosts_with_immunity), wanings)):
            if i == 33461:
                print("waning!")
            h_uid = h_immune_uids[i]
            self.immunity[h_uid] = {}
            self.is_immune_flag[h_uid] = False
            self.oldest_infection[h_uid] = np.nan
            self.prior_infections[h_uid] = 0
            self.immunity_counts -= 1

    def waning_vaccinations_first_dose(self, single_dose_uids, wanings):
        """ Get all the hosts in the population that has an vaccine immunity """
        rnd.shuffle(single_dose_uids)
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(single_dose_uids), wanings)):
            h_uid = single_dose_uids[i]
            self.vaccinations[h_uid] = None

    def waning_vaccinations_second_dose(self, second_dose_uids, wanings):
        rnd.shuffle(second_dose_uids)
        # For the selected hosts set the immunity to be None
        for i in range(min(len(second_dose_uids), wanings)):
            h_uid = second_dose_uids[i]
            self.vaccinations[h_uid] = None

    @staticmethod
    def get_strain_antigenic_name(strain):
        return "G" + str(strain[0]) + "P" + str(strain[1])

    @staticmethod
    def solve_quadratic(a, b, c):
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            root1 = (-b + discriminant ** 0.5) / (2 * a)
            root2 = (-b - discriminant ** 0.5) / (2 * a)
            return tuple(sorted([root1, root2]))
        else:
            return "No real roots"

    def breakdown_vaccine_efficacy(self, ve, x):
        (r1, r2) = self.solve_quadratic(x, -(1 + x), ve)
        if self.sim.pars.verbose: print(r1, r2)
        if r1 >= 0 and r1 <= 1:
            ve_s = r1
        elif r2 >= 0 and r2 <= 1:
            ve_s = r2
        else:
            raise RuntimeError("No valid solution to the equation: x: %d, ve: %d. Solutions: %f %f" % (x, ve, r1, r2))
        ve_i = x * ve_s
        return (ve_i, ve_s)

    def compute_combinations(self, uid):
        seg_combinations = []

        # We want to only reassort the GP types
        # Assumes that antigenic segments are at the start
        for i in range(self.pars.numAgSegments):
            availableVariants = set([])
            for j in self.infecting_pathogen[uid]:
                availableVariants.add((j.strain[i]))
            seg_combinations.append(availableVariants)

        # compute the parental strains
        parantal_strains = [j.strain[:self.pars.numAgSegments] for j in self.infecting_pathogen[uid]]

        # Itertools product returns all possible combinations
        # We are only interested in strain combinations that are reassortants of the parental strains
        # We need to skip all existing combinations from the parents
        # Ex: (1, 1, 2, 2) and (2, 2, 1, 1) should not create (1, 1, 1, 1) as a possible reassortant if only the antigenic parts reassort

        # below block is for reassorting antigenic segments only
        all_antigenic_combinations = [i for i in itertools.product(*seg_combinations) if i not in parantal_strains]
        all_nonantigenic_combinations = [j.strain[self.pars.numAgSegments:] for j in self.infecting_pathogen[uid]]
        all_strains = set([(i[0] + i[1]) for i in itertools.product(all_antigenic_combinations, all_nonantigenic_combinations)])
        all_pathogens = [RotaPathogen(self.sim, True, self.t.abstvec[self.ti], host_uid = uid, strain=tuple(i)) for i in all_strains]

        return all_pathogens

    def get_oldest_current_infection(self, uid):
        max_infection_times = max([self.t.abstvec[self.ti] - p.creation_time for p in self.infecting_pathogen[uid]])
        return max_infection_times

    def recover(self, uids, strain_counts):
        # We will use the pathogen creation time to count the number of infections
        for uid in uids:
            creation_times = set()
            for path in self.infecting_pathogen[uid]:
                strain = path.strain
                if not path.is_reassortant:
                    strain_counts[strain] -= 1
                    creation_times.add(path.creation_time)
                    self.immunity[uid][strain] = self.t.abstvec[self.ti]
                    self.is_immune_flag[uid] = True
                    if np.isnan(self.oldest_infection[uid]):
                        self.oldest_infection[uid] = self.t.abstvec[self.ti]
            self.prior_infections[uid] += len(creation_times)
            self.infecting_pathogen[uid] = []
            self.possibleCombinations[uid] = []

    def vaccinate(self, uid, vaccinated_strain):
        if len(self.prior_vaccinations[uid]) == 0:
            self.prior_vaccinations[uid].append(vaccinated_strain)
            self.vaccine[uid] = ([vaccinated_strain], self.t.abstvec[self.ti], 1)
        else:
            self.prior_vaccinations[uid].append(vaccinated_strain)
            self.vaccine[uid] = ([vaccinated_strain], self.t.abstvec[self.ti], 2)

    def collect_and_write_data(self, output_filename, vaccine_output_filename, vaccine_efficacy_output_filename, sample=False, sample_size=1000):
        """
        Collects data from the host population and writes it to a CSV file.
        If sample is True, it collects data from a random sample of the population.

        Args:
        - output_filename: Name of the file to write the data.
        - sample: Boolean indicating whether to collect data from a sample or the entire population.
        - sample_size: Size of the sample to collect data from if sample is True.
        """


        # Select the population to collect data from
        if sample:
            population_to_collect = np.random.choice(self.sim.people.alive.uids, sample_size, replace=False)
        else:
            population_to_collect = self.sim.people.alive.uids

        # Shuffle the population to avoid the need for random sampling
        # rnd.shuffle(population_to_collect) # CK: not needed?

        collected_data = []
        collected_vaccination_data = []

        # To measure vaccine efficacy we will gather data on the number of vaccinated hosts who get infected
        # along with the number of unvaccinated hosts that get infected
        vaccinated_uids = []
        unvaccinated_uids = []

        for uid in population_to_collect:
            if not sample:
                # For vaccination data file, we will count the number of agents with current vaccine immunity
                # This will exclude those who previously got the vaccine but the immunity waned.
                if self.vaccine[uid] is not None:
                    for vs in [self.get_strain_antigenic_name(s) for s in self.vaccine[uid][0]]:
                        collected_vaccination_data.append((uid, vs, self.t, self.get_age_category(uid), self.vaccine[uid][2]))
            if len(self.prior_vaccinations[uid]) != 0:
                if len(vaccinated_uids) < 1000:
                    vaccinated_uids.append(uid)
            else:
                if len(unvaccinated_uids) < 1000:
                    unvaccinated_uids.append(uid)
            if self.isInfected(uid):
                strain_str = [(path.get_strain_name(), path.is_severe, path.creation_time) for path in self.infecting_pathogen[uid] if not sample or not path.is_reassortant]
                for strain in strain_str:
                    collected_data.append((uid, strain[0], self.t.abstvec[self.ti], self.get_age_category(uid), strain[1], strain[2], len(self.sim.people.alive.uids)))

        # Only collect the vaccine efficacy data if we have vaccinated the hosts
        if self.done_vaccinated:
            num_vaccinated = len(vaccinated_uids)
            num_unvaccinated = len(unvaccinated_uids)
            num_vaccinated_infected = 0
            num_unvaccinated_infected = 0
            num_vaccinated_infected_severe = 0
            num_unvaccinated_infected_severe = 0
            num_full_heterotypic = [0, 0]
            num_partial_heterotypic = [0, 0]
            num_homotypic = [0, 0]

            for vaccinated_host in vaccinated_uids:
                if len(self.infections_with_vaccination[vaccinated_host]) > 0:
                    num_vaccinated_infected += 1
                was_there_a_severe_infection = False
                was_there_a_full_heterotypic_infection = [False, False]
                was_there_a_partial_heterotypic_infection = [False, False]
                was_there_a_homotypic_infection = [False, False]
                for infecting_pathogen in self.infections_with_vaccination[vaccinated_host]:
                    index = 0
                    if infecting_pathogen[0].is_severe:
                        index = 1
                        was_there_a_severe_infection = True
                    if infecting_pathogen[1] == PathogenMatch.HOMOTYPIC:
                        was_there_a_full_heterotypic_infection[index] = True
                    elif infecting_pathogen[1] == PathogenMatch.PARTIAL_HETERO:
                        was_there_a_partial_heterotypic_infection[index] = True
                    elif infecting_pathogen[1] == PathogenMatch.COMPLETE_HETERO:
                        was_there_a_homotypic_infection[index] = True

                if was_there_a_severe_infection:
                    num_vaccinated_infected_severe += 1
                if was_there_a_full_heterotypic_infection[0]:
                    num_full_heterotypic[0] += 1
                if was_there_a_full_heterotypic_infection[1]:
                    num_full_heterotypic[1] += 1
                if was_there_a_partial_heterotypic_infection[0]:
                    num_partial_heterotypic[0] += 1
                if was_there_a_partial_heterotypic_infection[1]:
                    num_partial_heterotypic[1] += 1
                if was_there_a_homotypic_infection[0]:
                    num_homotypic[0] += 1
                if was_there_a_homotypic_infection[1]:
                    num_homotypic[1] += 1

            for unvaccinated_host in unvaccinated_uids:
                if len(self.infections_without_vaccination[unvaccinated_host]) > 0:
                    num_unvaccinated_infected += 1
                was_there_a_severe_infection = False
                for infecting_pathogen in self.infections_without_vaccination[unvaccinated_host]:
                    if infecting_pathogen.is_severe:
                        was_there_a_severe_infection = True
                        break
                if was_there_a_severe_infection:
                    num_unvaccinated_infected_severe += 1

            if self.to_csv:
                with open(vaccine_efficacy_output_filename, "a", newline='') as outputfile:
                    write = csv.writer(outputfile)
                    write.writerow([self.t, num_vaccinated, num_unvaccinated, num_vaccinated_infected, num_vaccinated_infected_severe, num_unvaccinated_infected, num_unvaccinated_infected_severe,
                                    num_homotypic[0], num_homotypic[1], num_partial_heterotypic[0], num_partial_heterotypic[1], num_full_heterotypic[0], num_full_heterotypic[1]])

        # Write collected data to the output file
        if self.pars.to_csv:
            with open(output_filename, "a", newline='') as outputfile:
                writer = csv.writer(outputfile)
                writer.writerows(collected_data)
        if not sample:
            self.rota_results.infected_all.extend(collected_data)
            if self.pars.to_csv:
                with open(vaccine_output_filename, "a", newline='') as outputfile:
                    writer = csv.writer(outputfile)
                    writer.writerows(collected_vaccination_data)

    def get_age_category(self, uid):
        # Bin the age into categories
        for i in range(len(age_bins)):
            if self.sim.people.age[uid] < age_bins[i]:
                return age_labels[i]
        return age_labels[-1]

    def can_variant_infect_host(self, uid, infecting_strain):
        current_infections = self.infecting_pathogen[uid]
        numAgSegments = self.pars.numAgSegments
        immunity_hypothesis = self.pars.immunity_hypothesis
        partial_cross_immunity_rate = self.partial_cross_immunity_rate
        complete_heterotypic_immunity_rate = self.complete_heterotypic_immunity_rate
        homotypic_immunity_rate = self.homotypic_immunity_rate

        if self.vaccine[uid] is not None and self.is_vaccine_immune(uid, infecting_strain):
            return False

        current_infecting_strains = (i.strain[:numAgSegments] for i in current_infections)
        if infecting_strain[:numAgSegments] in current_infecting_strains:
            return False

        def is_completely_immune():
            immune_strains = (s[:numAgSegments] for s in self.immunity[uid].keys())
            return infecting_strain[:numAgSegments] in immune_strains

        def has_shared_genotype():
            for i in range(numAgSegments):
                immune_genotypes = (strain[i] for strain in self.immunity[uid].keys())
                if infecting_strain[i] in immune_genotypes:
                    return True
            return False


        if immunity_hypothesis == 1:
            if is_completely_immune():
                return False
            return True

        elif immunity_hypothesis == 2:
            if has_shared_genotype():
                return False
            return True

        elif immunity_hypothesis in [3, 4, 7, 8, 9, 10]:
            if is_completely_immune():
                if immunity_hypothesis in [7, 8, 9, 10]:
                    if rnd.random() < homotypic_immunity_rate:
                        return False
                else:
                    return False

            if has_shared_genotype():
                if rnd.random() < partial_cross_immunity_rate:
                    return False
            elif immunity_hypothesis in [4, 7, 8, 9, 10]:
                if rnd.random() < complete_heterotypic_immunity_rate:
                    return False
            return True

        elif immunity_hypothesis == 5:
            immune_ptypes = (strain[1] for strain in self.immunity.keys())
            return infecting_strain[1] not in immune_ptypes

        elif immunity_hypothesis == 6:
            immune_ptypes = (strain[0] for strain in self.immunity.keys())
            return infecting_strain[0] not in immune_ptypes

        else:
            raise NotImplementedError(f"Immunity hypothesis {immunity_hypothesis} is not implemented")

    def infect_with_pathogen(self, uid, pathogen_in, strain_counts):
        """ This function returns a fitness value to a strain based on the hypothesis """
        fitness = pathogen_in.get_fitness()

        # e.g. fitness = 0.8 (there's a 80% chance the virus infecting a host)
        if rnd.random() > fitness:
            return False

        # Probability of getting a severe decease depends on the number of previous infections and vaccination status of the host
        severity_probability = self.get_probability_of_severe(pathogen_in, self.vaccine[uid], self.prior_infections[uid])
        if rnd.random() < severity_probability:
            severe = True
        else:
            severe = False

        new_p = RotaPathogen(self.sim, False, self.t.abstvec[self.ti], host_uid=uid, strain=pathogen_in.strain, is_severe=severe)
        self.infecting_pathogen[uid].append(new_p)
        self.record_infection(new_p)

        strain_counts[new_p.strain] += 1

        return True

    def infect_with_reassortant(self, uid, reassortant_virus):
        self.infecting_pathogen[uid].append(reassortant_virus)

    def record_infection(self, new_p):
        uid = new_p.host_uid
        if len(self.prior_vaccinations[uid]) != 0:
            vaccine_strain = self.prior_vaccinations[uid][-1]
            self.infections_with_vaccination[uid].append((new_p, new_p.match(vaccine_strain)))
        else:
            self.infections_without_vaccination[uid].append(new_p)


    def __str__(self):
        return "Strain: " + self.get_strain_name() + " Severe: " + str(self.is_severe) + " Host: " + str(self.host.id) + str(self.creation_time)