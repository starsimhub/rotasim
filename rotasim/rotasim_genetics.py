from collections import defaultdict
import csv
import itertools
import math

import numpy as np
import random as rnd
import sciris as sc
import starsim as ss

from . import arrays as rsa

__all__ = ["Rota"]

# Define age bins and labels
age_bins = [2 / 12, 4 / 12, 6 / 12, 12 / 12, 24 / 12, 36 / 12, 48 / 12, 60 / 12, 100]
age_distribution = [
    0.006,
    0.006,
    0.006,
    0.036,
    0.025,
    0.025,
    0.025,
    0.025,
    0.846,
]  # needs to be changed to fit the site-specific population
age_labels = ["0-2", "2-4", "4-6", "6-12", "12-24", "24-36", "36-48", "48-60", "60+"]


### Pathogen classes


class PathogenMatch:
    """Define whether pathogens are completely heterotypic, partially heterotypic, or homotypic"""

    COMPLETE_HETERO = 1
    PARTIAL_HETERO = 2
    HOMOTYPIC = 3


def get_strain_name(strain):
    G, P, A, B = [str(strain[i]) for i in range(4)]
    return f"G{G}P{P}A{A}B{B}"



class RotaPathogen(sc.quickobj):
    """
    Pathogen dynamics
    """

    def __init__(
        self,
        rotasim,
        is_reassortant,
        creation_time,
        is_severe=False,
        host_uid=None,
        strain=None,
    ):
        self.rotasim = rotasim  # The Rota module
        self.host_uid = host_uid
        self.creation_time = creation_time
        self.is_reassortant = is_reassortant
        self.strain = strain
        self.is_severe = is_severe
        self.g = strain[0]  # Genotype
        self.p = strain[1]  # Phylogroup
        self.a = strain[2]
        self.b = strain[3]

        return

    # compares two strains
    # if they both have the same antigenic segments we return homotypic
    def match(self, strainIn):
        numAgSegments = self.rotasim.pars.numAgSegments
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
        """Get the fitness based on the fitness hypothesis and the two strains"""
        fitness_hypothesis = self.rotasim.pars.fitness_hypothesis
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
        return f"G{G}P{P}A{A}B{B}"

    def __str__(self):
        return (
            "Strain: "
            + self.get_strain_name()
            + " Severe: "
            + str(self.is_severe)
            + " Host: "
            + str(self.host.id)
            + str(self.creation_time)
        )


class Rota(ss.Module):
    """
    Pathogen dynamics
    """

    def __init__(self, to_csv=False, **kwargs):
        super().__init__()
        self.T = sc.timer()

        numSegments = 4
        numNoneAgSegments = 2
        numAgSegments = numSegments - numNoneAgSegments

        segmentVariants = [
            [1, 2, 3, 4, 9, 11, 12],
            [8, 4, 6],
            [i for i in range(1, 2)],
            [i for i in range(1, 2)],
        ]
        # segmentVariants for the Low baseline diversity setting
        # segmentVariants = [[1,2,3,4,9], [8,4], [i for i in range(1, 2)], [i for i in range(1, 2)]]

        self.define_pars(
            initial_diversity_setting="low",  # low or high
            num_initial_infected_per_strain=100,  # number of initial infected per strain
            reassortment_rate=0.1,
            fitness_hypothesis=2,
            initial_immunity=False,
            initial_immunity_rate=0.1,
            experiment_number=1,
            rel_beta=1.0,
            fitness_map={
                1: {"default": 1},
                2: {
                    "default": 0.90,
                    (1, 1): 0.93,
                    (2, 2): 0.93,
                    (3, 3): 0.93,
                    (4, 4): 0.93,
                },
                3: {
                    "default": 0.87,
                    (1, 1): 0.93,
                    (2, 2): 0.93,
                    (3, 3): 0.90,
                    (4, 4): 0.90,
                },
                4: {"default": 1, (1, 1): 1, (2, 2): 0.2},
                5: {"default": 0.2, (1, 1): 1, (2, 1): 0.5, (1, 3): 0.5},
                6: {"default": 0.05, (1, 8): 1, (2, 4): 0.2, (3, 8): 0.4, (4, 8): 0.5},
                7: {"default": 0.05, (1, 8): 1, (2, 4): 0.3, (3, 8): 0.7, (4, 8): 0.6},
                8: {"default": 0.05, (1, 8): 1, (2, 4): 0.4, (3, 8): 0.9, (4, 8): 0.8},
                9: {"default": 0.2, (1, 8): 1, (2, 4): 0.5, (3, 8): 0.9, (4, 8): 0.8},
                10: {"default": 0.4, (1, 8): 1, (2, 4): 0.6, (3, 8): 0.9, (4, 8): 0.9},
                11: {
                    "default": 0.5,
                    (1, 8): 0.98,
                    (2, 4): 0.7,
                    (3, 8): 0.8,
                    (4, 8): 0.8,
                },
                12: {
                    "default": 0.5,
                    (1, 8): 0.98,
                    (2, 4): 0.8,
                    (3, 8): 0.9,
                    (4, 8): 0.9,
                },
                13: {
                    "default": 0.7,
                    (1, 8): 0.98,
                    (2, 4): 0.8,
                    (3, 8): 0.9,
                    (4, 8): 0.9,
                },
                14: {
                    "default": 0.05,
                    (1, 8): 0.98,
                    (2, 4): 0.4,
                    (3, 8): 0.7,
                    (4, 8): 0.6,
                    (9, 8): 0.7,
                    (12, 8): 0.75,
                    (9, 6): 0.58,
                    (11, 8): 0.2,
                },
                15: {
                    "default": 0.4,
                    (1, 8): 1,
                    (2, 4): 0.7,
                    (3, 8): 0.93,
                    (4, 8): 0.93,
                    (9, 8): 0.95,
                    (12, 8): 0.94,
                    (9, 6): 0.3,
                    (11, 8): 0.35,
                },
                16: {
                    "default": 0.4,
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
                },
                17: {
                    "default": 0.7,
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
                },
                18: {
                    "default": 0.65,
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
                },
                19: {
                    "default": 0.4,
                    (1, 8): 1,
                    (2, 4): 0.5,
                    (3, 8): 0.55,
                    (4, 8): 0.55,
                    (9, 8): 0.6,
                },
            },
            numAgSegments=numAgSegments,
            to_csv=to_csv,  # whether to write files
            csv_buffer_size=1000,  # buffer size for CSV writes
            csv_flush_interval=100,  # flush every N steps
            mu=1.0 / 70.0,  # average life span is 70 years
            gamma=365 / 7,  # 1/average infectious period (1/gamma =7 days)
            omega=365 / 273,
            birth_rate=0.5 / 70.0,
            contact_rate=365 / 1,
            reassortmentRate_GP=None,
            vaccination_time=15,  # vaccinations begin at this year in the sim

            # Tau leap parameters
            tau=1 / 365.0,
            # default num_initial_immune to 0, will be updated later if needed
            num_initial_immune=0,
            segment_combinations=[
                tuple(i) for i in itertools.product(*segmentVariants)
            ],  # getting all possible combinations from a list of list
            # Time to wait before computing strain counts. Allows the simulation to reach a equilibrium.
            time_to_equilibrium=5,
            # Natural immunity rates: Relative protection to infection from natural immunity
            homotypic_immunity_rate=0.5,
            partial_heterotypic_immunity_rate=0.5,
            complete_heterotypic_immunity_rate=0.5,

        )



        # update the pars based on the kwargs
        self.update_pars(pars=kwargs)

        self.define_states(
            ss.BoolArr("is_reassortant"),  # is the pathogen a reassortant
            ss.BoolArr("is_severe"),  # is the pathogen severe
            ss.BoolArr("is_infected"),
            ss.FloatArr("strain"),
            rsa.MultiList(
                "infecting_pathogen", default=[]
            ),  # Ages at time of live births
            rsa.MultiList(
                "possibleCombinations", default=[]
            ),  # list of possible reassortant combinations
            rsa.MultiList(
                "immunity", default={}
            ),  # set of strains the host is immune to
            ss.FloatArr("prior_infections", default=0),
            rsa.MultiList("prior_vaccinations", default=[]),
            rsa.MultiList("infections_with_vaccination", default=[]),
            rsa.MultiList("infections_without_vaccination", default=[]),
            ss.BoolArr("is_immune_flag", default=False),
            ss.FloatArr("oldest_infection", default=np.nan),
            ss.FloatArr("rel_sus", default=1.0),  # relative susceptibility to infection
            ss.FloatArr("rel_sev", default=1.0),  # relative severity of infection
        )

        # Set filenames
        name_suffix = "".join(
            (
                f"{self.pars.initial_diversity_setting}_",
                f"{self.pars.homotypic_immunity_rate}_",
                f"{self.pars.partial_heterotypic_immunity_rate}_"
                f"{self.pars.complete_heterotypic_immunity_rate}_"
                f"{self.pars.reassortment_rate}_"
                f"{self.pars.fitness_hypothesis}_"
                f"{self.pars.omega}_"
                f"{self.pars.initial_immunity}_"
                # f"{self.pars.ve_i_to_ve_s_ratio}_"
                # f"{self.pars.vaccination_first_dose_waning_rate}_"
                # f"{self.pars.vaccination_second_dose_waning_rate}_"
                f"{self.pars.contact_rate}_"
                f"{self.pars.experiment_number}_"
                # f"{self.pars.vaccine_second_dose_coverage}_",
                # f"{self.pars.vaccine_efficacy_d2[PathogenMatch.HOMOTYPIC]}_",
                # f"{self.pars.vaccine_efficacy_d2[PathogenMatch.PARTIAL_HETERO]}_",
                # f"{self.pars.vaccine_efficacy_d2[PathogenMatch.COMPLETE_HETERO]}_",
                # f"{self.pars.vaccine_efficacy_d1_ratio}",
            )
        )
        self.files = sc.objdict()
        self.file_handles = sc.objdict()
        self.csv_buffers = sc.objdict()
        self.csv_write_counters = sc.objdict()

        self.files.output_filename = "./results/rota_strain_count_%s.csv" % (
            name_suffix
        )
        self.files.vaccinations_output_filename = (
            "./results/rota_vaccinecount_%s.csv" % (name_suffix)
        )
        self.files.sample_output_filename = "./results/rota_strains_sampled_%s.csv" % (
            name_suffix
        )
        self.files.infected_all_output_filename = (
            "./results/rota_strains_infected_all_%s.csv" % (name_suffix)
        )
        self.files.age_output_filename = "./results/rota_agecount_%s.csv" % (
            name_suffix
        )
        self.files.vaccine_efficacy_output_filename = (
            "./results/rota_vaccine_efficacy_%s.csv" % (name_suffix)
        )
        self.files.sample_vaccine_efficacy_output_filename = (
            "./results/rota_sample_vaccine_efficacy_%s.csv" % (name_suffix)
        )
        self.files.event_counts_filename = "./results/event_counts_%s.csv" % (
            name_suffix
        )
        self.files.immunity_file = "./results/immunity_counts_%s.csv" % (name_suffix)

        # todo convert to results
        self.event_dict = sc.objdict(
            births=0,
            deaths=0,
            recoveries=0,
            contacts=0,
            wanings=0,
            reassortments=0,
        )
        return

    def _setup_csv_writers(self):
        """Initialize CSV writers and file handles with proper buffering."""
        if not self.pars.to_csv:
            return

        # Always initialize these if they don't exist
        if not hasattr(self, "file_handles") or not self.file_handles:
            self.file_handles = sc.objdict()
        if not hasattr(self, "csv_buffers") or not self.csv_buffers:
            self.csv_buffers = sc.objdict()
        if not hasattr(self, "csv_write_counters") or not self.csv_write_counters:
            self.csv_write_counters = sc.objdict()

        files = self.files

        # Setup buffered file handles and writers
        file_configs = [
            (
                "output",
                files.output_filename,
                ["time"] + list(self.strain_count.keys()) + ["reassortment_count"],
            ),
            (
                "sample",
                files.sample_output_filename,
                [
                    "id",
                    "Strain",
                    "CollectionTime",
                    "Age",
                    "Severity",
                    "InfectionTime",
                    "PopulationSize",
                ],
            ),
            (
                "infected_all",
                files.infected_all_output_filename,
                [
                    "id",
                    "Strain",
                    "CollectionTime",
                    "Age",
                    "Severity",
                    "InfectionTime",
                    "PopulationSize",
                ],
            ),
            (
                "vaccinations",
                files.vaccinations_output_filename,
                ["id", "VaccineStrain", "CollectionTime", "Age", "VaccinationTime"],
            ),
            (
                "vaccine_efficacy",
                files.vaccine_efficacy_output_filename,
                [
                    "CollectionTime",
                    "Vaccinated",
                    "Unvaccinated",
                    "VaccinatedInfected",
                    "VaccinatedSevere",
                    "UnVaccinatedInfected",
                    "UnVaccinatedSevere",
                    "VaccinatedHomotypic",
                    "VaccinatedHomotypicSevere",
                    "VaccinatedpartialHetero",
                    "VaccinatedpartialHeteroSevere",
                    "VaccinatedFullHetero",
                    "VaccinatedFullHeteroSevere",
                ],
            ),
            ("age", files.age_output_filename, ["time"] + list(age_labels)),
            (
                "event_counts",
                files.event_counts_filename,
                [
                    "time",
                    "births",
                    "deaths",
                    "recoveries",
                    "contacts",
                    "wanings",
                    "reassortments",
                    "vaccine_dose_1_wanings",
                    "vaccine_dose_2_wanings",
                    "vaccine_dose_1_count",
                    "vaccine_dose_2_count",
                ],
            ),
            ("immunity", files.immunity_file, ["id", "strain", "time", "age"]),
        ]

        for name, filename, headers in file_configs:
            # Always create new file handles and write headers
            if name in self.file_handles:
                # Close existing handle if it exists
                if self.file_handles[name] and not self.file_handles[name].closed:
                    self.file_handles[name].close()

            self.file_handles[name] = open(filename, "w", newline="", buffering=8192)
            writer = csv.writer(self.file_handles[name])
            writer.writerow(headers)
            self.file_handles[name].flush()  # Ensure headers are written immediately
            self.csv_buffers[name] = []
            self.csv_write_counters[name] = 0

    def _flush_csv_buffer(self, name, force=False):
        """Flush CSV buffer to file when threshold is reached."""
        if name not in self.csv_buffers or not self.csv_buffers[name]:
            return

        if force or len(self.csv_buffers[name]) >= self.pars.csv_buffer_size:
            writer = csv.writer(self.file_handles[name])
            writer.writerows(self.csv_buffers[name])
            self.file_handles[name].flush()
            self.csv_buffers[name].clear()

    def _write_to_csv_buffer(self, name, row):
        """Add row to CSV buffer and flush if needed."""
        if name not in self.csv_buffers:
            return

        self.csv_buffers[name].append(row)
        self.csv_write_counters[name] += 1

        # Flush periodically based on interval
        if self.csv_write_counters[name] % self.pars.csv_flush_interval == 0:
            self._flush_csv_buffer(name)

    def _close_all_files(self):
        """Close all file handles and flush remaining buffers."""
        if not hasattr(self, "file_handles"):
            return

        # Flush all remaining buffers
        for name in self.csv_buffers:
            self._flush_csv_buffer(name, force=True)

        # Close all file handles
        for handle in self.file_handles.values():
            if handle and not handle.closed:
                handle.close()

        self.file_handles.clear()
        self.csv_buffers.clear()
        self.csv_write_counters.clear()

    def init_post(self):
        super().init_post()

        try:
            self.vx = self.sim.interventions.rotavaxprog
        except AttributeError:
            print("Rota module requires the 'rotavaxprog' intervention to be defined in the simulation. Creating default intervention.")
            from .interventions import RotaVaxProg
            self.sim.interventions.rotavaxprog = RotavaxProg()


        # Reset the seed
        rnd.seed(self.sim.pars.rand_seed)
        np.random.seed(self.sim.pars.rand_seed)

        if self.pars.tau != self.sim.pars.dt:
            raise ValueError(
                f"Warning: tau != sim.dt: {self.pars.tau} != {self.sim.pars.dt}"
            )

        self.reassortmentRate_GP = self.pars.reassortment_rate

        # Final initialization
        self.immunity_counts = 0
        self.reassortment_count = 0
        self.tau_steps = 0
        self.rota_results = sc.objdict(
            columns=[
                "id",
                "Strain",
                "CollectionTime",
                "Age",
                "Severity",
                "InfectionTime",
                "PopulationSize",
            ],
            infected_all=[],
        )

        # Unpack the tuple into the corresponding variables
        self.homotypic_immunity_rate = self.pars.homotypic_immunity_rate
        self.partial_heterotypic_immunity_rate = (
            self.pars.partial_heterotypic_immunity_rate
        )
        self.complete_heterotypic_immunity_rate = (
            self.pars.complete_heterotypic_immunity_rate
        )

        # self.total_strain_counts_vaccine = defaultdict(int)

        rnd.shuffle(self.pars.segment_combinations)
        number_all_strains = len(self.pars.segment_combinations)
        n_init_seg = self.pars.num_initial_infected_per_strain

        if self.pars.initial_diversity_setting == "high":
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
        elif self.pars.initial_diversity_setting == "low":
            # initial strains for the Low baseline diversity setting
            initial_segment_combinations = {
                (1, 8, 1, 1): n_init_seg,
                (3, 8, 1, 1): n_init_seg,
                (2, 4, 1, 1): n_init_seg,
                (4, 8, 1, 1): n_init_seg,
            }
        else:
            raise ValueError(
                "Invalid initial_diversity_setting: %s"
                % self.pars.initial_diversity_setting
            )

        # Track the number of immune hosts(immunity_counts) in the host population
        infected_uids = []
        pathogens_uids = []

        # for each strain track the number of hosts infected with it at current time: strain_count
        self.strain_count = {}

        # Store these for later
        self.infected_uids = infected_uids
        self.pathogens_uids = pathogens_uids

        for i in range(number_all_strains):
            self.strain_count[self.pars.segment_combinations[i]] = 0

        # if initial immunity is true
        if self.sim.pars.verbose > 0:
            if self.pars.initial_immunity:
                print("Initial immunity is set to True")
            else:
                print("Initial immunity is set to False")

        ### infecting the initial infecteds
        for initial_strain, num_infected in initial_segment_combinations.items():
            if self.pars.initial_immunity:
                for j in range(
                    int(self.sim.pars.n_agents * self.pars.initial_immunity_rate)
                ):
                    h_uid = rnd.choice(self.sim.people.uid)
                    self.immunity[h_uid][initial_strain] = self.t.abstvec[self.ti]
                    self.immunity_counts += 1
                    self.is_immune_flag[h_uid] = True

            # This does NOT guarantee num_infected will be the starting number of infected hosts. Need a random choice
            # without replacement for that.
            for j in range(num_infected):
                h = int(rnd.choice(self.sim.people.uid))
                if not self.isInfected(h):
                    infected_uids.append(h)
                p = RotaPathogen(
                    rotasim=self,
                    is_reassortant=False,
                    creation_time=self.t.abstvec[self.ti],
                    host_uid=h,
                    strain=initial_strain,
                )
                pathogens_uids.append(p)
                self.infecting_pathogen[h].append(p)
                self.strain_count[p.strain] += 1
        if self.sim.pars.verbose > 0:
            print(self.strain_count)

        if self.pars.to_csv:
            self.initialize_files()

        self.tau_steps = 0
        self.last_data_collected = 0
        self.data_collection_rate = 0.1

    # Initialize all the output files
    def initialize_files(self):
        if self.sim.pars.verbose > 0:
            print("Initializing files")
        sc.makefilepath("./results/", makedirs=True)  # Ensure results folder exists

        # Use the new buffered file setup
        self._setup_csv_writers()

        if self.sim.pars.verbose > 0:
            print("Files initialized")

    def start_step(self):
        self.rel_sus[:] = 1.0  # reset relative susceptibility

    def step(self):
        """
        Perform the actual integration loop
        """
        if self.tau_steps % 10 == 0:
            if self.sim.pars.verbose > 0:
                print(
                    f"Year: {self.t.abstvec[self.ti]}; step: {self.tau_steps}; hosts: {len(self.sim.people)}; elapsed: {self.T.total} s"
                )
            if self.sim.pars.verbose > 0:
                print(self.strain_count)

        ### Every 100 steps, write the age distribution of the population to a file
        if self.tau_steps % 100 == 0:
            age_dict = {}

            binned_ages = np.digitize(self.sim.people.age, age_bins)
            bin_counts = np.bincount(binned_ages, minlength=len(age_bins) + 1)
            for i in np.arange(len(age_labels)):
                age_dict[age_labels[i]] = bin_counts[i]

            if self.sim.pars.verbose > 0:
                print("Ages: ", age_dict)
            if self.pars.to_csv:
                self._write_to_csv_buffer(
                    "age", [self.t.abstvec[self.ti]] + list(age_dict.values())
                )

        co_infected_hosts = []
        for i in self.infected_uids:
            if len(self.infecting_pathogen[i]) >= 2:
                co_infected_hosts.append(i)

        # Calculate the number of events in a single tau step
        events = self.get_event_counts(
            len(self.sim.people),
            len(self.infected_uids),
            len(co_infected_hosts),
            self.immunity_counts,
            self.pars.tau,
            self.reassortmentRate_GP,
        )

        # Unpack the event counts
        (
            births,
            deaths,
            recoveries,
            contacts,
            wanings,
            reassortments,
        ) = events

        if len(self.infected_uids)  < recoveries:
            if self.sim.verbose:
                print("[Warning]: more recoveries than infected hosts after event counts. Setting recoveries to infected count.")
            recoveries = len(self.infected_uids)

        # Log the event counts if verbosity is enabled
        if self.sim.pars.verbose > 0:
            print(
                f"t={self.t.abstvec[self.ti]:.2f}, "
                f"births={births}, deaths={deaths}, recoveries={recoveries}, "
                f"contacts={contacts}, wanings={wanings}, reassortments={reassortments}, "
            )

        # Parse into dict
        self.event_dict[:] += events

        # perform the events for the obtained counts
        self.birth_events(births)
        self.reassortment_event(
            co_infected_hosts, reassortments
        )  # calling the function
        self.contact_event(contacts, self.infected_uids)
        self.death_event(deaths, self.infected_uids)
        self.recovery_event(recoveries, self.infected_uids)
        self.waning_event(wanings)

        if self.pars.to_csv:
            f = self.files
            if self.t.abstvec[self.ti] >= self.last_data_collected:
                self.collect_and_write_data(f.sample_output_filename, sample=True)
                self.collect_and_write_data(
                    f.infected_all_output_filename, sample=False
                )
                self.last_data_collected += self.data_collection_rate


            vx_results = self.vx.results

            self._write_to_csv_buffer(
                "event_counts",
                [
                    self.t.abstvec[self.ti],
                    births,
                    deaths,
                    recoveries,
                    contacts,
                    wanings,
                    reassortments,
                    vx_results.new_vaccinated_first_dose[self.ti],
                    vx_results.new_vaccinated_second_dose[self.ti],
                ],
            )

            self._write_to_csv_buffer(
                "output",
                [self.t.abstvec[self.ti]]
                + list(self.strain_count.values())
                + [self.reassortment_count],
            )

        self.tau_steps += 1

        if self.sim.pars.verbose > 0:
            self.T.toc()
        return

    def get_strain_name(self):
        G, P, A, B = [str(self.strain[i]) for i in range(4)]
        return f"G{G}P{P}A{A}B{B}"

    def get_probability_of_severe(
        self, immunity_count, pathogen_in=None, vaccine=None, n_doses=0,
    ):  # TEMP: refactor and include above
        if immunity_count >= 3:
            severity_probability = 0.18
        elif immunity_count == 2:
            severity_probability = 0.24
        elif immunity_count == 1:
            severity_probability = 0.23
        elif immunity_count == 0:
            severity_probability = 0.17

        if vaccine is not None and n_doses > 0:
            # Probability of severity also depends on the strain (homotypic/heterotypic/etc.)
            pathogen_match = vaccine.product.is_match(pathogen_in)
            # Effectiveness of the vaccination depends on the number of doses
            ve_s = vaccine.product.vaccine_efficacy_s[n_doses][pathogen_match]
            return severity_probability * (1 - ve_s)
        else:
            return severity_probability

    ############# tau-Function to calculate event counts ############################
    def get_event_counts(
        self,
        N,
        infected_count,
        co_infected_count,
        R,
        tau,
        RR_GP,
    ):
        births = np.random.poisson(size=1, lam=tau * N * self.pars.birth_rate)[0]
        deaths = np.random.poisson(size=1, lam=tau * N * self.pars.mu)[0]
        recoveries = np.random.poisson(
            size=1, lam=tau * self.pars.gamma * infected_count
        )[0]
        contacts = np.random.poisson(
            size=1, lam=tau * self.pars.contact_rate * infected_count
        )[0]
        wanings = np.random.poisson(size=1, lam=tau * self.pars.omega * R)[0]
        reassortments = np.random.poisson(size=1, lam=tau * RR_GP * co_infected_count)[
            0
        ]

        return (
            births,
            deaths,
            recoveries,
            contacts,
            wanings,
            reassortments,
        )


    def isInfected(self, uid):
        return len(self.infecting_pathogen[uid]) != 0

    def get_weights_by_age(self):
        weights = self.sim.people.age.values
        total_w = np.sum(weights)
        weights = weights / total_w
        return weights

    def birth_events(self, birth_count):
        new_uids = self.sim.people.grow(birth_count)  # add more people!
        self.sim.people.age[new_uids] = 0

    def death_event(self, num_deaths, infected_uids):
        # host_list = np.arange(len(self.host_pop))
        p = self.get_weights_by_age()
        dying_uids = np.random.choice(
            self.sim.people.alive.uids, p=p, size=num_deaths, replace=False
        )
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

    def recovery_event(self, num_recovered, infected_uids):
        weights = np.array(
            [self.get_oldest_current_infection(x) for x in infected_uids]
        )
        # If there is no one with an infection older than 0 return without recovery
        if sum(weights) == 0:
            return
        # weights_e = np.exp(weights)
        total_w = np.sum(weights)
        weights = weights / total_w

        recovering_hosts_uids = np.random.choice(
            infected_uids, p=weights, size=num_recovered, replace=False
        )
        not_immune = recovering_hosts_uids[
            self.is_immune_flag[ss.uids(recovering_hosts_uids)] == False
        ]
        self.immunity_counts += len(not_immune)

        self.recover(recovering_hosts_uids)
        for recovering_host_uid in recovering_hosts_uids:
            infected_uids.remove(recovering_host_uid)

    def contact_event(self, contacts, infected_uids):
        if len(infected_uids) == 0 and self.sim.verbose:
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

        rel_beta = self.pars.rel_beta
        for h1_uid, h2_uid, rnd_num in zip(h1_uids, h2_uids, rnd_nums):

            # If the contact is the same as the infected host, pick another host at random
            while h1_uid == h2_uid:
                h2_uid = rnd.choice(self.sim.people.alive.uids)

            infecting_probability = infecting_probability_map.get(
                self.prior_infections[h2_uid], 0
            )

            infecting_probability *= rel_beta # Scale by this calibration parameter

            if rnd_num > infecting_probability:
                continue


            h2_previously_infected = self.isInfected(uid=h2_uid)

            h1_pathogens = self.infecting_pathogen[h1_uid]
            transmit_all = False
            if len(h1_pathogens) > 1:
                # small chance to transmit all pathogens
                transmit_all = rnd.random() < 0.02
            else:
                h1_pathogens.sort(
                    key=lambda path: (path.get_fitness(), rnd.random()), reverse=True
                )

            for h1_pathogen in h1_pathogens:
                if self.can_variant_infect_host(h2_uid, h1_pathogen):
                    infected = self.infect_with_pathogen(
                        h2_uid, self.infecting_pathogen[h1_uid][0]
                        )

                    if infected and not transmit_all:
                        break


            # in this case h2 was not infected before but is infected now
            if not h2_previously_infected and self.isInfected(h2_uid):
                infected_uids.append(h2_uid)

        return # counter

    def reassortment_event(self, coinfectedhosts, reassortment_count):
        reassortment_hosts = np.random.choice(
            coinfectedhosts, min(len(coinfectedhosts), reassortment_count)
        )

        for reassortment_host in reassortment_hosts:
            parentalstrains = [
                path.strain for path in self.infecting_pathogen[reassortment_host]
            ]
            possible_reassortants = [
                path
                for path in self.compute_combinations(reassortment_host)
                if path.strain not in parentalstrains
            ]
            for path in possible_reassortants:
                self.infect_with_reassortant(reassortment_host, path)

    def waning_event(self, wanings):
        # Get all the hosts in the population that has an immunity
        h_immune_uids = (self.is_immune_flag).uids
        oldest_infections = self.oldest_infection[h_immune_uids]
        order = np.argsort(oldest_infections, stable=True)

        # For the selected hosts set the immunity to be None
        for i in order[:wanings]:
            h_uid = h_immune_uids[i]
            self.immunity[h_uid] = {}
            self.is_immune_flag[h_uid] = False
            self.oldest_infection[h_uid] = np.nan
            self.prior_infections[h_uid] = 0
            self.immunity_counts -= 1


    @staticmethod
    def get_strain_antigenic_name(strain):
        return "G" + str(strain[0]) + "P" + str(strain[1])




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
        parantal_strains = [
            j.strain[: self.pars.numAgSegments] for j in self.infecting_pathogen[uid]
        ]

        # Itertools product returns all possible combinations
        # We are only interested in strain combinations that are reassortants of the parental strains
        # We need to skip all existing combinations from the parents
        # Ex: (1, 1, 2, 2) and (2, 2, 1, 1) should not create (1, 1, 1, 1) as a possible reassortant if only the antigenic parts reassort

        # below block is for reassorting antigenic segments only
        all_antigenic_combinations = [
            i for i in itertools.product(*seg_combinations) if i not in parantal_strains
        ]
        all_nonantigenic_combinations = [
            j.strain[self.pars.numAgSegments :] for j in self.infecting_pathogen[uid]
        ]
        all_strains = set(
            [
                (i[0] + i[1])
                for i in itertools.product(
                    all_antigenic_combinations, all_nonantigenic_combinations
                )
            ]
        )
        all_pathogens = [
            RotaPathogen(
                rotasim=self,
                is_reassortant=True,
                creation_time=self.t.abstvec[self.ti],
                host_uid=uid,
                strain=tuple(i),
            )
            for i in all_strains
        ]

        return all_pathogens

    def get_oldest_current_infection(self, uid):
        max_infection_times = max(
            [
                self.t.abstvec[self.ti] - p.creation_time
                for p in self.infecting_pathogen[uid]
            ]
        )
        return max_infection_times

    def recover(self, uids):
        # We will use the pathogen creation time to count the number of infections
        for uid in uids:
            creation_times = set()
            for path in self.infecting_pathogen[uid]:
                strain = path.strain
                if not path.is_reassortant:
                    self.strain_count[strain] -= 1
                    creation_times.add(path.creation_time)
                    self.immunity[uid][strain] = self.t.abstvec[self.ti]
                    self.is_immune_flag[uid] = True
                    if np.isnan(self.oldest_infection[uid]):
                        self.oldest_infection[uid] = self.t.abstvec[self.ti]
            self.prior_infections[uid] += len(creation_times)
            self.infecting_pathogen[uid] = []
            self.possibleCombinations[uid] = []


    def is_vaccine_immune(self, uid, infecting_strain):
        vx_intv = self.vx

        n_doses = vx_intv.n_doses[uid]

        if n_doses == 0:
            return False

        strain_match = vx_intv.product.is_match(infecting_strain)


        # get the vaccine efficacy for the given strain match and number of doses and scale it by waned effectiveness
        return rnd.random() < (vx_intv.product.vaccine_efficacy_i[n_doses][strain_match] * vx_intv.waned_effectiveness[uid])


    def collect_and_write_data(self, output_filename, sample=False, sample_size=1000):
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
            buffer_name = "sample"
            population_to_collect = np.random.choice(
                self.sim.people.alive.uids, sample_size, replace=False
            )
        else:
            buffer_name = "infected_all"
            population_to_collect = self.sim.people.alive.uids

        collected_data = []
        immunity_data = []
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
                    for vs in [
                        self.get_strain_antigenic_name(s)
                        for s in self.vaccine[uid].strain
                    ]:
                        collected_vaccination_data.append(
                            [
                                uid,
                                vs,
                                self.t.abstvec[self.ti],
                                self.get_age_category(uid),
                                self.vaccine[uid].time,
                            ]
                        )
            if len(self.prior_vaccinations[uid]) != 0:
                if len(vaccinated_uids) < 1000:
                    vaccinated_uids.append(uid)
            else:
                if len(unvaccinated_uids) < 1000:
                    unvaccinated_uids.append(uid)
            if self.isInfected(uid):
                strain_str = [
                    (path.get_strain_name(), path.is_severe, path.creation_time)
                    for path in self.infecting_pathogen[uid]
                    if not sample or not path.is_reassortant
                ]
                for strain in strain_str:
                    collected_data.append(
                        [
                            uid,
                            strain[0],
                            self.t.abstvec[self.ti],
                            self.get_age_category(uid),
                            strain[1],
                            strain[2],
                            len(self.sim.people.alive.uids),
                        ]
                    )
            for immune_strain in self.immunity[uid].keys():
                immunity_data.append(
                    [
                        uid,
                        get_strain_name(immune_strain),
                        self.t.abstvec[self.ti],
                        self.get_age_category(uid),
                    ]
                )

        # Use buffered writing
        for row in collected_data:
            self._write_to_csv_buffer(buffer_name, row)

        if not sample:
            self.rota_results.infected_all.extend(collected_data)
            for row in collected_vaccination_data:
                self._write_to_csv_buffer("vaccinations", row)
            for row in immunity_data:
                self._write_to_csv_buffer("immunity", row)

        # Only collect the vaccine efficacy data if we have vaccinated the hosts
        if self.vaccine_campaign_started:
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
                for infecting_pathogen in self.infections_with_vaccination[
                    vaccinated_host
                ]:
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
                for infecting_pathogen in self.infections_without_vaccination[
                    unvaccinated_host
                ]:
                    if infecting_pathogen.is_severe:
                        was_there_a_severe_infection = True
                        break
                if was_there_a_severe_infection:
                    num_unvaccinated_infected_severe += 1

            self._write_to_csv_buffer(
                "vaccine_efficacy",
                [
                    self.t.abstvec[self.ti],
                    self.t.abstvec[self.ti] - self.pars.vaccination_time,
                    num_vaccinated,
                    num_unvaccinated,
                    num_vaccinated_infected,
                    num_vaccinated_infected_severe,
                    num_unvaccinated_infected,
                    num_unvaccinated_infected_severe,
                    num_homotypic[0],
                    num_homotypic[1],
                    num_partial_heterotypic[0],
                    num_partial_heterotypic[1],
                    num_full_heterotypic[0],
                    num_full_heterotypic[1],
                ],
            )

    def get_age_category(self, uid):
        # Bin the age into categories
        for i in range(len(age_bins)):
            if self.sim.people.age[uid] < age_bins[i]:
                return age_labels[i]
        return age_labels[-1]

    def can_variant_infect_host(self, uid, infecting_pathogen):
        current_infections = self.infecting_pathogen[uid]
        numAgSegments = self.pars.numAgSegments
        partial_heterotypic_immunity_rate = self.partial_heterotypic_immunity_rate
        complete_heterotypic_immunity_rate = self.complete_heterotypic_immunity_rate
        homotypic_immunity_rate = self.homotypic_immunity_rate
        infecting_strain = infecting_pathogen.strain

        # If the host is vaccinated, draw for vaccine immunity first
        if self.is_vaccine_immune(
            uid, infecting_strain
        ):
            return False

        # If the host is infected with the same strain, cannot reinfect with exact same strain
        current_infecting_strains = (
            i.strain[:numAgSegments] for i in current_infections
        )
        if infecting_strain[:numAgSegments] in current_infecting_strains:
            return False



        def is_complete_antigenic_match():
            immune_strains = (s[:numAgSegments] for s in self.immunity[uid].keys())
            return infecting_strain[:numAgSegments] in immune_strains

        def has_shared_antigenic_genotype():
            for i in range(numAgSegments):
                immune_genotypes = (strain[i] for strain in self.immunity[uid].keys())
                if infecting_strain[i] in immune_genotypes:
                    return True
            return False

        if is_complete_antigenic_match():
            return rnd.random() > homotypic_immunity_rate

        if has_shared_antigenic_genotype():
            return rnd.random() > partial_heterotypic_immunity_rate

        # If the strain is complete heterotypic
        return rnd.random() > complete_heterotypic_immunity_rate


    def infect_with_pathogen(self, uid, pathogen_in):
        """This function returns a fitness value to a strain based on the hypothesis"""
        fitness = pathogen_in.get_fitness()

        # e.g. fitness = 0.8 (there's a 80% chance the virus infecting a host)
        if rnd.random() > fitness:
            return False

        # Probability of getting a severe disease depends on the number of previous infections and vaccination status of the host
        vx = self.vx
        severity_probability = self.get_probability_of_severe(
           self.prior_infections[uid], pathogen_in.strain, vx, vx.n_doses[uid]
        )
        if rnd.random() < severity_probability:
            severe = True
        else:
            severe = False

        if pathogen_in.is_reassortant:
            self.reassortment_count += 1

        new_p = RotaPathogen(
            rotasim=self,
            is_reassortant=False,
            creation_time=self.t.abstvec[self.ti],
            host_uid=uid,
            strain=pathogen_in.strain,
            is_severe=severe,
        )
        self.infecting_pathogen[uid].append(new_p)
        self.record_infection(new_p)

        self.strain_count[new_p.strain] += 1

        return True

    def infect_with_reassortant(self, uid, reassortant_virus):
        self.infecting_pathogen[uid].append(reassortant_virus)

    def record_infection(self, new_p):
        uid = new_p.host_uid
        if len(self.prior_vaccinations[uid]) != 0:
            vaccine_strain = self.prior_vaccinations[uid][-1]
            self.infections_with_vaccination[uid].append(
                (new_p, new_p.match(vaccine_strain))
            )
        else:
            self.infections_without_vaccination[uid].append(new_p)

    def to_df(self):
        """Convert results to a dataframe"""
        cols = self.rota_results.columns
        res = self.rota_results.infected_all
        df = sc.dataframe(data=res, columns=cols)
        self.df = df
        return df

    def finalize(self):
        """Finalize the module and ensure proper cleanup."""
        # Close all file handles and flush remaining buffers
        self._close_all_files()

        self.df = self.to_df()
        super().finalize()

    def __del__(self):
        """Destructor to ensure files are closed when object is deleted."""
        self._close_all_files()

    def __getstate__(self):
        """Prepare object for pickling by removing unpickleable attributes."""
        # Flush all buffers before pickling
        if hasattr(self, "csv_buffers") and self.pars.to_csv:
            for name in self.csv_buffers:
                self._flush_csv_buffer(name, force=True)

        state = self.__dict__.copy()
        # Remove the unpickleable CSV writers and file handles
        state["file_handles"] = {}
        state["csv_buffers"] = {}
        state["csv_write_counters"] = {}
        return state

    def __setstate__(self, state):
        """Restore object after unpickling by recreating file handles."""
        self.__dict__.update(state)
        # Initialize empty containers if they don't exist
        if not hasattr(self, "file_handles"):
            self.file_handles = sc.objdict()
        if not hasattr(self, "csv_buffers"):
            self.csv_buffers = sc.objdict()
        if not hasattr(self, "csv_write_counters"):
            self.csv_write_counters = sc.objdict()

        # Recreate the writers if CSV output is enabled
        if self.pars.to_csv:
            self._setup_csv_writers_append_mode()

    def _setup_csv_writers_append_mode(self):
        """Initialize CSV writers in append mode after unpickling."""
        if not self.pars.to_csv:
            return

        files = self.files

        # Setup buffered file handles in append mode (no headers needed)
        file_configs = [
            ("output", files.output_filename),
            ("sample", files.sample_output_filename),
            ("infected_all", files.infected_all_output_filename),
            ("vaccinations", files.vaccinations_output_filename),
            ("vaccine_efficacy", files.vaccine_efficacy_output_filename),
            ("age", files.age_output_filename),
            ("event_counts", files.event_counts_filename),
            ("immunity", files.immunity_file),
        ]

        for name, filename in file_configs:
            if name not in self.file_handles:
                # Open in append mode since headers already exist
                self.file_handles[name] = open(
                    filename, "a", newline="", buffering=8192
                )
                self.csv_buffers[name] = []
                self.csv_write_counters[name] = 0

    def __str__(self):
        return (
            "Strain: "
            + self.get_strain_name()
            + " Severe: "
            + str(self.is_severe)
            + " Host: "
            + str(self.host.id)
            + str(self.creation_time)
        )
