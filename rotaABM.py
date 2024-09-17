"""
RotaABM class

Usage:
    import rotaABM as rabm
    rota = rabm.RotaABM()
    rota.run()
    
TODO:
    - Replace host with array
    - Replace pathogen with array
    - Replace random with numpy
    - Replace math with numpy
    - Refactor Pathogen.get_fitness()
"""

import sys
import csv
import math
import random as rnd
import numpy as np
import itertools
import sciris as sc

# Define age bins and labels
age_bins = [2/12, 4/12, 6/12, 12/12, 24/12, 36/12, 48/12, 60/12, 100]
age_distribution = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.84]                  # needs to be changed to fit the site-specific population
age_labels = ['0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60+']


### Host class
class Host:
    """
    A rotavirus host
    """
    def __init__(self, host_id, sim):
        self.sim = sim
        self.id = host_id
        self.bday = self.t - self.get_random_age()
        self.immunity = {} # set of strains the host is immune to
        self.vaccine = None
        self.infecting_pathogen = []
        self.priorInfections = 0
        self.prior_vaccinations = []
        self.infections_with_vaccination = []
        self.infections_without_vaccination = []
    
    @property
    def t(self):
        return self.sim.t
    
    @property
    def numAgSegments(self):
        return self.sim.numAgSegments

    @staticmethod
    def get_random_age():
        # pick a age bin
        random_age_bin = np.random.choice(list(range(len(age_bins))), p=age_distribution)
        # generate a random age in the bin
        if random_age_bin > 0:
            min_age = age_bins[random_age_bin-1]
        else:
            min_age = 0
        max_age = age_bins[random_age_bin]
        return rnd.uniform(min_age, max_age)

    def get_age_category(self):
        # Bin the age into categories
        for i in range(len(age_bins)):
            if self.t - self.bday < age_bins[i]:
                return age_labels[i]
        return age_labels[-1]
    
    def get_oldest_current_infection(self):
        max_infection_times = max([self.t - p.creation_time for p in self.infecting_pathogen])
        return max_infection_times

    def get_oldest_infection(self):
        max_infection_times = max([self.t - p[1] for p in self.immunity.items()])
        return max_infection_times
    
    def compute_combinations(self):
        seg_combinations = []

        # We want to only reassort the GP types
        # Assumes that antigenic segments are at the start
        for i in range(self.numAgSegments):
            availableVariants = set([])                 
            for j in self.infecting_pathogen:
                availableVariants.add((j.strain[i]))
            seg_combinations.append(availableVariants)

        # compute the parental strains 
        parantal_strains = [j.strain[:self.numAgSegments] for j in self.infecting_pathogen]

        # Itertools product returns all possible combinations
        # We are only interested in strain combinations that are reassortants of the parental strains
        # We need to skip all existing combinations from the parents
        # Ex: (1, 1, 2, 2) and (2, 2, 1, 1) should not create (1, 1, 1, 1) as a possible reassortant if only the antigenic parts reassort
        
        # below block is for reassorting antigenic segments only
        all_antigenic_combinations = [i for i in itertools.product(*seg_combinations) if i not in parantal_strains]
        all_nonantigenic_combinations = [j.strain[self.numAgSegments:] for j in self.infecting_pathogen]
        all_strains = set([(i[0] + i[1]) for i in itertools.product(all_antigenic_combinations, all_nonantigenic_combinations)])
        all_pathogens = [Pathogen(self.sim, True, self.t, host = self, strain=tuple(i)) for i in all_strains]

        return all_pathogens

    def isInfected(self):
        return len(self.infecting_pathogen) != 0

    def recover(self,strain_counts):
        # We will use the pathogen creation time to count the number of infections
        creation_times = set()
        for path in self.infecting_pathogen:
            if not path.is_reassortant:
                strain_counts[path.strain] -= 1
                creation_times.add(path.creation_time)
                self.immunity[path.strain] = self.t
        self.priorInfections += len(creation_times)
        self.infecting_pathogen = []                  
        self.possibleCombinations = []
    
    def is_immune(self):
        return len(self.immunity) != 0
    
    def vaccinate(self, vaccinated_strain):
        if len(self.prior_vaccinations) == 0:
            self.prior_vaccinations.append(vaccinated_strain)
            self.vaccine = ([vaccinated_strain], self.t, 1)
        else:
            self.prior_vaccinations.append(vaccinated_strain)
            self.vaccine = ([vaccinated_strain], self.t, 2)
    
    def is_vaccine_immune(self, infecting_strain):
        # Effectiveness of the vaccination depends on the number of doses
        if self.vaccine[2] == 1:
            ve_i_rates = self.sim.vaccine_efficacy_i_d1
        elif self.vaccine[2] == 2:
            ve_i_rates = self.sim.vaccine_efficacy_i_d2
        else:
            print("Unsupported vaccine dose")
            exit(-1)

        # Vaccine strain only contains the antigenic parts
        vaccine_strain = self.vaccine[0]
        vaccine_hypothesis = self.sim.vaccine_hypothesis
        
        if vaccine_hypothesis == 0:
            return False
        if vaccine_hypothesis == 1:            
            if infecting_strain[:self.numAgSegments] in vaccine_strain:
                if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                    return True
                else:
                    return False
        elif vaccine_hypothesis == 2:
            if infecting_strain[:self.numAgSegments] in vaccine_strain:
                if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                    return True
                else:
                    return False
            strains_match = False
            for i in range(self.numAgSegments):         
                immune_genotypes = [strain[i] for strain in vaccine_strain]
                if infecting_strain[i] in immune_genotypes:
                    strains_match = True
            if strains_match:
                if rnd.random() < ve_i_rates[PathogenMatch.PARTIAL_HETERO]:
                    return True
            else:
                return False
        # used below hypothesis for the analysis in the report
        elif vaccine_hypothesis == 3:           
            if infecting_strain[:self.numAgSegments] in vaccine_strain:
                if rnd.random() < ve_i_rates[PathogenMatch.HOMOTYPIC]:
                    return True
                else:
                    return False
            strains_match = False
            for i in range(self.numAgSegments):         
                immune_genotypes = [strain[i] for strain in vaccine_strain]
                if infecting_strain[i] in immune_genotypes:
                    strains_match = True
            if strains_match:
                if rnd.random() < ve_i_rates[PathogenMatch.PARTIAL_HETERO]:
                    return True
            else:
                if rnd.random() < ve_i_rates[PathogenMatch.COMPLETE_HETERO]:
                    return True
                else:
                    return False
        else:
            raise NotImplementedError("Unsupported vaccine hypothesis")
    
    def can_variant_infect_host(self, infecting_strain, current_infections):
        numAgSegments = self.numAgSegments
        immunity_hypothesis = self.sim.immunity_hypothesis
        partial_cross_immunity_rate = self.sim.partial_cross_immunity_rate
        complete_heterotypic_immunity_rate = self.sim.complete_heterotypic_immunity_rate
        homotypic_immunity_rate = self.sim.homotypic_immunity_rate
    
        if self.vaccine is not None and self.is_vaccine_immune(infecting_strain):
            return False
    
        current_infecting_strains = [i.strain[:numAgSegments] for i in current_infections]
        if infecting_strain[:numAgSegments] in current_infecting_strains:
            return False
    
        def is_completely_immune():
            immune_strains = [s[:numAgSegments] for s in self.immunity.keys()]
            return infecting_strain[:numAgSegments] in immune_strains
    
        def has_shared_genotype():
            for i in range(numAgSegments):
                immune_genotypes = [strain[i] for strain in self.immunity.keys()]
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
            immune_ptypes = [strain[1] for strain in self.immunity.keys()]
            return infecting_strain[1] not in immune_ptypes
    
        elif immunity_hypothesis == 6:
            immune_ptypes = [strain[0] for strain in self.immunity.keys()]
            return infecting_strain[0] not in immune_ptypes
    
        else:
            print("[Error] Immunity hypothesis not implemented")
            exit(-1)

    def record_infection(self, new_p):        
        if len(self.prior_vaccinations) != 0:
            vaccine_strain = self.prior_vaccinations[-1]            
            self.infections_with_vaccination.append((new_p, new_p.match(vaccine_strain)))
        else:
            self.infections_without_vaccination.append(new_p)
            
    def infect_with_pathogen(self, pathogen_in, strain_counts):
        """ This function returns a fitness value to a strain based on the hypothesis """ 
        fitness = pathogen_in.get_fitness()
        
        # e.g. fitness = 0.8 (there's a 80% chance the virus infecting a host)
        if rnd.random() > fitness:                
            return False
        
        # Probability of getting a severe decease depends on the number of previous infections and vaccination status of the host   
        severity_probability = self.sim.get_probability_of_severe(self.sim, pathogen_in, self.vaccine, self.priorInfections)
        if rnd.random() < severity_probability:
            severe = True
        else:
            severe = False

        new_p = Pathogen(self.sim, False, self.t, host = self, strain=pathogen_in.strain, is_severe=severe)
        self.infecting_pathogen.append(new_p)
        self.record_infection(new_p)

        strain_counts[new_p.strain] += 1

        return True

    def infect_with_reassortant(self, reassortant_virus):
        self.infecting_pathogen.append(reassortant_virus)



### Pathogen classes

class PathogenMatch:
    """ Define whether pathogens are completely heterotypic, partially heterotypic, or homotypic """
    COMPLETE_HETERO = 1
    PARTIAL_HETERO = 2
    HOMOTYPIC = 3


class Pathogen(object): 
    """
    Pathogen dynamics
    """
    def __init__(self, sim, is_reassortant, creation_time, is_severe=False, host=None, strain=None): 
        self.sim = sim
        self.host = host
        self.creation_time = creation_time
        self.is_reassortant = is_reassortant
        self.strain = strain
        self.is_severe = is_severe
        
        self.fitness_map = {
            (1, 'default'): 1,
            (2, 1, 1): 0.93, (2, 2, 2): 0.93, (2, 3, 3): 0.93, (2, 4, 4): 0.93, (2, 'default'): 0.90,
            (3, 1, 1): 0.93, (3, 2, 2): 0.93, (3, 3, 3): 0.90, (3, 4, 4): 0.90, (3, 'default'): 0.87,
            (4, 1, 1): 1, (4, 2, 2): 0.2, (4, 'default'): 1,
            (5, 1, 1): 1, (5, 2, 1): 0.5, (5, 1, 3): 0.5, (5, 'default'): 0.2,
            (6, 1, 8): 1, (6, 2, 4): 0.2, (6, 3, 8): 0.4, (6, 4, 8): 0.5, (6, 'default'): 0.05,
            (7, 1, 8): 1, (7, 2, 4): 0.3, (7, 3, 8): 0.7, (7, 4, 8): 0.6, (7, 'default'): 0.05,
            (8, 1, 8): 1, (8, 2, 4): 0.4, (8, 3, 8): 0.9, (8, 4, 8): 0.8, (8, 'default'): 0.05,
            (9, 1, 8): 1, (9, 2, 4): 0.5, (9, 3, 8): 0.9, (9, 4, 8): 0.8, (9, 'default'): 0.2,
            (10, 1, 8): 1, (10, 2, 4): 0.6, (10, 3, 8): 0.9, (10, 4, 8): 0.9, (10, 'default'): 0.4,
            (11, 1, 8): 0.98, (11, 2, 4): 0.7, (11, 3, 8): 0.8, (11, 4, 8): 0.8, (11, 'default'): 0.5,
            (12, 1, 8): 0.98, (12, 2, 4): 0.8, (12, 3, 8): 0.9, (12, 4, 8): 0.9, (12, 'default'): 0.5,
            (13, 1, 8): 0.98, (13, 2, 4): 0.8, (13, 3, 8): 0.9, (13, 4, 8): 0.9, (13, 'default'): 0.7,
            (14, 1, 8): 0.98, (14, 2, 4): 0.4, (14, 3, 8): 0.7, (14, 4, 8): 0.6, (14, 9, 8): 0.7, (14, 12, 8): 0.75, (14, 9, 6): 0.58, (14, 11, 8): 0.2, (14, 'default'): 0.05,
            (15, 1, 8): 1, (15, 2, 4): 0.7, (15, 3, 8): 0.93, (15, 4, 8): 0.93, (15, 9, 8): 0.95, (15, 12, 8): 0.94, (15, 9, 6): 0.3, (15, 11, 8): 0.35, (15, 'default'): 0.4,
            (16, 1, 8): 1, (16, 2, 4): 0.7, (16, 3, 8): 0.85, (16, 4, 8): 0.88, (16, 9, 8): 0.95, (16, 12, 8): 0.93, (16, 9, 6): 0.85, (16, 12, 6): 0.90, (16, 9, 4): 0.90, (16, 1, 6): 0.6, (16, 2, 8): 0.6, (16, 2, 6): 0.6, (16, 'default'): 0.4,
            (17, 1, 8): 1, (17, 2, 4): 0.85, (17, 3, 8): 0.85, (17, 4, 8): 0.88, (17, 9, 8): 0.95, (17, 12, 8): 0.93, (17, 9, 6): 0.83, (17, 12, 6): 0.90, (17, 9, 4): 0.90, (17, 1, 6): 0.8, (17, 2, 8): 0.8, (17, 2, 6): 0.8, (17, 'default'): 0.7,
            (18, 1, 8): 1, (18, 2, 4): 0.92, (18, 3, 8): 0.79, (18, 4, 8): 0.81, (18, 9, 8): 0.95, (18, 12, 8): 0.89, (18, 9, 6): 0.80, (18, 12, 6): 0.86, (18, 9, 4): 0.83, (18, 1, 6): 0.75, (18, 2, 8): 0.75, (18, 2, 6): 0.75, (18, 'default'): 0.65,
            (19, 1, 8): 1, (19, 2, 4): 0.5, (19, 3, 8): 0.55, (19, 4, 8): 0.55, (19, 9, 8): 0.6, (19, 'default'): 0.4,
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
        fitness_hypothesis = self.sim.fitness_hypothesis
        strain = self.strain
        fitness_map = self.fitness_map
        key = (fitness_hypothesis, strain[0], strain[1])
        try:
            return fitness_map.get(key, fitness_map.get((fitness_hypothesis, 'default'), 1))
        except:
            raise NotImplementedError(f"Invalid fitness_hypothesis: {fitness_hypothesis}")
        
    def get_strain_name(self):
        G,P,A,B = [str(self.strain[i]) for i in range(4)]
        return f'G{G}P{P}A{A}B{B}'
    
    def __str__(self): 
        return "Strain: " + self.get_strain_name() + " Severe: " + str(self.is_severe) + " Host: " + str(self.host.id) + str(self.creation_time)


### RotaABM class
class RotaABM:
    """
    Run the simulation
    """
    
    def __init__(self, 
            N = 100_000,
            timelimit = 40,
            verbose = None,
            **kwargs,
        ):
        """
        Create the simulation.
        
        Args:
            defaults (list): a list of parameters matching the command-line inputs; see below
            verbose (bool): the "verbosity" of the output: if False, print nothing; if None, print the timestep; if True, print out results
        """
        # Define the default parameters
        args = sc.objdict(
            immunity_hypothesis = 1,
            reassortment_rate = 0.1,
            fitness_hypothesis = 1,
            vaccine_hypothesis = 1,
            waning_hypothesis = 1,
            initial_immunity = 0,
            ve_i_to_ve_s_ratio = 0.5,
            experiment_number = 1,
        )
        
        # Update with any keyword arguments
        for k,v in kwargs.items():
            if k in args:
                args[k] = v
            else:
                KeyError(k)
        
        # Loop over command line input arguments, if provided
        for i,arg in enumerate(sys.argv[1:]):
            args[i] = arg
        
        if verbose is not False:
            print(f'Creating simulation with N={N}, timelimit={timelimit} and parameters:')
            print(args)
        
        # Store parameters directly in the sim
        self.immunity_hypothesis = int(args[0])
        self.reassortment_rate = float(args[1])
        self.fitness_hypothesis = int(args[2])
        self.vaccine_hypothesis = int(args[3])
        self.waning_hypothesis = int(args[4])
        self.initial_immunity = int(args[5]) # 0 = no immunity
        self.ve_i_to_ve_s_ratio = float(args[6])
        self.experiment_number = int(args[7])
        self.verbose = verbose
        
        # Reset the seed
        rnd.seed(self.experiment_number)
        np.random.seed(self.experiment_number)
        
        # Set filenames
        name_suffix =  '%r_%r_%r_%r_%r_%r_%r_%r' % (self.immunity_hypothesis, self.reassortment_rate, self.fitness_hypothesis, self.vaccine_hypothesis, self.waning_hypothesis, self.initial_immunity, self.ve_i_to_ve_s_ratio, self.experiment_number)
        self.files = sc.objdict()
        self.files.outputfilename = './results/rota_strain_count_%s.csv' % (name_suffix)
        self.files.vaccinations_outputfilename = './results/rota_vaccinecount_%s.csv' % (name_suffix)
        self.files.sample_outputfilename = './results/rota_strains_sampled_%s.csv' % (name_suffix)
        self.files.infected_all_outputfilename = './results/rota_strains_infected_all_%s.csv' % (name_suffix)
        self.files.age_outputfilename = './results/rota_agecount_%s.csv' % (name_suffix)
        self.files.vaccine_efficacy_output_filename = './results/rota_vaccine_efficacy_%s.csv' % (name_suffix)
        self.files.sample_vaccine_efficacy_output_filename = './results/rota_sample_vaccine_efficacy_%s.csv' % (name_suffix)

        # Set other parameters
        self.N = N  # initial population size
        self.timelimit = timelimit  # simulation years
        self.mu = 1.0/70.0     # average life span is 70 years
        self.gamma = 365/7  # 1/average infectious period (1/gamma =7 days)
        if self.waning_hypothesis == 1:
            omega = 365/273  # duration of immunity by infection= 39 weeks
        elif self.waning_hypothesis == 2:
            omega = 365/50  
        elif self.waning_hypothesis == 3:
            omega = 365/100  
        self.omega = omega
        self.birth_rate = self.mu * 4
        
        self.contact_rate = 365/1    
        self.reassortmentRate_GP = self.reassortment_rate
        
        self.vaccination_time =  20
    
        # Efficacy of the vaccine first dose
        self.vaccine_efficacy_d1 = {
            PathogenMatch.HOMOTYPIC: 0.6,
            PathogenMatch.PARTIAL_HETERO: 0.45,
            PathogenMatch.COMPLETE_HETERO:0.15,
        }
        # Efficacy of the vaccine second dose
        self.vaccine_efficacy_d2 = {
            PathogenMatch.HOMOTYPIC: 0.8,
            PathogenMatch.PARTIAL_HETERO: 0.65,
            PathogenMatch.COMPLETE_HETERO:0.35,
        }
        
        self.vaccination_single_dose_waning_rate = 365/273 #365/1273
        self.vaccination_double_dose_waning_rate = 365/546 #365/2600
        # vaccination_waning_lower_bound = 20 * 7 / 365.0
        
        # Tau leap parametes
        self.tau = 1/365.0
        
        # if initialization starts with a proportion of immune agents:
        self.num_initial_immune = 10000
        
        # Final initialization
        self.immunityCounts = 0
        self.ReassortmentCount = 0
        self.pop_id = 0
        self.t = 0.0
        
        return
        
    @staticmethod
    def get_probability_of_severe(sim, pathogen_in, vaccine, immunity_count): # TEMP: refactor and include above
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
                ve_s = sim.vaccine_efficacy_s_d1[pathogen_strain_type]
            elif vaccine[2] == 2:
                ve_s = sim.vaccine_efficacy_s_d2[pathogen_strain_type]
            else:
                print("Unsupported vaccine dose")
                exit(-1)
            return severity_probability * (1-ve_s)
        else:
            return severity_probability
        
    # Initialize all the output files
    def initialize_files(self, strain_count):
        files = self.files
        with open(files.outputfilename, "w+", newline='') as outputfile:
            write = csv.writer(outputfile)
            write.writerow(["time"] + list(strain_count.keys()) + ["ReassortmentCount"])  # header for the csv file
            write.writerow([self.t] + list(strain_count.values()) + [self.ReassortmentCount])  # first row of the csv file will be the initial state
    
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
    
    ############# tau-Function to calculate event counts ############################
    def get_event_counts(self, N, I, R, tau, RR_GP, single_dose_count, double_dose_count): 
        births = np.random.poisson(size=1, lam=tau*N*self.birth_rate)[0]
        deaths = np.random.poisson(size=1, lam=tau*N*self.mu)[0]
        recoveries = np.random.poisson(size=1, lam=tau*self.gamma*I)[0]
        contacts = np.random.poisson(size=1, lam=tau*self.contact_rate*I)[0]
        wanings = np.random.poisson(size=1, lam=tau*self.omega*R)[0]
        reassortments = np.random.poisson(size=1, lam=tau*RR_GP*I)[0]
        vaccination_wanings_one_dose = np.random.poisson(size=1, lam=tau*self.vaccination_single_dose_waning_rate*single_dose_count)[0]
        vaccination_wanings_two_dose = np.random.poisson(size=1, lam=tau*self.vaccination_double_dose_waning_rate*double_dose_count)[0]
        return (births, deaths, recoveries, contacts, wanings, reassortments, vaccination_wanings_one_dose, vaccination_wanings_two_dose)
    
    @staticmethod
    def coInfected_contacts(host1, host2, strain_counts):  
        h2existing_pathogens = list(host2.infecting_pathogen)
        randomnumber = rnd.random()
        if randomnumber < 0.02:       # giving all the possible strains
            for path in host1.infecting_pathogen:
                if host2.can_variant_infect_host(path.strain, h2existing_pathogens):
                    host2.infect_with_pathogen(path, strain_counts)
        else:  # give only one strain depending on fitness
            host1paths = list(host1.infecting_pathogen)     
            # Sort by fitness first and randomize the ones with the same fitness
            host1paths.sort(key=lambda path: (path.get_fitness(), rnd.random()), reverse=True)
            for path in host1paths:
                if host2.can_variant_infect_host(path.strain, h2existing_pathogens):
                    infected = host2.infect_with_pathogen(path, strain_counts)
                    if infected:
                        break
    
    def contact_event(self, infected_pop, host_pop, strain_count):
        if len(infected_pop) == 0:
            print("[Warning] No infected hosts in a contact event. Skipping")
            return
        
        h1 = rnd.choice(infected_pop)
        h2 = rnd.choice(host_pop)
    
        while h1 == h2:
            h2 = rnd.choice(host_pop)

        # based on proir infections and current infections, the relative risk of subsequent infections
        """ number_of_current_infections = len(h2.infecting_pathogen) """
        number_of_current_infections = 0

        if h2.priorInfections + number_of_current_infections == 0:
            infecting_probability = 1
        elif h2.priorInfections + number_of_current_infections == 1:
            infecting_probability = 0.61
        elif h2.priorInfections + number_of_current_infections == 2:  
            infecting_probability = 0.48
        elif h2.priorInfections + number_of_current_infections == 3:
            infecting_probability = 0.33
        else:
            infecting_probability = 0         
        rnd_num = rnd.random()
        if rnd_num > infecting_probability:
            return 
    
        h2_previously_infected = h2.isInfected()
    
        if len(h1.infecting_pathogen)==1:
            if h2.can_variant_infect_host(h1.infecting_pathogen[0].strain, h2.infecting_pathogen):
                h2.infect_with_pathogen(h1.infecting_pathogen[0], strain_count)
        else:
            self.coInfected_contacts(h1,h2,strain_count)
        
        # in this case h2 was not infected before but is infected now
        if not h2_previously_infected and h2.isInfected():
            infected_pop.append(h2)
            
    def get_weights_by_age(self, host_pop):
        weights = np.array([self.t - x.bday for x in host_pop])
        total_w = np.sum(weights)
        weights = weights / total_w
        return weights
    
    def death_event(self, num_deaths, infected_pop, host_pop, strain_count):
        host_list = np.arange(len(host_pop))
        p = self.get_weights_by_age(host_pop)
        inds = np.random.choice(host_list, p=p, size=num_deaths, replace=False)
        dying_hosts = [host_pop[ind] for ind in inds]
        for h in dying_hosts:
            if h.isInfected():
                infected_pop.remove(h)
                for path in h.infecting_pathogen:
                    if not path.is_reassortant:
                        strain_count[path.strain] -= 1
            if h.is_immune():
                self.immunityCounts -= 1
            host_pop.remove(h)
            
    def recovery_event(self, num_recovered, infected_pop, strain_count):
        weights=np.array([x.get_oldest_current_infection() for x in infected_pop])
        # If there is no one with an infection older than 0 return without recovery
        if (sum(weights) == 0):
            return
        # weights_e = np.exp(weights)
        total_w = np.sum(weights)
        weights = weights / total_w
    
        recovering_hosts = np.random.choice(infected_pop, p=weights, size=num_recovered, replace=False)
        for host in recovering_hosts:
            if not host.is_immune():
                self.immunityCounts +=1 
            host.recover(strain_count)
            infected_pop.remove(host)
    
    @staticmethod
    def reassortment_event(infected_pop, reassortment_count):
        coinfectedhosts = []
        for i in infected_pop:
            if len(i.infecting_pathogen) >= 2:
                coinfectedhosts.append(i)
        rnd.shuffle(coinfectedhosts)
    
        for i in range(min(len(coinfectedhosts),reassortment_count)):
            parentalstrains = [path.strain for path in coinfectedhosts[i].infecting_pathogen]
            possible_reassortants = [path for path in coinfectedhosts[i].compute_combinations() if path not in parentalstrains]
            for path in possible_reassortants:
                coinfectedhosts[i].infect_with_reassortant(path)
    
    def waning_event(self, host_pop, wanings):
        # Get all the hosts in the population that has an immunity
        h_immune = [h for h in host_pop if h.is_immune()]
        age_tiebreak = lambda x: (x.get_oldest_infection(), rnd.random())
        hosts_with_immunity = sorted(h_immune, key=age_tiebreak, reverse=True)
        
        # Alternate implementation -- not faster, but left in as a placeholder 
        # immune_inds = sc.findinds([h.is_immune for h in host_pop])
        # ages = np.array([host_pop[i].get_oldest_infection() for i in immune_inds])
        # ages += np.random.rand(len(ages))*1e-12 # Add noise to break ties
        # immunity_sort_inds = np.argsort(ages)[::-1]
        # immunity_sort_inds = immunity_sort_inds[:wanings]
    
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(hosts_with_immunity), wanings)):
            h = hosts_with_immunity[i]
            h.immunity =  {}
            # h.is_immune = False
            h.priorInfections = 0
            self.immunityCounts -= 1
    
    @staticmethod
    def waning_vaccinations_first_dose(single_dose_pop, wanings):
        """ Get all the hosts in the population that has an vaccine immunity """
        rnd.shuffle(single_dose_pop)
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(single_dose_pop), wanings)):
            h = single_dose_pop[i]
            h.vaccinations =  None
    
    @staticmethod
    def waning_vaccinations_second_dose(second_dose_pop, wanings):
        rnd.shuffle(second_dose_pop)
        # For the selcted hosts set the immunity to be None
        for i in range(min(len(second_dose_pop), wanings)):
            h = second_dose_pop[i]
            h.vaccinations =  None
    
    def birth_events(self, birth_count, host_pop):
        for _ in range(birth_count):
            self.pop_id += 1
            new_host = Host(self.pop_id, sim=self)
            new_host.bday = self.t
            host_pop.append(new_host)
            if self.vaccine_hypothesis !=0 and self.done_vaccinated:
                if rnd.random() < self.vaccine_first_dose_rate:
                    self.to_be_vaccinated_pop.append(new_host)
    
    def get_strain_antigenic_name(strain):
        return "G" + str(strain[0]) + "P" + str(strain[1])
    
    @staticmethod
    def solve_quadratic(a, b, c):
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            root1 = (-b + discriminant**0.5) / (2*a)
            root2 = (-b - discriminant**0.5) / (2*a)
            return tuple(sorted([root1, root2]))
        else:
            return "No real roots"
        
    def breakdown_vaccine_efficacy(self, ve, x):
        (r1, r2) = self.solve_quadratic(x, -(1+x), ve)
        if self.verbose: print(r1, r2)
        if r1 >= 0 and r1 <= 1:
            ve_s = r1
        elif r2 >= 0 and r2 <= 1:
            ve_s = r2
        else:
            print("No valid solution to the equation: x: %d, ve: %d. Solutions: %f %f" % (x, ve, r1, r2))
            exit(-1)
        ve_i = x * ve_s
        return (ve_i, ve_s)
    
    def collect_and_write_data(self, host_population, output_filename, vaccine_output_filename, vaccine_efficacy_output_filename, sample=False, sample_size=1000):
        """
        Collects data from the host population and writes it to a CSV file.
        If sample is True, it collects data from a random sample of the population.
        
        Args:
        - host_population: List of host objects.
        - output_filename: Name of the file to write the data.
        - sample: Boolean indicating whether to collect data from a sample or the entire population.
        - sample_size: Size of the sample to collect data from if sample is True.
        """
        # Select the population to collect data from
        if sample:
            population_to_collect = np.random.choice(host_population, sample_size, replace=False)
        else:
            population_to_collect = host_population
    
        # Shuffle the population to avoid the need for random sampling
        rnd.shuffle(population_to_collect)
        
        collected_data = []
        collected_vaccination_data = []
    
        # To measure vaccine efficacy we will gather data on the number of vaccinated hosts who get infected
        # along with the number of unvaccinated hosts that get infected
        vaccinated_hosts = []
        unvaccinated_hosts = []
    
        for h in population_to_collect:
            if not sample:
                # For vaccination data file, we will count the number of agents with current vaccine immunity
                # This will exclude those who previously got the vaccine but the immunity waned.
                if h.vaccine is not None:
                    for vs in [self.get_strain_antigenic_name(s) for s in h.vaccine[0]]:
                        collected_vaccination_data.append((h.id, vs, self.t, h.get_age_category(), h.vaccine[2]))
            if len(h.prior_vaccinations) != 0:
                if len(vaccinated_hosts) < 1000:
                    vaccinated_hosts.append(h)
            else:
                if len(unvaccinated_hosts) < 1000:
                    unvaccinated_hosts.append(h)
            if h.isInfected():
                strain_str = [(path.get_strain_name(), path.is_severe, path.creation_time) for path in h.infecting_pathogen if not sample or not path.is_reassortant]
                for strain in strain_str:
                    collected_data.append((h.id, strain[0], self.t, h.get_age_category(), strain[1], strain[2], len(self.host_pop)))
                    
        # Only collect the vaccine efficacy data if we have vaccinated the hosts
        if self.done_vaccinated:
            num_vaccinated = len(vaccinated_hosts)
            num_unvaccinated = len(unvaccinated_hosts)
            num_vaccinated_infected = 0
            num_unvaccinated_infected = 0           
            num_vaccinated_infected_severe = 0
            num_unvaccinated_infected_severe = 0
            num_full_heterotypic = [0, 0]
            num_partial_heterotypic = [0, 0]
            num_homotypic = [0, 0]
    
            for vaccinated_host in vaccinated_hosts:
                if len(vaccinated_host.infections_with_vaccination) > 0:
                    num_vaccinated_infected += 1
                was_there_a_severe_infection = False
                was_there_a_full_heterotypic_infection = [False, False]
                was_there_a_partial_heterotypic_infection = [False, False]
                was_there_a_homotypic_infection = [False, False]
                for infecting_pathogen in vaccinated_host.infections_with_vaccination:     
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
    
            for unvaccinated_host in unvaccinated_hosts:
                if len(unvaccinated_host.infections_without_vaccination) > 0:
                    num_unvaccinated_infected += 1
                was_there_a_severe_infection = False
                for infecting_pathogen in unvaccinated_host.infections_without_vaccination:                
                    if infecting_pathogen.is_severe:
                        was_there_a_severe_infection = True
                        break
                if was_there_a_severe_infection:
                    num_unvaccinated_infected_severe += 1
    
            with open(vaccine_efficacy_output_filename, "a", newline='') as outputfile:
                write = csv.writer(outputfile)
                write.writerow([self.t, num_vaccinated, num_unvaccinated, num_vaccinated_infected, num_vaccinated_infected_severe, num_unvaccinated_infected, num_unvaccinated_infected_severe,
                                num_homotypic[0], num_homotypic[1], num_partial_heterotypic[0], num_partial_heterotypic[1], num_full_heterotypic[0], num_full_heterotypic[1]]) 
    
        # Write collected data to the output file
        with open(output_filename, "a", newline='') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerows(collected_data)
        if not sample:
            with open(vaccine_output_filename, "a", newline='') as outputfile:
                writer = csv.writer(outputfile)
                writer.writerows(collected_vaccination_data)
    
    def run(self):
        """
        Run the simulation
        """
        self.prepare_run()
        events = self.integrate()
        return events
        
    def prepare_run(self):
        """
        Set up the variables for the run
        """
        # relative protection for infection from natural immunity 
        immunity_hypothesis = self.immunity_hypothesis
        homotypic_immunity_rate = 0 # TEMP, not defined in all if statements
        if immunity_hypothesis == 1 or immunity_hypothesis == 4 or immunity_hypothesis == 5:
            partial_cross_immunity_rate = 0
            complete_heterotypic_immunity_rate = 0
        elif immunity_hypothesis == 2 :
            partial_cross_immunity_rate = 1
            complete_heterotypic_immunity_rate = 0
        elif immunity_hypothesis == 3:
            partial_cross_immunity_rate = 0.5
            complete_heterotypic_immunity_rate = 0
        elif immunity_hypothesis == 4:
            partial_cross_immunity_rate = 0.95
            complete_heterotypic_immunity_rate = 0.9
        elif immunity_hypothesis == 5:
            partial_cross_immunity_rate = 0.95
            complete_heterotypic_immunity_rate = 0.90
        elif immunity_hypothesis == 7:
            homotypic_immunity_rate = 0.95
            partial_cross_immunity_rate = 0.90
            complete_heterotypic_immunity_rate = 0.2
        elif immunity_hypothesis == 8:
            homotypic_immunity_rate = 0.9
            partial_cross_immunity_rate = 0.5
            complete_heterotypic_immunity_rate = 0
        elif immunity_hypothesis == 9:
            homotypic_immunity_rate = 0.9
            partial_cross_immunity_rate = 0.45
            complete_heterotypic_immunity_rate = 0.35
        # below combination is what I used for the analysis in the report
        elif immunity_hypothesis == 10:
            homotypic_immunity_rate = 0.8
            partial_cross_immunity_rate = 0.45
            complete_heterotypic_immunity_rate = 0.35
        else:
            print("No partial cross immunity rate for immunity hypothesis: ", immunity_hypothesis)
            exit(-1) 
            
        self.homotypic_immunity_rate = homotypic_immunity_rate
        self.partial_cross_immunity_rate = partial_cross_immunity_rate
        self.complete_heterotypic_immunity_rate = complete_heterotypic_immunity_rate
        
        self.done_vaccinated = False
        
        vaccine_efficacy_i_d1 = {}
        vaccine_efficacy_s_d1 = {}
        vaccine_efficacy_i_d2 = {}
        vaccine_efficacy_s_d2 = {}
        for (k, v) in self.vaccine_efficacy_d1.items():
            (ve_i, ve_s) = self.breakdown_vaccine_efficacy(v, self.ve_i_to_ve_s_ratio)
            vaccine_efficacy_i_d1[k] = ve_i
            vaccine_efficacy_s_d1[k] = ve_s
        for (k, v) in self.vaccine_efficacy_d2.items():
            (ve_i, ve_s) = self.breakdown_vaccine_efficacy(v, self.ve_i_to_ve_s_ratio)
            vaccine_efficacy_i_d2[k] = ve_i
            vaccine_efficacy_s_d2[k] = ve_s
        
        if self.verbose: print("VE_i: ", vaccine_efficacy_i_d1)
        if self.verbose: print("VE_s: ", vaccine_efficacy_s_d1)
        
        # Vaccination rates are derived based on the following formula
        vaccine_second_dose_rate = 0.8
        vaccine_first_dose_rate = math.sqrt(vaccine_second_dose_rate)
        if self.verbose: print("Vaccination - first dose rate: %s, second dose rate %s" % (vaccine_first_dose_rate, vaccine_second_dose_rate))
        
        self.total_strain_counts_vaccine = {}
        
        numSegments = 4
        numNoneAgSegments = 2
        self.numAgSegments = numSegments - numNoneAgSegments
        #segmentVariants = [[i for i in range(1, 3)], [i for i in range(1, 3)], [i for i in range(1, 2)], [i for i in range(1, 2)]]     ## creating variats for the segments
        segmentVariants = [[1,2,3,4,9,11,12], [8,4,6], [i for i in range(1, 2)], [i for i in range(1, 2)]]
        # segmentVariants for the Low baseline diversity setting
        #segmentVariants = [[1,2,3,4,9], [8,4], [i for i in range(1, 2)], [i for i in range(1, 2)]]
        segment_combinations = [tuple(i) for i in itertools.product(*segmentVariants)]  # getting all possible combinations from a list of list
        rnd.shuffle(segment_combinations)
        number_all_strains = len(segment_combinations)
        n_init_seg = 100
        initial_segment_combinations = {
            (1,8,1,1) : n_init_seg,
            (2,4,1,1) : n_init_seg,
            (9,8,1,1) : n_init_seg,
            (4,8,1,1) : n_init_seg,
            (3,8,1,1) : n_init_seg,
            (12,8,1,1): n_init_seg,
            (12,6,1,1): n_init_seg,
            (9,4,1,1) : n_init_seg,
            (9,6,1,1) : n_init_seg,
            (1,6,1,1) : n_init_seg,
            (2,8,1,1) : n_init_seg,
            (2,6,1,1) : n_init_seg,
            (11,8,1,1): n_init_seg,
            (11,6,1,1): n_init_seg,
            (1,4,1,1) : n_init_seg,
            (12,4,1,1): n_init_seg,
        }
        # initial strains for the Low baseline diversity setting
        #initial_segment_combinations = {(1,8,1,1): 100, (2,4,1,1): 100} #, (9,8,1,1): 100} #, (4,8,1,1): 100} 
    
        # Track the number of immune hosts(immunityCounts) in the host population
        infected_pop = []
        pathogens_pop = []
        
        # for each strain track the number of hosts infected with it at current time: strain_count  
        strain_count = {}   
        
        # for each number in range of N, make a new Host object, i is the id.
        host_pop = [Host(i, self) for i in range(self.N)]   
        
        self.pop_id = self.N
        self.to_be_vaccinated_pop = [] 
        self.single_dose_vaccinated_pop = []
        
        # Store these for later
        self.infected_pop = infected_pop
        self.pathogens_pop = pathogens_pop
        self.host_pop = host_pop
        self.strain_count = strain_count
        
        for i in range(number_all_strains):
            self.strain_count[segment_combinations[i]] = 0
    
        # if initial immunity is true 
        if self.verbose:
            if self.initial_immunity:
                print("Initial immunity is set to True")
            else:
                print("Initial immunity is set to False")
    
        ### infecting the initial infecteds
        for (initial_strain, num_infected) in initial_segment_combinations.items():
            if self.initial_immunity:
                for j in range(self.num_initial_immune):
                    h = rnd.choice(host_pop)
                    h.immunity[initial_strain] = self.t
                    self.immunityCounts += 1
            
            for j in range(num_infected):                     
                h = rnd.choice(host_pop)
                if not h.isInfected():
                    infected_pop.append(h) 
                p = Pathogen(self, False, self.t, host = h, strain = initial_strain)
                pathogens_pop.append(p)
                h.infecting_pathogen.append(p)
                strain_count[p.strain] += 1                       
        if self.verbose: print(strain_count)
        
        self.initialize_files(strain_count)   
        
        self.tau_steps = 0
        self.T = sc.timer() # for us to track the time it takes to run the simulation
        self.last_data_colllected = 0
        self.data_collection_rate = 0.1
        
        for strain, count in strain_count.items():
            if strain[:self.numAgSegments] in self.total_strain_counts_vaccine:
                self.total_strain_counts_vaccine[strain[:self.numAgSegments]] += count
            else:
                self.total_strain_counts_vaccine[strain[:self.numAgSegments]] = count
        return
                
    def integrate(self):
        """
        Perform the actual integration loop
        """
        host_pop = self.host_pop
        strain_count = self.strain_count
        infected_pop = self.infected_pop
        single_dose_vaccinated_pop = self.single_dose_vaccinated_pop
        to_be_vaccinated_pop = self.to_be_vaccinated_pop
        total_strain_counts_vaccine = self.total_strain_counts_vaccine
        
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
        while self.t<self.timelimit:
            if self.tau_steps % 10 == 0:
                if self.verbose is not False: print("Current time: %f (Number of steps = %d)" % (self.t, self.tau_steps))
                if self.verbose: print(self.strain_count)
        
            ### Every 100 steps, write the age distribution of the population to a file
            if self.tau_steps % 100 == 0:
                age_dict = {}
                for age_range in age_labels:
                    age_dict[age_range] = 0
                for h in self.host_pop:
                    age_dict[h.get_age_category()] += 1
                if self.verbose: print("Ages: ", age_dict)
                with open(self.files.age_outputfilename, "a", newline='') as outputfile:
                    write = csv.writer(outputfile)
                    write.writerow(["{:.2}".format(self.t)] + list(age_dict.values()))
            
            # Count the number of hosts with 1 or 2 vaccinations
            single_dose_hosts = []
            double_dose_hosts = []
            for h in host_pop:
                if h.vaccine is not None:
                    if h.vaccine[2] == 1:
                        single_dose_hosts.append(h)
                    elif h.vaccine[2] == 2:
                        double_dose_hosts.append(h)
        
            # Get the number of events in a single tau step
            events = self.get_event_counts(len(host_pop), len(infected_pop), self.immunityCounts, self.tau, self.reassortmentRate_GP, len(single_dose_hosts), len(double_dose_hosts))
            births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings, vaccine_dose_2_wanings = events
            if self.verbose: print("t={}, births={}, deaths={}, recoveries={}, contacts={}, wanings={}, reassortments={}, waning_vaccine_d1={}, waning_vaccine_d2={}".format(self.t, births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings, vaccine_dose_2_wanings))
        
            # Parse into dict
            self.event_dict[:] += events
            
            # perform the events for the obtained counts
            self.birth_events(births, host_pop)
            self.reassortment_event(infected_pop, reassortments) # calling the function
            for _ in range(contacts):
                self.contact_event(infected_pop, host_pop, strain_count)
            self.death_event(deaths, infected_pop, host_pop, strain_count)
            self.recovery_event(recoveries, infected_pop, strain_count)    
            self.waning_event(host_pop, wanings)
            self.waning_vaccinations_first_dose(single_dose_hosts, vaccine_dose_1_wanings)
            self.waning_vaccinations_second_dose(double_dose_hosts, vaccine_dose_2_wanings)
            
            # Collect the total counts of strains at each time step to determine the most prevalent strain for vaccination
            if not self.done_vaccinated:
                for strain, count in strain_count.items():
                    total_strain_counts_vaccine[strain[:self.numAgSegments]] += count
            
            # Administer the first dose of the vaccine
            # Vaccination strain is the most prevalent strain in the population before the vaccination starts
            if self.vaccine_hypothesis!=0 and (not self.done_vaccinated) and self.t >= self.vaccination_time:
                # Sort the strains by the number of hosts infected with it in the past
                # Pick the last one from the sorted list as the most prevalent strain
                vaccinated_strain = sorted(list(total_strain_counts_vaccine.keys()), key=lambda x: total_strain_counts_vaccine[x])[-1]
                # Select hosts under 6.5 weeks and over 4.55 weeks of age for vaccinate
                child_host_pop = [h for h in host_pop if self.t - h.bday <= 0.13 and self.t - h.bday >= 0.09]
                # Use the vaccination rate to determine the number of hosts to vaccinate
                vaccination_count = int(len(child_host_pop)*self.vaccine_first_dose_rate)            
                sample_population = rnd.sample(child_host_pop, vaccination_count)
                if self.verbose: print("Vaccinating with strain: ", vaccinated_strain, vaccination_count)
                if self.verbose: print("Number of people vaccinated: {} Number of people under 6 weeks: {}".format(len(sample_population), len(child_host_pop)))
                for h in sample_population:
                    h.vaccinate(vaccinated_strain)
                    single_dose_vaccinated_pop.append(h)
                self.done_vaccinated = True
            elif self.done_vaccinated:
                for child in to_be_vaccinated_pop:
                    if self.t - child.bday >= 0.11:
                        child.vaccinate(vaccinated_strain)
                        to_be_vaccinated_pop.remove(child)
                        single_dose_vaccinated_pop.append(child)
        
            # Administer the second dose of the vaccine if first dose has already been administered.
            # The second dose is administered 6 weeks after the first dose with probability vaccine_second_dose_rate
            if self.done_vaccinated:
                while len(single_dose_vaccinated_pop) > 0:
                    # If the first dose of the vaccine is older than 6 weeks then administer the second dose
                    if self.t - single_dose_vaccinated_pop[0].vaccine[1] >= 0.11:
                        child = single_dose_vaccinated_pop.pop(0)
                        if rnd.random() < self.vaccine_second_dose_rate:
                            child.vaccinate(vaccinated_strain)
                    else:
                        break
        
            f = self.files
            if self.t >= self.last_data_colllected:
                self.collect_and_write_data(host_pop, f.sample_outputfilename, f.vaccinations_outputfilename, f.sample_vaccine_efficacy_output_filename, sample=True)
                self.collect_and_write_data(host_pop, f.infected_all_outputfilename, f.vaccinations_outputfilename, f.vaccine_efficacy_output_filename, sample=False)
                self.last_data_colllected += self.data_collection_rate
                
            with open(f.outputfilename, "a", newline='') as outputfile:
                write = csv.writer(outputfile)
                write.writerow([self.t] + list(strain_count.values()) + [self.ReassortmentCount])
        
            self.tau_steps += 1
            self.t += self.tau
        
        if self.verbose is not False:
            self.T.toc()
            print(self.event_dict)
        return self.event_dict


if __name__ == '__main__':
    rota = RotaABM(N=2000, timelimit=10)
    events = rota.run()