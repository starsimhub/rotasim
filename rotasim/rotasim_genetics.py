import sciris as sc
import starsim as ss


class Host(sc.prettyobj):
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
        self.prior_infections = 0
        self.prior_vaccinations = []
        self.infections_with_vaccination = []
        self.infections_without_vaccination = []
        self.is_immune_flag = False
        self.oldest_infection = np.nan
        return

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
            strain = path.strain
            if not path.is_reassortant:
                strain_counts[strain] -= 1
                creation_times.add(path.creation_time)
                self.immunity[strain] = self.t
                self.is_immune_flag = True
                if np.isnan(self.oldest_infection):
                    self.oldest_infection = self.t
        self.prior_infections += len(creation_times)
        self.infecting_pathogen = []
        self.possibleCombinations = []

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
            raise NotImplementedError(f"Unsupported vaccine dose: {self.vaccine[2]}")

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

        current_infecting_strains = (i.strain[:numAgSegments] for i in current_infections)
        if infecting_strain[:numAgSegments] in current_infecting_strains:
            return False

        def is_completely_immune():
            immune_strains = (s[:numAgSegments] for s in self.immunity.keys())
            return infecting_strain[:numAgSegments] in immune_strains

        def has_shared_genotype():
            for i in range(numAgSegments):
                immune_genotypes = (strain[i] for strain in self.immunity.keys())
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
        severity_probability = self.sim.get_probability_of_severe(self.sim, pathogen_in, self.vaccine, self.prior_infections)
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


class RotaPathogen(ss.Disease):
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
            14: {'default': 0.05, (1, 8): 0.98, (2, 4): 0.4, (3, 8): 0.7, (4, 8): 0.6, (9, 8): 0.7, (12, 8): 0.75, (9, 6): 0.58, (11, 8): 0.2},
            15: {'default': 0.4, (1, 8): 1, (2, 4): 0.7, (3, 8): 0.93, (4, 8): 0.93, (9, 8): 0.95, (12, 8): 0.94, (9, 6): 0.3, (11, 8): 0.35},
            16: {'default': 0.4, (1, 8): 1, (2, 4): 0.7, (3, 8): 0.85, (4, 8): 0.88, (9, 8): 0.95, (12, 8): 0.93, (9, 6): 0.85, (12, 6): 0.90, (9, 4): 0.90, (1, 6): 0.6, (2, 8): 0.6, (2, 6): 0.6},
            17: {'default': 0.7, (1, 8): 1, (2, 4): 0.85, (3, 8): 0.85, (4, 8): 0.88, (9, 8): 0.95, (12, 8): 0.93, (9, 6): 0.83, (12, 6): 0.90, (9, 4): 0.90, (1, 6): 0.8, (2, 8): 0.8, (2, 6): 0.8},
            18: {'default': 0.65, (1, 8): 1, (2, 4): 0.92, (3, 8): 0.79, (4, 8): 0.81, (9, 8): 0.95, (12, 8): 0.89, (9, 6): 0.80, (12, 6): 0.86, (9, 4): 0.83, (1, 6): 0.75, (2, 8): 0.75, (2, 6): 0.75},
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
        fitness_hypothesis = self.sim.fitness_hypothesis
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
                default=0.5
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

    # def get_fitness(self):
    #     """ Get the fitness based on the fitness hypothesis and the two strains """
    #     fitness_hypothesis = self.sim.fitness_hypothesis
    #     strain = self.strain

    #     if fitness_hypothesis in self.fitness_map:
    #         fitness_values = self.fitness_map[fitness_hypothesis]
    #         return fitness_values.get((strain[0], strain[1]), fitness_values.get('default', 1))
    #     else:
    #         raise NotImplementedError(f"Invalid fitness_hypothesis: {fitness_hypothesis}")

    def get_strain_name(self):
        G,P,A,B = [str(self.strain[i]) for i in range(4)]
        return f'G{G}P{P}A{A}B{B}'

    def __str__(self):
        return "Strain: " + self.get_strain_name() + " Severe: " + str(self.is_severe) + " Host: " + str(self.host.id) + str(self.creation_time)