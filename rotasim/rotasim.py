"""
Rotasim model

Usage:
    import rotasim as rs
    sim = rs.Sim()
    sim.run()

TODO:
    - Figure out how to make host vaccination more efficient
    - Replace host with array
    - Replace pathogen with array
    - Replace random with numpy
    - Replace math with numpy
"""

import sys
import csv
import math
import random as rnd
import numpy as np
import itertools
import sciris as sc
import starsim as ss
from . import rotasim_genetics as rg



__all__ = ['Sim']


### Sim class
class Sim(ss.Sim):
    """
    Run the simulation
    """

    def __init__(self,
            n_agents = 100_000,
            timelimit = 40,
            start = 2000,
            verbose = 0,
            to_csv = True,
            rand_seed = 1,
            **kwargs,
        ):
        """
        Create the simulation.

        Args:
            defaults (list): a list of parameters matching the command-line inputs; see below
            verbose (bool): the "verbosity" of the output: if False, print nothing; if None, print the timestep; if True, print out results
        """

        super().__init__(n_agents=n_agents, start=start, stop=start+timelimit, unit='year', dt=1/365, verbose=verbose, rand_seed=rand_seed, **kwargs)



        # Update with any keyword arguments
        # for k,v in kwargs.items():
        #     if k in args:
        #         args[k] = v
        #     else:
        #         KeyError(k)

        # Loop over command line input arguments, if provided
        # Using sys.argv in this way breaks when using pytest because it passes two args instead of one (runner and script)
        # for i,arg in enumerate(sys.argv[1:]):
        #     args[i] = arg

        if verbose:
            print(f'Creating simulation with N={n_agents}, timelimit={timelimit} and parameters:')
            # print(args)

        # # Store parameters directly in the sim
        # self.immunity_hypothesis = int(args[0])
        # self.reassortment_rate = float(args[1])
        # self.fitness_hypothesis = int(args[2])
        # self.vaccine_hypothesis = int(args[3])
        # self.waning_hypothesis = int(args[4])
        # self.initial_immunity = int(args[5]) # 0 = no immunity
        # self.ve_i_to_ve_s_ratio = float(args[6])
        # self.experiment_number = int(args[7])
        # self.rel_beta = float(args[8])
        # self.verbose = verbose

        # Reset the seed
        # rnd.seed(self.experiment_number)
        # np.random.seed(self.experiment_number)

        # Set filenames
        # name_suffix =  '%r_%r_%r_%r_%r_%r_%r_%r' % (self.immunity_hypothesis, self.reassortment_rate, self.fitness_hypothesis, self.vaccine_hypothesis, self.waning_hypothesis, self.initial_immunity, self.ve_i_to_ve_s_ratio, self.experiment_number)
        # self.files = sc.objdict()
        # self.files.outputfilename = './results/rota_strain_count_%s.csv' % (name_suffix)
        # self.files.vaccinations_outputfilename = './results/rota_vaccinecount_%s.csv' % (name_suffix)
        # self.files.sample_outputfilename = './results/rota_strains_sampled_%s.csv' % (name_suffix)
        # self.files.infected_all_outputfilename = './results/rota_strains_infected_all_%s.csv' % (name_suffix)
        # self.files.age_outputfilename = './results/rota_agecount_%s.csv' % (name_suffix)
        # self.files.vaccine_efficacy_output_filename = './results/rota_vaccine_efficacy_%s.csv' % (name_suffix)
        # self.files.sample_vaccine_efficacy_output_filename = './results/rota_sample_vaccine_efficacy_%s.csv' % (name_suffix)
        #
        # # Set other parameters
        # self.to_csv = to_csv # whether to write files
        # # self.n_agents = n_agents  # initial population size
        # self.timelimit = timelimit  # simulation years
        # self.mu = 1.0/70.0     # average life span is 70 years
        # self.gamma = 365/7  # 1/average infectious period (1/gamma =7 days)
        # if self.waning_hypothesis == 1:
        #     omega = 365/273  # duration of immunity by infection= 39 weeks
        # elif self.waning_hypothesis == 2:
        #     omega = 365/50
        # elif self.waning_hypothesis == 3:
        #     omega = 365/100
        # self.omega = omega
        # self.birth_rate = self.mu * 2 # Adjust birth rate to be more in line with Bangladesh
        #
        # self.contact_rate = 365/1
        # self.reassortmentRate_GP = self.reassortment_rate
        #
        # self.vaccination_time =  20
        #
        # # Efficacy of the vaccine first dose
        # self.vaccine_efficacy_d1 = {
        #     rg.PathogenMatch.HOMOTYPIC: 0.6,
        #     rg.PathogenMatch.PARTIAL_HETERO: 0.45,
        #     rg.PathogenMatch.COMPLETE_HETERO:0.15,
        # }
        # # Efficacy of the vaccine second dose
        # self.vaccine_efficacy_d2 = {
        #     rg.PathogenMatch.HOMOTYPIC: 0.8,
        #     rg.PathogenMatch.PARTIAL_HETERO: 0.65,
        #     rg.PathogenMatch.COMPLETE_HETERO:0.35,
        # }
        #
        # self.vaccination_single_dose_waning_rate = 365/273 #365/1273
        # self.vaccination_double_dose_waning_rate = 365/546 #365/2600
        # # vaccination_waning_lower_bound = 20 * 7 / 365.0
        #
        # # Tau leap parametes
        # self.tau = 1/365.0
        #
        # # if initialization starts with a proportion of immune agents:
        # self.num_initial_immune = 10000
        #
        # # Final initialization
        # self.immunity_counts = 0
        # self.reassortment_count = 0
        # self.pop_id = 0
        # self.t = 0.0
        # self.rota_results = sc.objdict(
        #     columns = ["id", "Strain", "CollectionTime", "Age", "Severity", "InfectionTime", "PopulationSize"],
        #     infected_all = [],
        # )

        return



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
            population_to_collect = np.random.choice(self.host_pop, sample_size, replace=False)
        else:
            population_to_collect = self.host_pop

        # Shuffle the population to avoid the need for random sampling
        # rnd.shuffle(population_to_collect) # CK: not needed?

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

            if self.to_csv:
                with open(vaccine_efficacy_output_filename, "a", newline='') as outputfile:
                    write = csv.writer(outputfile)
                    write.writerow([self.t, num_vaccinated, num_unvaccinated, num_vaccinated_infected, num_vaccinated_infected_severe, num_unvaccinated_infected, num_unvaccinated_infected_severe,
                                    num_homotypic[0], num_homotypic[1], num_partial_heterotypic[0], num_partial_heterotypic[1], num_full_heterotypic[0], num_full_heterotypic[1]])

        # Write collected data to the output file
        if self.to_csv:
            with open(output_filename, "a", newline='') as outputfile:
                writer = csv.writer(outputfile)
                writer.writerows(collected_data)
        if not sample:
            self.results.infected_all.extend(collected_data)
            if self.to_csv:
                with open(vaccine_output_filename, "a", newline='') as outputfile:
                    writer = csv.writer(outputfile)
                    writer.writerows(collected_vaccination_data)

    def init(self, force=False):
        """
        Set up the variables for the run
        """
        if force or not self.initialized:

            if self.pars.people is None:
                self.pars.people = ss.People(n_agents=self.pars.n_agents)

            super().init(force=force)





        return self

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

        self.T = sc.timer() # To track the time it takes to run the simulation
        while self.t<self.timelimit:
            if self.tau_steps % 10 == 0:
                if self.verbose is not False: print(f"Year: {self.t:n}; step: {self.tau_steps}; hosts: {len(host_pop)}; elapsed: {self.T.total:n} s")
                if self.verbose: print(self.strain_count)

            ### Every 100 steps, write the age distribution of the population to a file
            if self.tau_steps % 100 == 0:
                age_dict = {}
                for age_range in age_labels:
                    age_dict[age_range] = 0
                for h in host_pop:
                    age_dict[h.get_age_category()] += 1
                if self.verbose: print("Ages: ", age_dict)
                if self.to_csv:
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
            events = self.get_event_counts(len(host_pop), len(infected_pop), self.immunity_counts, self.tau, self.reassortmentRate_GP, len(single_dose_hosts), len(double_dose_hosts))
            births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings, vaccine_dose_2_wanings = events
            if self.verbose: print("t={}, births={}, deaths={}, recoveries={}, contacts={}, wanings={}, reassortments={}, waning_vaccine_d1={}, waning_vaccine_d2={}".format(self.t, births, deaths, recoveries, contacts, wanings, reassortments, vaccine_dose_1_wanings, vaccine_dose_2_wanings))

            # Parse into dict
            self.event_dict[:] += events

            # perform the events for the obtained counts
            self.birth_events(births)
            self.reassortment_event(infected_pop, reassortments) # calling the function
            counter = self.contact_event(contacts, infected_pop, strain_count)
            self.death_event(deaths, infected_pop, strain_count)
            self.recovery_event(recoveries, infected_pop, strain_count)
            self.waning_event(wanings)
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
                self.collect_and_write_data(f.sample_outputfilename, f.vaccinations_outputfilename, f.sample_vaccine_efficacy_output_filename, sample=True)
                self.collect_and_write_data(f.infected_all_outputfilename, f.vaccinations_outputfilename, f.vaccine_efficacy_output_filename, sample=False)
                self.last_data_colllected += self.data_collection_rate

            if self.to_csv:
                with open(f.outputfilename, "a", newline='') as outputfile:
                    write = csv.writer(outputfile)
                    write.writerow([self.t] + list(strain_count.values()) + [self.reassortment_count])

            self.tau_steps += 1
            self.t += self.tau

        if self.verbose is not False:
            self.T.toc()
            print(self.event_dict)
        return self.event_dict

    def to_df(self):
        """ Convert results to a dataframe """
        cols = self.results.columns
        res = self.results.infected_all
        df = sc.dataframe(data=res, columns=cols)
        self.df = df
        return df



if __name__ == '__main__':
    sim = Sim(N=10_000, timelimit=2)
    events = sim.run()

