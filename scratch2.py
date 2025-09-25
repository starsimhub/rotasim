import sciris as sc
import rotasim as rs

N = 2_000
timelimit = 10
verbose = False



sc.heading("Testing default parameters")

# Check old baseline

with sc.timer() as T:
    interventions = [rs.RotaVaxProg(product=rs.RotaVax(), start_year=2005),]
    rota = rs.Sim(N=N, timelimit=timelimit, verbose=verbose, interventions=interventions)
    sc.profile(rota.run, [rs.Rota.contact_event, rs.Rota.can_variant_infect_host, rs.Rota.is_vaccine_immune])
    # rota.run()
    events = rota.connectors["rota"].event_dict
print(f"Defaults matched:\n{events}")

#
# def can_variant_infect_host(self, uid, infecting_strain):
#     current_infections = self.infecting_pathogen[uid]
#     numAgSegments = self.pars.numAgSegments
#     partial_heterotypic_immunity_rate = self.partial_heterotypic_immunity_rate
#     complete_heterotypic_immunity_rate = self.complete_heterotypic_immunity_rate
#     homotypic_immunity_rate = self.homotypic_immunity_rate
#
#     if self.vaccine[uid] is not None and self.is_vaccine_immune(
#             uid, infecting_strain
#     ):
#         return False
#
#     current_infecting_strains = (
#         i.strain[:numAgSegments] for i in current_infections
#     )
#     if infecting_strain[:numAgSegments] in current_infecting_strains:
#         return False
#
#     def is_complete_antigenic_match():
#         immune_strains = (s[:numAgSegments] for s in self.immunity[uid].keys())
#         return infecting_strain[:numAgSegments] in immune_strains
#
#     def has_shared_antigenic_genotype():
#         for i in range(numAgSegments):
#             immune_genotypes = (strain[i] for strain in self.immunity[uid].keys())
#             if infecting_strain[i] in immune_genotypes:
#                 return True
#         return False
#
#     if is_complete_antigenic_match():
#         return rnd.random() > homotypic_immunity_rate
#
#     if has_shared_antigenic_genotype():
#         return rnd.random() > partial_heterotypic_immunity_rate
#
#     # If the strain is complete heterotypic
#     return rnd.random() > complete_heterotypic_immunity_rate
#
#
# def contact_event(self, contacts, infected_uids):
#     if len(infected_uids) == 0:
#         print("[Warning] No infected hosts in a contact event. Skipping")
#         return
#
#     h1_uids = np.random.choice(infected_uids, size=contacts)
#     h2_uids = np.random.choice(self.sim.people.alive.uids, size=contacts)
#     rnd_nums = np.random.random(size=contacts)
#     counter = 0
#
#     # based on prior infections and current infections, the relative risk of subsequent infections
#     infecting_probability_map = {
#         0: 1,
#         1: 0.61,
#         2: 0.48,
#         3: 0.33,
#     }
#
#     for h1_uid, h2_uid, rnd_num in zip(h1_uids, h2_uids, rnd_nums):
#         # h1 = infected_pop[h1_ind]
#         # h2 = self.host_pop[h2_ind]
#
#         # If the contact is the same as the infected host, pick another host at random
#         while h1_uid == h2_uid:
#             h2_uid = rnd.choice(self.sim.people.alive.uids)
#
#         infecting_probability = infecting_probability_map.get(
#             self.prior_infections[h2_uid], 0
#         )
#         infecting_probability *= (
#             self.pars.rel_beta
#         )  # Scale by this calibration parameter
#
#         # No infection occurs
#         if rnd_num > infecting_probability:
#             continue
#         else:
#             counter += 1
#             h2_previously_infected = self.isInfected(uid=h2_uid)
#
#             if len(self.infecting_pathogen[h1_uid]) == 1:
#                 if self.can_variant_infect_host(
#                     h2_uid, self.infecting_pathogen[h1_uid][0].strain
#                 ):
#                     self.infect_with_pathogen(
#                         h2_uid, self.infecting_pathogen[h1_uid][0]
#                     )
#             else:
#                 self.coInfected_contacts(h1_uid, h2_uid)
#
#             # in this case h2 was not infected before but is infected now
#             if not h2_previously_infected and self.isInfected(h2_uid):
#                 infected_uids.append(h2_uid)
#     return counter