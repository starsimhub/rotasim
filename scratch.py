import sciris as sc
import rotasim as rs

N = 2_000
timelimit = 10
verbose = False



sc.heading("Testing default parameters")




# Check old baseline

with sc.timer() as T:
    rota = rs.Sim(N=N, timelimit=timelimit, rota_kwargs={'vaccination_time': 5, 'time_to_equilibrium': 0}, verbose=verbose)
    # rota.run()
    sc.profile(rota.run, [rs.Rota.contact_event, rs.Rota.can_variant_infect_host, rs.Rota.is_vaccine_immune])
    events = rota.connectors["rota"].event_dict
print(f"Defaults matched:\n{events}")


def coInfected_contacts(self, h1_uid, h2_uid):
    random_number = rnd.random()
    if random_number < 0.02:  # giving all the possible strains
        for path in self.infecting_pathogen[h1_uid]:
            if self.can_variant_infect_host(h2_uid, path.strain):
                self.infect_with_pathogen(h2_uid, path)
    else:  # give only one strain depending on fitness
        host1paths = list(self.infecting_pathogen[h1_uid])
        # Sort by fitness first and randomize the ones with the same fitness
        host1paths.sort(
            key=lambda path: (path.get_fitness(), rnd.random()), reverse=True
        )
        for path in host1paths:
            if self.can_variant_infect_host(h2_uid, path.strain):
                infected = self.infect_with_pathogen(h2_uid, path)
                if infected:
                    break