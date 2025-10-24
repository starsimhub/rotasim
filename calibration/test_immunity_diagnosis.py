"""
Diagnostic test to check if immunity is being applied correctly
"""
import rotasim as rs
import starsim as ss
import numpy as np

print("="*80)
print("Immunity Diagnostic Test")
print("="*80)

# Create a simple sim with 2 strains
sim = rs.Sim(
    n_agents=1000,
    start='2000-01-01',
    stop='2000-04-01',  # 3 months for diagnosis
    verbose=False,  # Turn off verbose to reduce output
    scenario='baseline',
    base_beta=0.16,
    override_prevalence=0.002,
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(70)),
        ss.Deaths(death_rate=ss.peryear(20)),
    ],
)

sim.init()

# Get immunity connector
immunity_conn = sim.get_connector_by_type('RotaImmunityConnector')

print(f"\n{'='*80}")
print("Immunity Parameters:")
print(f"{'='*80}")
print(f"  homotypic_immunity_efficacy: {immunity_conn.pars.homotypic_immunity_efficacy}")
print(f"  partial_heterotypic_immunity_efficacy: {immunity_conn.pars.partial_heterotypic_immunity_efficacy}")
print(f"  complete_heterotypic_immunity_efficacy: {immunity_conn.pars.complete_heterotypic_immunity_efficacy}")
print(f"  naive_immunity_efficacy: {immunity_conn.pars.naive_immunity_efficacy}")
print(f"  immunity_waning_delay: {immunity_conn.pars.immunity_waning_delay}")

print(f"\n{'='*80}")
print("Diseases in simulation:")
print(f"{'='*80}")
for disease_name, disease in sim.diseases.items():
    if hasattr(disease, 'G'):
        print(f"  {disease_name}: G{disease.G}P{disease.P}")

print(f"\n{'='*80}")
print("Running simulation for 3 months...")
print(f"{'='*80}")

# Run full simulation
sim.run()

# Check rel_sus after simulation
print(f"\nAfter simulation:")
for disease_name, disease in sim.diseases.items():
    if hasattr(disease, 'G'):
        # Get unique rel_sus values
        unique_rel_sus = np.unique(disease.rel_sus[:])
        print(f"\n  {disease_name}:")
        print(f"    Unique rel_sus values (first 10): {unique_rel_sus[:10]}")

        # Count how many people have each level of susceptibility
        fully_susceptible = np.sum(disease.rel_sus[:] >= 0.99)
        some_protection = np.sum((disease.rel_sus[:] > 0.1) & (disease.rel_sus[:] < 0.99))
        high_protection = np.sum(disease.rel_sus[:] <= 0.1)

        print(f"    Fully susceptible (rel_sus≥0.99): {fully_susceptible}")
        print(f"    Some protection (0.1<rel_sus<0.99): {some_protection}")
        print(f"    High protection (rel_sus≤0.1): {high_protection}")

        # Check infections
        n_ever_infected = np.sum(disease.ti_infected > 0)
        n_ever_recovered = np.sum(disease.ti_recovered > 0)
        print(f"    Ever infected: {n_ever_infected}, Ever recovered: {n_ever_recovered}")

print(f"\n{'='*80}")
print("Checking immunity bitmasks after 3 months:")
print(f"{'='*80}")

# Check how many people have been exposed to different strains
n_exposed_any = np.sum(immunity_conn.has_immunity[:])
print(f"People with any immunity: {n_exposed_any}")

# Check bitmask patterns
for i in range(min(20, sim.pars.n_agents)):
    if immunity_conn.has_immunity[i]:
        G_bits = immunity_conn.exposed_G_bitmask[i]
        P_bits = immunity_conn.exposed_P_bitmask[i]
        GP_bits = immunity_conn.exposed_GP_bitmask[i]
        print(f"  Agent {i}: G_bits={bin(G_bits)}, P_bits={bin(P_bits)}, GP_bits={bin(GP_bits)}")

print(f"\n{'='*80}")
print("✓ Diagnostic complete")
print(f"{'='*80}")
