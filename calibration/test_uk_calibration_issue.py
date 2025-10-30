"""
Diagnostic test to understand why UK calibration shows 0 infections with SIRS model
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

print("="*60)
print("Diagnosing UK Calibration Issue with SIRS Model")
print("="*60)

# Replicate EXACT UK calibration setup
print("\nTest 1: Exact UK calibration setup (5 years, no burn-in)")
print("-"*60)

sim = rs.Sim(
    n_agents=5000,
    start='2008-01-01',  # No burn-in!
    stop='2013-01-01',   # 5 years
    verbose=False,
    scenario='baseline',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print(f"Setup: 5000 agents, 2008-01-01 to 2013-01-01")
print(f"Initial prevalence: {0.002*100:.2f}% = {int(5000*0.002)} initially infected agents")
print(f"Base beta: {0.16}")

sim.run()

print("\n" + "="*60)
print("Results from Test 1:")
print("="*60)

# Check analyzer
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

if analyzer:
    df = analyzer.to_df()
    if df is not None and len(df) > 0:
        print(f"\n✓ Analyzer collected {len(df)} infection events")

        # By year
        df['Year'] = df['CollectionTime'] / 365.0 + 2008
        yearly = df.groupby(df['Year'].astype(int)).size()
        print("\nInfections by year:")
        for year, count in yearly.items():
            print(f"  {year}: {count}")
    else:
        print("\n✗ Analyzer returned empty DataFrame")
else:
    print("\n✗ Analyzer not found")

# Check disease-level counts
print("\nDisease-level infection totals:")
total_infections = 0
for disease in sim.diseases.values():
    if hasattr(disease, 'n_infections'):
        n_inf = disease.n_infections[sim.people.alive].sum()
        total_infections += n_inf
        if n_inf > 0:
            print(f"  {disease.name}: {n_inf:.0f} infections")

print(f"\n  Total: {total_infections:.0f} infections")

# Check new_infections results
print("\nChecking new_infections results:")
for disease in sim.diseases.values():
    if hasattr(disease, 'results') and 'new_infections' in disease.results:
        new_inf = disease.results['new_infections']
        total_new = new_inf.sum()
        if total_new > 0:
            print(f"  {disease.name}: {total_new:.0f} new infections in results")

print("\n" + "="*60)
print("Test 2: With 5-year burn-in (like SIRS test)")
print("-"*60)

sim2 = rs.Sim(
    n_agents=5000,
    start='2003-01-01',  # 5 year burn-in
    stop='2013-01-01',   # 10 years total
    verbose=False,
    scenario='baseline',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print(f"Setup: 5000 agents, 2003-01-01 to 2013-01-01")
print(f"Burn-in: 2003-2007 (5 years)")
print(f"Follow-up: 2008-2012 (5 years)")

sim2.run()

print("\n" + "="*60)
print("Results from Test 2:")
print("="*60)

# Check analyzer
analyzer2 = None
for a in sim2.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer2 = a
        break

if analyzer2:
    df2 = analyzer2.to_df()
    if df2 is not None and len(df2) > 0:
        print(f"\n✓ Analyzer collected {len(df2)} infection events")

        # Split by burn-in vs follow-up
        df2['Year'] = df2['CollectionTime'] / 365.0 + 2003
        burnin = df2[df2['Year'] < 2008]
        followup = df2[(df2['Year'] >= 2008) & (df2['Year'] < 2013)]

        print(f"\nBurn-in period (2003-2007): {len(burnin)} infections")
        print(f"Follow-up period (2008-2012): {len(followup)} infections")

        if len(followup) > 0:
            yearly = followup.groupby(followup['Year'].astype(int)).size()
            print("\nFollow-up infections by year:")
            for year, count in yearly.items():
                print(f"  {year}: {count}")
    else:
        print("\n✗ Analyzer returned empty DataFrame")

# Check disease counts
print("\nDisease-level infection totals:")
total_infections2 = 0
for disease in sim2.diseases.values():
    if hasattr(disease, 'n_infections'):
        n_inf = disease.n_infections[sim2.people.alive].sum()
        total_infections2 += n_inf
        if n_inf > 0:
            print(f"  {disease.name}: {n_inf:.0f} infections")

print(f"\n  Total: {total_infections2:.0f} infections")

print("\n" + "="*60)
print("Diagnosis Summary")
print("="*60)

print(f"\nTest 1 (no burn-in, 5 years):")
print(f"  Total infections: {total_infections:.0f}")
if total_infections == 0:
    print("  ✗ PROBLEM: Disease died out or never spread")
else:
    print("  ✓ Disease circulating")

print(f"\nTest 2 (5-year burn-in, 10 years total):")
print(f"  Total infections: {total_infections2:.0f}")
if total_infections2 == 0:
    print("  ✗ PROBLEM: Disease died out")
else:
    print("  ✓ Disease circulating")

print("\n" + "="*60)
print("Hypothesis:")
print("="*60)
if total_infections == 0 and total_infections2 > 0:
    print("✓ CONFIRMED: UK calibration needs burn-in period!")
    print("  - Without burn-in: disease dies out from initial 10 infections")
    print("  - With burn-in: disease establishes endemic circulation")
elif total_infections == 0 and total_infections2 == 0:
    print("✗ BIGGER PROBLEM: Disease dying out even with burn-in")
    print("  - May need to adjust SIRS parameters (13 weeks too long?)")
    print("  - May need higher base_beta")
else:
    print("? UNEXPECTED: Disease survives without burn-in")
