"""UK pre-vaccine surveillance targets: the age-DISTRIBUTION of (genotyped) symptomatic
rotavirus cases, England & Wales 2008-2012, pooled over years.

Cross-sectional surveillance gives a case age-distribution (a shape), NOT a birth cohort
-- so there is no first-detection KM or repeat fraction (unlike MAL-ED). We calibrate to
the SHAPE: the proportion of cases in each of 5 age bins. Genotyping was not age-biased
(confirmed by A. Kraay), so the genotyped subset preserves the age-distribution and any
overall reporting rate cancels out of the proportions.

Bins match the MAL-ED bins plus an open older tail:
    <6 / 6-11 / 12-23 / 24-35 / 36+ months   (left edges 0,6,12,24,36).

Each bin proportion is treated as an HM feature with a multinomial SE
sqrt(p(1-p)/N_total) -- compatible with the existing (mean, std) observation machinery.
"""
import pathlib
import numpy as np
import pandas as pd

THISDIR = pathlib.Path(__file__).resolve().parent
UK_XLSX = THISDIR / 'UK_prevax_age_2008_2012.xlsx'
# Two-era file: PreVaccine (2008-2012) + PostVaccine (2014-2019; vaccine-derived strains excluded).
UK_ERA_XLSX = THISDIR / 'UK_age_byEra.xlsx'
UK_ERA_YEARS = {'pre': (2008, 2009, 2010, 2011, 2012),
                'post': (2015, 2016, 2017, 2018, 2019)}   # drop 2014 (intro/ramp-up year)

# Bin labels as they appear in the spreadsheet, in age order. Finer bins through <5y (the
# xlsx now resolves 36-47 / 48-59 mo instead of an open 36+ tail). All bins are CLOSED.
# Default cap = 60 mo (<5y): keeps the high/uniform-care-seeking range while including the 2-5y
# bins that diagnose whether infection-number symptoms overshoot older-child reinfections.
UK_BIN_LABELS = ['<6 m', '6-11 m', '12-23 m', '24-35 m', '36-47 m', '48-59 m']
UK_BIN_EDGES_M = (0.0, 6.0, 12.0, 24.0, 36.0, 48.0, 60.0)
# Short feature keys used as HM column names (sanitised).
UK_FEATURE_KEYS = ['prop_0_6', 'prop_6_11', 'prop_12_23', 'prop_24_35', 'prop_36_47', 'prop_48_59']
# Leading bins kept for a given upper age cap (sim cases >= cap excluded; proportions renormalize
# within the kept range). 60 -> all 6 (<5y, default), 36 -> <3y/4 bins, 24 -> <2y/3 bins.
_NBINS_FOR_CAP = {60.0: 6, 36.0: 4, 24.0: 3}


def load_uk_counts(years=(2008, 2009, 2010, 2011, 2012)):
    """Return pooled case counts per age bin (numpy int array, length 5) over `years`."""
    df = pd.read_excel(UK_XLSX, sheet_name=0, header=0)
    df = df.rename(columns={df.columns[0]: 'bin'}).set_index('bin')
    df = df.reindex(UK_BIN_LABELS)            # enforce age order
    cols = [y for y in years if y in df.columns]
    counts = df[cols].sum(axis=1).to_numpy()
    return counts.astype(int)


def load_targets_uk(years=(2008, 2009, 2010, 2011, 2012), cap_age_m=60.0):
    """UK surveillance target: per-bin case proportions + multinomial SEs + raw counts.

    cap_age_m caps the observation at an upper age (sim cases >= cap excluded): 60 -> 6 bins
    (<5y; default — high/uniform care-seeking, includes the 2-5y drop-off diagnostic), 36 -> 4
    bins (<3y; matches MALEDCohort's 36mo censoring), 24 -> 3 bins (<2y; Bangladesh IR fit bins).
    Proportions/SEs renormalize within the kept bins (total N is the kept-bin total).

    Returns dict with: counts, total, proportions, se (all over the kept bins),
      bin_labels, feature_keys, bin_edges_m, cap_age_m.
    """
    if cap_age_m not in _NBINS_FOR_CAP:
        raise ValueError(f"cap_age_m must be one of {sorted(_NBINS_FOR_CAP)}")
    nb = _NBINS_FOR_CAP[cap_age_m]
    counts = load_uk_counts(years)[:nb]                 # keep leading bins only
    total = int(counts.sum())
    prop = counts / total                               # renormalized within the capped range
    se = np.sqrt(prop * (1.0 - prop) / total)
    edges = UK_BIN_EDGES_M[:nb] + (cap_age_m,)          # leading edges + the cap as upper bound
    return dict(counts=counts, total=total, proportions=prop, se=se,
                bin_labels=UK_BIN_LABELS[:nb], feature_keys=UK_FEATURE_KEYS[:nb],
                bin_edges_m=edges, cap_age_m=cap_age_m)


def load_targets_uk_era(era='pre', cap_age_m=60.0, years=None):
    """UK surveillance target for a given vaccine era, from UK_age_byEra.xlsx.
    era='pre' (2008-2012, no vaccine) or 'post' (2015-2019, vaccine ~90% coverage, vaccine-derived
    strains already excluded). Same 6-bin structure / cap semantics as load_targets_uk."""
    if era not in ('pre', 'post'):
        raise ValueError("era must be 'pre' or 'post'")
    if cap_age_m not in _NBINS_FOR_CAP:
        raise ValueError(f"cap_age_m must be one of {sorted(_NBINS_FOR_CAP)}")
    years = years or UK_ERA_YEARS[era]
    sheet = 'PreVaccine' if era == 'pre' else 'PostVaccine'
    df = pd.read_excel(UK_ERA_XLSX, sheet_name=sheet, header=0)
    df = df.rename(columns={df.columns[0]: 'bin'}).set_index('bin').reindex(UK_BIN_LABELS)
    cols = [y for y in years if y in df.columns]
    counts = df[cols].sum(axis=1).to_numpy().astype(int)
    nb = _NBINS_FOR_CAP[cap_age_m]
    counts = counts[:nb]; total = int(counts.sum()); prop = counts / total
    se = np.sqrt(prop * (1.0 - prop) / total)
    return dict(counts=counts, total=total, proportions=prop, se=se,
                bin_labels=UK_BIN_LABELS[:nb], feature_keys=UK_FEATURE_KEYS[:nb],
                bin_edges_m=UK_BIN_EDGES_M[:nb] + (cap_age_m,), cap_age_m=cap_age_m, era=era, years=tuple(cols))


if __name__ == '__main__':
    for era in ('pre', 'post'):
        t = load_targets_uk_era(era)
        print(f"\nUK {era}-vaccine ({t['years']}): N={t['total']}")
        for lab, c, p in zip(t['bin_labels'], t['counts'], t['proportions']):
            print(f"  {lab:8s} n={c:5d}  prop={p:.3f}")
    for cap in (60.0, 36.0, 24.0):
        t = load_targets_uk(cap_age_m=cap)
        print(f"\nUK pre-vax surveillance, cap={cap} mo -> {len(t['counts'])} bins, "
              f"pooled {t['total']} cases, edges={t['bin_edges_m']}")
        for lab, c, p, s in zip(t['bin_labels'], t['counts'], t['proportions'], t['se']):
            print(f"  {lab:8s}  n={c:5d}  prop={p:.3f}  se={s:.4f}")
