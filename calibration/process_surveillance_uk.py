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

# Bin labels as they appear in the spreadsheet, in age order. The left edges (months)
# define the Surveillance observer's bins; the final bin is open-ended (36+) unless capped.
UK_BIN_LABELS = ['<6 m', '6-11 m', '12-23 m', '24-35 m', '36 m +']
UK_BIN_EDGES_M = (0.0, 6.0, 12.0, 24.0, 36.0)
# Short feature keys used as HM column names (sanitised).
UK_FEATURE_KEYS = ['prop_0_6', 'prop_6_11', 'prop_12_23', 'prop_24_35', 'prop_36plus']
# Number of leading bins kept for a given upper age cap (cases >= cap are excluded). Capping
# matches the cohort window and avoids modelling adult cases (the symptom model extrapolates
# poorly to adults under all-age surveillance). cap None -> all 5 bins (open 36+ tail).
_NBINS_FOR_CAP = {None: 5, 36.0: 4, 24.0: 3}


def load_uk_counts(years=(2008, 2009, 2010, 2011, 2012)):
    """Return pooled case counts per age bin (numpy int array, length 5) over `years`."""
    df = pd.read_excel(UK_XLSX, sheet_name=0, header=0)
    df = df.rename(columns={df.columns[0]: 'bin'}).set_index('bin')
    df = df.reindex(UK_BIN_LABELS)            # enforce age order
    cols = [y for y in years if y in df.columns]
    counts = df[cols].sum(axis=1).to_numpy()
    return counts.astype(int)


def load_targets_uk(years=(2008, 2009, 2010, 2011, 2012), cap_age_m=36.0):
    """UK surveillance target: per-bin case proportions + multinomial SEs + raw counts.

    cap_age_m caps the observation at an upper age (cases >= cap excluded) to match the cohort
    window: 36 -> 4 bins (<6,6-11,12-23,24-35; matches MALEDCohort's 36mo censoring), 24 -> 3
    bins (<6,6-11,12-23; matches the Bangladesh IR fit bins), None -> all 5 bins (open 36+ tail).
    Proportions/SEs renormalize within the kept bins (the total N is the kept-bin total).

    Returns dict with: counts, total, proportions, se (all over the kept bins),
      bin_labels, feature_keys, bin_edges_m, cap_age_m.
    """
    if cap_age_m not in _NBINS_FOR_CAP:
        raise ValueError(f"cap_age_m must be one of {sorted(k for k in _NBINS_FOR_CAP if k)} or None")
    nb = _NBINS_FOR_CAP[cap_age_m]
    counts = load_uk_counts(years)[:nb]                 # keep leading bins only (drop older/open tail)
    total = int(counts.sum())
    prop = counts / total                               # renormalized within the capped range
    se = np.sqrt(prop * (1.0 - prop) / total)
    # edges for the observer: leading edges; with a cap the last kept edge is the upper bound.
    edges = (UK_BIN_EDGES_M[:nb] + ((cap_age_m,) if cap_age_m is not None else ()))
    return dict(counts=counts, total=total, proportions=prop, se=se,
                bin_labels=UK_BIN_LABELS[:nb], feature_keys=UK_FEATURE_KEYS[:nb],
                bin_edges_m=edges, cap_age_m=cap_age_m)


if __name__ == '__main__':
    for cap in (None, 36.0, 24.0):
        t = load_targets_uk(cap_age_m=cap)
        print(f"\nUK pre-vax surveillance, cap={cap} mo -> {len(t['counts'])} bins, "
              f"pooled {t['total']} cases, edges={t['bin_edges_m']}")
        for lab, c, p, s in zip(t['bin_labels'], t['counts'], t['proportions'], t['se']):
            print(f"  {lab:8s}  n={c:5d}  prop={p:.3f}  se={s:.4f}")
