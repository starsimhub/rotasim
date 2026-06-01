"""
Process incidence from the model and data — MAL-ED birth-cohort version.

Calibration targets per site:
  1. Symptomatic incidence rate per age bin (cases per 100 person-months),
     from `maled_data/ir_by_age_symp_<site>.csv`.
  2. Quartiles (median, Q25, Q75) of age at first rotavirus infection,
     from `maled_data/first_infection_<site>.csv` (events only).

GOF formula (equal-weight after standardisation; weights tunable):
  GOF_inc      = sum over age bins of (log(IR_target + eps) - log(IR_model + eps))^2
  GOF_first    = ((med_target - med_model)^2 + 0.5*((Q25_diff)^2 + (Q75_diff)^2)) / scale^2
                 where scale = median age (months) of the target — keeps the term unitless
  GOF          = w_inc * GOF_inc + w_first * GOF_first

The age-bin mapping from the analyzer's fine bins to MAL-ED bins (months):
  '<6 m'   = '0-2', '2-4', '4-6'
  '6-11 m' = '6-12'
  '12-23 m'= '12-24'
  '24-35 m'= '24-36'
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import sciris as sc

# Reuse the age-symptom logistic from the UK module.
from process_incidence_uk_age import calculate_symptom_probability  # noqa: F401

warnings.simplefilter("ignore", FutureWarning)
pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)

thisdir = sc.thispath(__file__)
DATA_DIR = thisdir / 'maled_data'

# MAL-ED age bins (in order) and the fine-grained analyzer bins that compose each.
MALED_AGE_BINS = ['<6 m', '6-11 m', '12-23 m', '24-35 m']

# Months covered by each MAL-ED bin (low inclusive, high exclusive).
MALED_AGE_MONTH_RANGE = {
    '<6 m':    (0.0, 6.0),
    '6-11 m':  (6.0, 12.0),
    '12-23 m': (12.0, 24.0),
    '24-35 m': (24.0, 36.0),
}

# Analyzer fine-bin labels -> MAL-ED bin label (only bins that fall fully inside an MAL-ED bin).
FINE_TO_MALED = {
    '0-2':   '<6 m',
    '2-4':   '<6 m',
    '4-6':   '<6 m',
    '6-12':  '6-11 m',
    '12-24': '12-23 m',
    '24-36': '24-35 m',
    # '36-48', '48-60', '60+' are >=36 months -> ignored for these sites.
}

# Approximate midpoint (in months) of each analyzer fine bin — used when we need
# a numeric age estimate per infection event. Good enough for quartile estimation
# given the bin width is at most 12 months and the bins are narrow at young ages.
FINE_BIN_MIDPOINT_MONTHS = {
    '0-2':   1.0,
    '2-4':   3.0,
    '4-6':   5.0,
    '6-12':  9.0,
    '12-24': 18.0,
    '24-36': 30.0,
    '36-48': 42.0,
    '48-60': 54.0,
    '60+':   72.0,
}

LOG_EPS = 0.01  # small offset to avoid log(0) when an age bin has zero modelled cases


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------
def load_ir_targets(site: str) -> pd.DataFrame:
    """Symptomatic IR targets for `site` (one of 'bangladesh', 'pakistan'). Returns
    a DataFrame indexed by MAL-ED age bin with columns: cases, PT, IR."""
    path = DATA_DIR / f'ir_by_age_symp_{site.lower()}.csv'
    df = pd.read_csv(path)
    df = df.set_index('age_cat').reindex(MALED_AGE_BINS)
    return df


def load_first_infection_quartiles(site: str) -> dict:
    """Quartiles (in months) of age at first rotavirus infection from MAL-ED CoxDat,
    using observed events only. Returns dict with keys: median, q25, q75, n_events,
    n_total, scale (= median, for GOF normalisation)."""
    path = DATA_DIR / f'first_infection_{site.lower()}.csv'
    df = pd.read_csv(path)
    events = df[df['event_observed'] == 1]['age_event_months'].dropna()
    q25, med, q75 = np.quantile(events, [0.25, 0.5, 0.75])
    return dict(
        median=float(med),
        q25=float(q25),
        q75=float(q75),
        n_events=int(len(events)),
        n_total=int(len(df)),
        scale=float(med),  # use the target median as the natural scale for normalisation
    )


def load_targets(site: str) -> dict:
    """Bundle of all calibration targets for a site."""
    return dict(
        site=site,
        ir_by_age=load_ir_targets(site),
        first_infection=load_first_infection_quartiles(site),
    )


# ---------------------------------------------------------------------------
# Model output processing
# ---------------------------------------------------------------------------
def _apply_age_symptom_filter(dat: pd.DataFrame, symptom_model: str,
                               beta0=0.0, beta1=0.0, beta2=0.0,
                               reporting_rate: float | None = None,
                               rng_seed: int | None = None) -> pd.DataFrame:
    """Apply the same symptom-probability + reporting filter used by the UK module.
    Operates on a copy of `dat`; expects columns `age_months_est`, `n_infections`,
    and (if reporting_rate is given) `severity`."""
    out = dat.copy()
    if symptom_model in ('age_and_infection', 'age_and_infection_simple'):
        probs = out.apply(
            lambda row: calculate_symptom_probability(
                age_months=row['age_months_est'],
                n_infections=row['n_infections'],
                symptom_model='age_only',
                beta0=beta0, beta1=beta1, beta2=beta2, beta3=0,
            ),
            axis=1,
        )
        rng = np.random.default_rng(rng_seed)
        keep = rng.random(len(out)) < probs.values
        out = out.loc[keep].copy()
        if reporting_rate is not None and 'severity' in out.columns:
            rep_keep = rng.random(len(out)) < (reporting_rate * out['severity'].values)
            out = out.loc[rep_keep].copy()
    return out


def compute_model_ir_by_age(events: pd.DataFrame, person_months_by_bin: dict[str, float]) -> pd.DataFrame:
    """Aggregate symptomatic events to MAL-ED bins and divide by person-time.

    Args:
      events: filtered infection events with column 'Age' (analyzer fine bin label).
      person_months_by_bin: total person-months in each MAL-ED bin
                            (e.g. {'<6 m': X, '6-11 m': Y, ...}).

    Returns DataFrame indexed by MAL-ED age bin with columns cases, PT, IR (per 100 PM).
    """
    df = events.copy()
    df['maled_bin'] = df['Age'].map(FINE_TO_MALED)
    df = df.dropna(subset=['maled_bin'])
    cases = df.groupby('maled_bin').size().reindex(MALED_AGE_BINS).fillna(0).astype(int)

    pt = pd.Series({b: person_months_by_bin.get(b, 0.0) for b in MALED_AGE_BINS})
    ir = np.where(pt.values > 0, (cases.values / pt.values) * 100.0, 0.0)
    return pd.DataFrame({'cases': cases.values, 'PT': pt.values, 'IR': ir},
                        index=MALED_AGE_BINS)


def compute_model_first_inf_quartiles(events: pd.DataFrame,
                                       censor_at_months: float | None = 36.0) -> dict:
    """Quartiles of age at first infection from modelled events.

    Args:
      events: full (unfiltered or symptomatic-filtered — caller's choice) event log,
              must contain 'n_infections' and 'age_months_est'.
      censor_at_months: drop events occurring at ages above this threshold to match
                        MAL-ED's 36-month follow-up window. None = no censoring.
    """
    first = events[events['n_infections'] == 1]
    ages = first['age_months_est'].dropna().values
    if censor_at_months is not None:
        ages = ages[ages <= censor_at_months]
    if len(ages) == 0:
        return dict(median=np.nan, q25=np.nan, q75=np.nan, n_events=0)
    q25, med, q75 = np.quantile(ages, [0.25, 0.5, 0.75])
    return dict(median=float(med), q25=float(q25), q75=float(q75), n_events=int(len(ages)))


# ---------------------------------------------------------------------------
# Top-level: process a sim's events into the comparable summary
# ---------------------------------------------------------------------------
def process_model(dat: pd.DataFrame,
                  person_months_by_bin: dict[str, float],
                  symptom_model: str = 'age_and_infection_simple',
                  beta0: float = 0.0, beta1: float = 0.0, beta2: float = 0.0,
                  reporting_rate: float | None = None,
                  censor_at_months: float = 36.0,
                  calibration_window: tuple[float, float] = (5.0, 10.0),
                  rng_seed: int | None = None,
                  verbose: bool = False) -> dict:
    """Extract MAL-ED-comparable summary statistics from a sim's InfectedStrainStats output.

    `dat` is the dataframe returned by `analyzer.to_df()` — columns include
    id, Strain, CollectionTime, Age (categorical bin), n_infections, severity.

    `person_months_by_bin` must be supplied by the caller. The cleanest way to
    compute it is from the sim's age-distribution snapshot: for each MAL-ED bin,
    PT = (mean number of agents whose age falls in that bin during the window)
         * window length in months. We expose it as an argument rather than
         computing it here, so the same function can serve birth-cohort and
         steady-state slicings.
    """
    df = dat.copy()
    df = df[(df['CollectionTime'] >= calibration_window[0]) &
            (df['CollectionTime'] <  calibration_window[1])]

    # Per-agent infection order within the calibration window.
    df = df.sort_values(['id', 'CollectionTime'])
    df['n_infections'] = df.groupby('id').cumcount() + 1
    df['age_months_est'] = df['Age'].map(FINE_BIN_MIDPOINT_MONTHS)

    # Symptomatic filtering (mirrors UK pipeline).
    df_symp = _apply_age_symptom_filter(df, symptom_model=symptom_model,
                                        beta0=beta0, beta1=beta1, beta2=beta2,
                                        reporting_rate=reporting_rate,
                                        rng_seed=rng_seed)
    df_symp_36 = df_symp[df_symp['age_months_est'] <= censor_at_months]

    ir = compute_model_ir_by_age(df_symp_36, person_months_by_bin)

    # First-infection quartiles: use all infections (not just symptomatic) so the
    # survival distribution matches MAL-ED's TAC-based first-positive definition.
    df_all_36 = df[df['age_months_est'] <= censor_at_months]
    quartiles = compute_model_first_inf_quartiles(df_all_36, censor_at_months=censor_at_months)

    if verbose:
        print(f"  symptomatic events <=36m: {len(df_symp_36)}")
        print(f"  first-infection events <=36m: {quartiles['n_events']}")
        print(f"  IR by age bin: {ir['IR'].to_dict()}")
        print(f"  first-inf quartiles (months): {quartiles}")

    return dict(ir_by_age=ir, first_infection=quartiles)


# ---------------------------------------------------------------------------
# GOF
# ---------------------------------------------------------------------------
def gof_incidence(model_ir: pd.DataFrame, target_ir: pd.DataFrame) -> float:
    """Sum of squared log-IR differences across MAL-ED age bins."""
    m = np.log(model_ir['IR'].values + LOG_EPS)
    t = np.log(target_ir['IR'].values + LOG_EPS)
    return float(np.sum((m - t) ** 2))


def gof_first_infection(model_q: dict, target_q: dict) -> float:
    """Squared difference in median/Q25/Q75 of age-at-first-infection, normalised
    by the target median so the result is unitless and on a similar scale to GOF_inc."""
    scale = target_q['scale']
    if not np.isfinite(model_q['median']):
        return 100.0  # heavy penalty if the model produces no first infections
    med = (model_q['median'] - target_q['median']) / scale
    q25 = (model_q['q25']    - target_q['q25'])    / scale
    q75 = (model_q['q75']    - target_q['q75'])    / scale
    return float(med ** 2 + 0.5 * (q25 ** 2 + q75 ** 2))


def gof(model_out: dict, targets: dict,
        w_inc: float = 1.0, w_first: float = 1.0) -> dict:
    g_inc = gof_incidence(model_out['ir_by_age'], targets['ir_by_age'])
    g_first = gof_first_infection(model_out['first_infection'], targets['first_infection'])
    total = w_inc * g_inc + w_first * g_first
    return dict(gof=total, gof_incidence=g_inc, gof_first_infection=g_first,
                w_inc=w_inc, w_first=w_first)


# ---------------------------------------------------------------------------
# Quick sanity test (no sim — just exercise loaders + GOF on hand-built model output)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    for site in ('bangladesh', 'pakistan'):
        print(f"\n=== {site.title()} ===")
        targets = load_targets(site)
        print("Target IR by age:")
        print(targets['ir_by_age'])
        print("Target first-infection quartiles (months):", targets['first_infection'])

        # Identity check: GOF of target-against-target should be ~0.
        model_out = dict(ir_by_age=targets['ir_by_age'].copy(),
                          first_infection={k: targets['first_infection'][k]
                                            for k in ('median', 'q25', 'q75')})
        identity_gof = gof(model_out, targets)
        print(f"Identity GOF (should be ~0): {identity_gof}")

        # Perturbation check: bump model IR by 50% in every bin; expect non-zero GOF.
        perturbed = targets['ir_by_age'].copy()
        perturbed['IR'] = perturbed['IR'] * 1.5
        model_perturbed = dict(ir_by_age=perturbed, first_infection=model_out['first_infection'])
        print(f"Perturbed (+50% IR) GOF: {gof(model_perturbed, targets)}")
