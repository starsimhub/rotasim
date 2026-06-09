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


def km_quartiles(times, observed):
    """Censoring-aware Kaplan-Meier quartiles (q25, median, q75) of a time-to-event
    distribution. `times` = per-subject time to event-or-censoring; `observed` = 1 if the
    event was seen, 0 if right-censored. A quantile is NaN if KM survival never drops to
    that level (too much censoring). No external dependency."""
    times = np.asarray(times, float)
    observed = np.asarray(observed).astype(bool)
    n = len(times)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    t_sorted = np.sort(times)
    event_times = np.unique(times[observed])
    surv = 1.0
    res = {}
    for ut in event_times:
        at_risk = n - int(np.searchsorted(t_sorted, ut, side='left'))
        d = int(((times == ut) & observed).sum())
        if at_risk > 0:
            surv *= (1.0 - d / at_risk)
        for p in (0.25, 0.5, 0.75):
            if p not in res and surv <= 1.0 - p:
                res[p] = float(ut)
    return res.get(0.25, float('nan')), res.get(0.5, float('nan')), res.get(0.75, float('nan'))


def load_first_infection_km(site: str) -> dict:
    """Censoring-aware KM quartiles of age-at-first-DETECTION from MAL-ED CoxDat (all
    children, using the right-censoring indicator) -- the cohort-consistent first-infection
    target. Unlike load_first_infection_quartiles (events only), this keeps the ~44%
    censored children, so the median is not biased young."""
    path = DATA_DIR / f'first_infection_{site.lower()}.csv'
    df = pd.read_csv(path)
    df = df[df['age_event_months'] > 0]  # drop a <=0 edge row
    q25, med, q75 = km_quartiles(df['age_event_months'].values, df['event_observed'].values)
    return dict(median=float(med), q25=float(q25), q75=float(q75),
                n_total=int(len(df)), scale=float(med))


# Repeat-detected fraction target (among children with >=1 detected infection, the
# fraction with >=2 detected), in the TAC cohort frame -- matches MALEDCohort's
# repeat_detected_frac. Bangladesh: 60/149 (A. Kraay, 2026-06-09).
REPEAT_FRAC = {'bangladesh': dict(frac=0.403, n=149)}


def load_repeat_fraction(site: str):
    """Target repeat-detected fraction + binomial SE (for GOF normalisation), or None."""
    r = REPEAT_FRAC.get(site.lower())
    if r is None:
        return None
    p, n = r['frac'], r['n']
    return dict(frac=float(p), n=int(n), se=float((p * (1 - p) / n) ** 0.5))


def load_targets(site: str) -> dict:
    """Bundle of all calibration targets for a site."""
    return dict(
        site=site,
        ir_by_age=load_ir_targets(site),
        first_infection=load_first_infection_quartiles(site),       # events-only (process_model path)
        first_infection_km=load_first_infection_km(site),           # KM, censoring-aware (cohort path)
        repeat_frac=load_repeat_fraction(site),                     # repeat-detected fraction (cohort path)
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


def compute_person_months_steady_state(ages_years: np.ndarray,
                                        window_months: float) -> dict[str, float]:
    """Estimate person-months per MAL-ED age bin from a steady-state population.

    Snapshots the population age distribution at a single moment (typically sim
    end) and multiplies each bin's headcount by the calibration window length
    in months. Valid for a steady-state sim where the age distribution is
    stationary; would be inaccurate for a true cohort sim (use the dedicated
    cohort accounting instead).

    Args:
      ages_years: 1D array of agent ages (years), e.g. sim.people.age.values.
      window_months: length of the calibration window in months.
    """
    pt = {}
    for bin_label, (lo_m, hi_m) in MALED_AGE_MONTH_RANGE.items():
        lo_y, hi_y = lo_m / 12.0, hi_m / 12.0
        count = int(((ages_years >= lo_y) & (ages_years < hi_y)).sum())
        pt[bin_label] = float(count) * float(window_months)
    return pt


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
                  beta0: float = 0.0, beta1: float = 0.0, beta2: float = 0.0, beta3: float = 0.0,
                  p_symp_1: float = 1.0, p_symp_2: float = 1.0, p_symp_3plus: float = 0.0,
                  gamma_2: float = 0.0, gamma_3plus: float = 0.0,
                  reporting_rate: float | None = None,
                  censor_at_months: float = 36.0,
                  calibration_window: tuple[float, float] = (5.0, 10.0),
                  rng_seed: int | None = None,
                  p_asymp_detect: float = 0.4,
                  verbose: bool = False) -> dict:
    """Extract MAL-ED-comparable summary statistics from a sim's InfectedStrainStats output.

    Detection model (matches MAL-ED's stool-collection regime):
      - Symptomatic infections: detected 100% (diarrheal stool sampled near event time).
      - Asymptomatic infections: detected with prob `p_asymp_detect` (~0.3-0.5 for
        rotavirus, depending on shedding duration vs monthly collection schedule).
      - Each event's symptomatic/asymptomatic classification is sampled ONCE and
        used for both the IR comparison and the first-detection-age comparison
        (so the two GOF terms are internally consistent).
    """
    df = dat.copy()
    df = df[(df['CollectionTime'] >= calibration_window[0]) &
            (df['CollectionTime'] <  calibration_window[1])]

    # Per-agent infection order within the calibration window.
    df = df.sort_values(['id', 'CollectionTime'])
    df['n_infections'] = df.groupby('id').cumcount() + 1

    # Precise continuous age in months if the analyzer recorded it; fall back to
    # the fine-bin midpoint for older analyzer outputs that lack this column.
    if 'age_months_precise' in df.columns:
        df['age_months_est'] = df['age_months_precise'].astype(float)
    else:
        df['age_months_est'] = df['Age'].map(FINE_BIN_MIDPOINT_MONTHS)

    # --- Classify each event as symptomatic ONCE (used by both filters below) ---
    # Four symptom-probability families. The age families share the same logistic
    # age predictor (centered at 12 mo, capped at 60 mo, matching Lewnard et al. and
    # process_incidence_uk_age.calculate_symptom_probability); they differ only in
    # how prior-infection count enters:
    #   age_only / age_and_infection_simple : logit = age_poly                       (no infection term)
    #   age_and_infection                    : logit = age_poly + beta3 * min(n, 5)   (Lewnard: single linear slope)
    #   age_and_infection_offsets            : logit = age_poly + gamma_n             (categorical offsets; gamma_1 = 0)
    #   infection_number                     : P = per-infection probs p_symp_1/2/3+  (no age term)
    # age_only / age_and_infection (gamma=0) / age_and_infection_offsets all nest:
    # age_only is the offsets model with gamma_2 = gamma_3plus = 0.
    n_inf = df['n_infections'].values
    if symptom_model in ('age_only', 'age_and_infection_simple',
                         'age_and_infection', 'age_and_infection_offsets'):
        age_capped = np.minimum(df['age_months_est'].values, 60.0)
        ac = age_capped - 12.0
        lp = beta0 + beta1 * ac + beta2 * ac ** 2
        if symptom_model == 'age_and_infection':
            lp = lp + beta3 * np.minimum(n_inf, 5)
        elif symptom_model == 'age_and_infection_offsets':
            lp = lp + np.where(n_inf == 1, 0.0,
                               np.where(n_inf == 2, gamma_2, gamma_3plus))
        # Clip the logit before the logistic: wide priors (e.g. beta2 * age^2 with
        # age up to 60 mo) can send lp to +-1000s and overflow np.exp. The logistic
        # is already saturated (~0/1) well before +-30, so this changes nothing but
        # the numerics.
        lp = np.clip(lp, -30.0, 30.0)
        symp_probs = pd.Series(1.0 / (1.0 + np.exp(-lp)), index=df.index)
    elif symptom_model == 'infection_number':
        # Per-infection symptomatic probability (declining with successive infections),
        # the classic Pitzer/Lewnard structure. p_symp_1/2/3plus are the probabilities
        # that the 1st / 2nd / 3rd-or-later infection is symptomatic.
        n = df['n_infections']
        symp_probs = pd.Series(p_symp_3plus, index=df.index, dtype=float)
        symp_probs[n == 1] = p_symp_1
        symp_probs[n == 2] = p_symp_2
    else:
        raise ValueError(f"Unknown symptom_model: {symptom_model}")

    rng = np.random.default_rng(rng_seed)
    df['is_symptomatic'] = rng.random(len(df)) < symp_probs.values

    # Apply optional reporting filter on symptomatic events (kept for parity with
    # UK pipeline; for MAL-ED we set reporting_rate=1.0 so this is a no-op).
    if reporting_rate is not None and 'severity' in df.columns:
        rep_keep = rng.random(len(df)) < (reporting_rate * df['severity'].values)
        df.loc[df['is_symptomatic'] & ~rep_keep, 'is_symptomatic'] = False

    # --- Symptomatic IR by age bin (matches MAL-ED diarrheal-stool surveillance) ---
    df_symp_36 = df[df['is_symptomatic'] & (df['age_months_est'] <= censor_at_months)]
    ir = compute_model_ir_by_age(df_symp_36, person_months_by_bin)

    # --- First-DETECTED infection per agent (mirrors MAL-ED's monthly+diarrheal regime) ---
    # Symptomatic events always detected; asymptomatic events detected with p_asymp_detect.
    is_asymp = ~df['is_symptomatic']
    df['is_detected'] = df['is_symptomatic'] | (is_asymp & (rng.random(len(df)) < p_asymp_detect))
    df_detected_36 = df[df['is_detected'] & (df['age_months_est'] <= censor_at_months)].copy()
    df_detected_36 = df_detected_36.sort_values(['id', 'CollectionTime'])
    df_detected_36['n_detected'] = df_detected_36.groupby('id').cumcount() + 1
    first_detected = df_detected_36[df_detected_36['n_detected'] == 1]
    quartiles = _quartiles_from_ages(first_detected['age_months_est'].values, censor_at_months)

    if verbose:
        print(f"  symptomatic events <=36m: {len(df_symp_36)}")
        print(f"  detected events <=36m:    {len(df_detected_36)}")
        print(f"  first-detected per-agent: {len(first_detected)}")
        print(f"  IR by age bin: {ir['IR'].to_dict()}")
        print(f"  first-detected quartiles (months): {quartiles}")

    return dict(ir_by_age=ir, first_infection=quartiles)


def _quartiles_from_ages(ages: np.ndarray, censor_at_months: float | None) -> dict:
    """Compute Q25/median/Q75 of an array of ages, with optional age censoring."""
    ages = np.asarray(ages, dtype=float)
    ages = ages[~np.isnan(ages)]
    if censor_at_months is not None:
        ages = ages[ages <= censor_at_months]
    if len(ages) == 0:
        return dict(median=np.nan, q25=np.nan, q75=np.nan, n_events=0)
    q25, med, q75 = np.quantile(ages, [0.25, 0.5, 0.75])
    return dict(median=float(med), q25=float(q25), q75=float(q75), n_events=int(len(ages)))


# ---------------------------------------------------------------------------
# GOF
# ---------------------------------------------------------------------------
def gof_incidence(model_ir: pd.DataFrame, target_ir: pd.DataFrame) -> float:
    """Sum of squared log-IR differences across MAL-ED age bins."""
    m = np.log(model_ir['IR'].values + LOG_EPS)
    t = np.log(target_ir['IR'].values + LOG_EPS)
    return float(np.sum((m - t) ** 2))


def gof_incidence_poisson(model_ir: pd.DataFrame, target_ir: pd.DataFrame) -> float:
    """Poisson deviance of the observed (MAL-ED) age-bin case counts under the model's
    predicted rate. Pure per-bin Poisson MLE (no free weights): minimizing this is
    maximizing the per-bin Poisson likelihood. Unlike the squared-log-IR term it
    penalizes putting cases in the wrong age BIN (shape), not just per-bin level, and
    it weights the sparse 24-35mo bin (1 case) correctly instead of letting the
    log+LOG_EPS over-weight it.

    Expected count in bin k = (model IR_k / 100) * observed person-months_k; observed
    count = MAL-ED cases_k. Deviance = 2*sum_k[c_k*log(c_k/lam_k) - (c_k - lam_k)]
    (the c*log(c/lam) term is 0 where c_k=0). Deviance keeps the term on a chi-square
    scale (O(few) at a good fit), comparable to gof_first_infection, so the joint
    weighting carries over from the squared-log objective. Lower = better."""
    c = target_ir['cases'].values.astype(float)            # observed MAL-ED counts
    pt = target_ir['PT'].values.astype(float)              # observed person-months
    lam = np.maximum(model_ir['IR'].values / 100.0 * pt, 1e-9)  # model-expected counts
    dev = np.where(c > 0, c * np.log(np.where(c > 0, c, 1.0) / lam), 0.0)
    return float(2.0 * np.sum(dev + (lam - c)))


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


def gof_repeat(model_frac, target) -> float:
    """Repeat-detected-fraction GOF: z-score^2 vs the target, normalised by the target's
    binomial SE so a 1-SE miss = 1.0 (chi-square scale, comparable to the other terms).
    Returns 0 if no target or model value (term simply doesn't count)."""
    if target is None or model_frac is None or not np.isfinite(model_frac):
        return 0.0
    return float(((model_frac - target['frac']) / target['se']) ** 2)


def gof(model_out: dict, targets: dict,
        w_inc: float = 1.0, w_first: float = 1.0, w_repeat: float = 1.0,
        fit_target: str = 'joint') -> dict:
    """Combined GOF. `fit_target` selects which terms count:
      - 'joint':           w_inc * GOF_inc(squared-log) + w_first * GOF_first
      - 'symptomatic_ir':  GOF_inc (squared-log) only (w_first ignored)
      - 'first_infection': GOF_first only (w_inc ignored)
      - 'poisson':         w_inc * GOF_inc_poisson(deviance) + w_first * GOF_first
      - 'poisson_ir':      GOF_inc_poisson only (w_first ignored)
      - 'cohort':          w_inc * GOF_inc_poisson + w_first * GOF_first(KM) +
                           w_repeat * GOF_repeat  (cohort-emulation objective)
    For 'cohort', GOF_first compares the model's KM first-DETECTION quartiles to the
    censoring-aware KM target; otherwise to the events-only target. Per-component values
    are always returned for logging; `gof_incidence_active` is the incidence term in use.
    """
    g_inc = gof_incidence(model_out['ir_by_age'], targets['ir_by_age'])
    g_inc_pois = gof_incidence_poisson(model_out['ir_by_age'], targets['ir_by_age'])
    first_target = targets['first_infection_km'] if fit_target == 'cohort' else targets['first_infection']
    g_first = gof_first_infection(model_out['first_infection'], first_target)
    g_repeat = gof_repeat(model_out.get('repeat_frac'), targets.get('repeat_frac'))
    g_inc_active = g_inc
    if fit_target == 'symptomatic_ir':
        total = g_inc
    elif fit_target == 'first_infection':
        total = g_first
    elif fit_target == 'joint':
        total = w_inc * g_inc + w_first * g_first
    elif fit_target == 'poisson':
        g_inc_active = g_inc_pois
        total = w_inc * g_inc_pois + w_first * g_first
    elif fit_target == 'poisson_ir':
        g_inc_active = g_inc_pois
        total = g_inc_pois
    elif fit_target == 'cohort':
        g_inc_active = g_inc_pois
        total = w_inc * g_inc_pois + w_first * g_first + w_repeat * g_repeat
    else:
        raise ValueError(f"Unknown fit_target: {fit_target}")
    return dict(gof=total, gof_incidence=g_inc, gof_incidence_poisson=g_inc_pois,
                gof_incidence_active=g_inc_active, gof_first_infection=g_first,
                gof_repeat=g_repeat, w_inc=w_inc, w_first=w_first, w_repeat=w_repeat,
                fit_target=fit_target)


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
