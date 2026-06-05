"""
Exp 07 — History matching wave(s) on MAL-ED Bangladesh (historymatching v2 API).

- 16 box-bounded params; log-transform base_beta + titer_median; monotone ladders
  (sus_after_*, p_symp_*) reparameterized as ratios so the constraint is a box.
- Simulator runs the exp-06 cohort model (1 seed/point, no replicates); EXTINCT
  sims -> NaN (treated as model failures, dropped from emulator training).
- Targets: (mean, std) per observable; std = observational (Poisson/binomial) (+)
  small model SD from the replicate check. Auto feature selection (1/wave, cooldown).
- Output (all waves) inside this experiment folder; checkpoints for spot-resume.

Usage (one wave at a time, inspect between):
  uv run --python 3.13 python experiments/07_history_matching/run_wave.py --max-iter 1
  uv run --python 3.13 python experiments/07_history_matching/run_wave.py --max-iter 2 --resume
  uv run --python 3.13 python experiments/07_history_matching/run_wave.py --smoke
"""
import os, argparse
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc
import historymatching as hm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
exp06 = sc.importbypath(REPO / 'experiments' / '06_titer_maternal_peak' / 'run.py')

N_AGENTS = 40_000
CROSS = 0.5
_fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
CENS = _fi.loc[_fi['event_observed'] == 0, 'age_event_months'].dropna().values
CENS = CENS[CENS > 0]
N_WORKERS = int(os.environ.get('HM_WORKERS', '118'))

# Box bounds (log where noted; ratios for monotone ladders).
BOUNDS = {
    'log_base_beta':        (np.log(0.10), np.log(0.6)),
    'young_reservoir':      (1.0, 15.0),
    'adult_contacts':       (0.5, 2.5),
    'infant_exposure':      (0.3, 4.0),
    'sus_after_1':          (0.1, 1.0),
    'sus_r2':               (0.0, 1.0),   # sus_2 = sus_1 * r2
    'sus_r3':               (0.0, 1.0),   # sus_3 = sus_2 * r3
    'log_titer_median':     (np.log(4.0), np.log(60.0)),
    'titer_gsd':            (1.3, 3.5),
    'titer_half_life_days': (25.0, 70.0),
    'hill_slope':           (1.5, 8.0),
    'maternal_efficacy':    (0.7, 0.99),
    'p_symp_1':             (0.4, 1.0),
    'p_r2':                 (0.0, 1.0),   # p_symp_2 = p_symp_1 * r2
    'p_r3':                 (0.0, 1.0),   # p_symp_3 = p_symp_2 * r3
}

# Targets: (mean, std). std = sqrt(observational^2 + model^2).
# Observational: IR ~ Poisson on cases (SD=IR/sqrt(cases); cases <6m=27,6-11m=74,12-23m=59);
# fractions ~ binomial SE; model SD from replicate_variance (persisting).
OBSERVATIONS = {
    'ir_symp_<6 m':         (1.91, np.hypot(1.91/np.sqrt(27), 0.121)),
    'ir_symp_6-11 m':       (5.37, np.hypot(5.37/np.sqrt(74), 0.198)),
    'ir_symp_12-23 m':      (2.35, np.hypot(2.35/np.sqrt(59), 0.145)),
    'repeat_detected_frac': (0.43, np.hypot(np.sqrt(0.43*0.57/136), 0.016)),
    'frac_ever_detected':   (0.638, np.hypot(np.sqrt(0.638*0.362/213), 0.009)),
    'first_inf_median':     (7.98, np.hypot(0.528, 0.108)),  # bootstrap median SE (+) model SD
}
OBS_COLS = list(OBSERVATIONS)


def _untransform(row):
    s1 = float(row['sus_after_1']); s2 = s1 * float(row['sus_r2']); s3 = s2 * float(row['sus_r3'])
    p1 = float(row['p_symp_1']);     p2 = p1 * float(row['p_r2']);   p3 = p2 * float(row['p_r3'])
    return dict(base_beta=float(np.exp(row['log_base_beta'])),
                young_reservoir=float(row['young_reservoir']), adult_contacts=float(row['adult_contacts']),
                infant_exposure=float(row['infant_exposure']),
                sus_after_1=s1, sus_after_2=s2, sus_after_3plus=s3,
                maternal_efficacy=float(row['maternal_efficacy']),
                titer_median=float(np.exp(row['log_titer_median'])), titer_gsd=float(row['titer_gsd']),
                titer_half_life_days=float(row['titer_half_life_days']), hill_slope=float(row['hill_slope']),
                p_symp_1=p1, p_symp_2=p2, p_symp_3plus=p3)


def simulate(params_df: pd.DataFrame) -> pd.DataFrame:
    """HM simulator: params DataFrame -> observables DataFrame (NaN where extinct)."""
    tasks = []
    for i, (_, row) in enumerate(params_df.iterrows()):
        p = _untransform(row)
        seed = abs(hash(tuple(np.round(row.values, 6)))) % (2**31)
        tasks.append((i, p, N_AGENTS, int(seed), CENS, CROSS))
    with get_context('spawn').Pool(processes=min(N_WORKERS, len(tasks))) as pool:
        recs = pool.map(exp06._run_one, tasks)  # ordered
    rows = []
    for rec in recs:
        if (not rec.get('ok')) or rec.get('frac_ever_infected', 0) < 0.05:
            rows.append({c: np.nan for c in OBS_COLS})        # extinct/failed -> NaN
            continue
        kt = np.asarray(rec['km_time']); ko = np.asarray(rec['km_observed'])
        fim = float(np.median(kt[ko == 1])) if (ko == 1).any() else np.nan
        rows.append({'ir_symp_<6 m': rec['ir_symp_<6 m'], 'ir_symp_6-11 m': rec['ir_symp_6-11 m'],
                     'ir_symp_12-23 m': rec['ir_symp_12-23 m'], 'repeat_detected_frac': rec['repeat_detected_frac'],
                     'frac_ever_detected': rec['frac_ever_detected'], 'first_inf_median': fim})
    return pd.DataFrame(rows, index=params_df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-samples', type=int, default=2000)
    ap.add_argument('--max-iter', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.n_samples = 24
        global N_AGENTS; N_AGENTS = 8000

    engine = hm.HistoryMatching(
        function=simulate, bounds=BOUNDS, observations=OBSERVATIONS,
        emulator_type='bayes_linear', sampling_strategy='lhs',
        feature_selection=hm.AutoFeatureSelection(method='mean_sq_z', max_features=1, cooldown_period=2),
        n_samples=args.n_samples, implausibility_threshold=3.0, max_iterations=args.max_iter,
        output_dir=str(HERE / 'outputs' / 'hm'), run_name='maled_bd', random_seed=20260605,
    )
    t0 = sc.tic()
    results = engine.run(resume=args.resume)
    print(f'\nHM run done in {sc.toc(t0, output=True):.0f}s')
    try:
        print(engine.get_status_summary())
    except Exception as e:
        print('status summary unavailable:', e)


if __name__ == '__main__':
    main()
