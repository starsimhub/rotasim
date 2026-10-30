"""Exp 37 — UK HM (infnum): VE + case-shape joint calibration; Bangladesh-constrained immunity.

Identifiability problem from exp28/36: shape-only calibration is degenerate —
high-beta/high-sus_after_1 and low-beta/lower-sus_after_1 produce the same case age
distribution, but very different VE (~10% vs ~74%). This adds VE as a live HM target,
which directly distinguishes the two regimes.

Changes from exp36:
  - Replaced first_inf_median_months (structurally unreachable, exp36 diagnosis) with
    ve_overall (observed UK Rotarix VE ~74% from UK test-negative design studies).
  - Replaced sero_2y/sero_5to6y (IgA waning makes 20U/ml threshold non-comparable to
    model "ever infected") with ve_overall as FOI anchor.
  - Two sims per parameter set: no-vaccine run (case shape vs pre-vaccine UK data)
    + vaccine run (VE vs UK Rotarix ~74%).
  - sus_after bounds widened from ±25% to ±50% of Bangladesh posterior medians (exp27).
    Bangladesh medians (exp27 corrected maternal): sus_after_1=0.751, sus_r2=0.615, sus_r3=0.659.
  - init_prevalence_override=0.005 + flat age init (fixes extinction in low-beta region).
  - 40k agents.

VE target rationale: UK Rotarix 2-dose (2+4mo) test-negative VE from published studies
  is ~74-77%. In the ABM, VE is computed test-negative-analog: within-sim cases among
  fully vaccinated vs zero-dose infants. With Bangladesh sus_after (~0.75) VE emerges
  ~74%; with exp28's artifact sus_after_1=0.975, VE is ~10%.

Run on zebra (160 cores):
  cd /home/akraay/rotasim/rotasim/calibration
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \\
    python experiments/37_uk_seroprev_hm/run.py --n-samples 2000 --max-iter 3
"""
import os, sys, pathlib
from multiprocessing import get_context
import numpy as np
import pandas as pd
import sciris as sc
import historymatching as hm

THISDIR = pathlib.Path(__file__).resolve().parent.parent.parent   # calibration/
sys.path.insert(0, str(THISDIR))
import calibrate_maled as cm
import process_surveillance_uk as PU
from hm_calibrate import (untransform, CycleFeatureSelection, SYMPTOM_MODEL,
                          FIXED_TITER_SHAPE)
from hm_calibrate_uk import build_sim_config, MODEL_SD_PROP, DEFAULT_CAP_M

OUT_DIR = pathlib.Path(__file__).resolve().parent / 'outputs' / 'hm'
N_AGENTS = 40_000
CAL_WINDOW = (5.0, 10.0)
N_WORKERS = int(os.environ.get('HM_WORKERS', '150'))

# Bangladesh posterior medians (exp27, infnum corrected maternal), ±50% bounds.
_BD = dict(sus_after_1=0.751, sus_r2=0.615, sus_r3=0.659)
CONSTRAINED_SUS_BOUNDS = {k: (round(max(v * 0.50, 0.05), 4), round(min(v * 1.50, 0.99), 4))
                          for k, v in _BD.items()}

# UK Rotarix 2-dose (2+4mo) test-negative VE target.
VE_OBS = {'ve_overall': (0.74, 0.05)}

# Vaccine configuration fixed at UK programme values.
_VACCINE_CONFIG = dict(response_prob=0.85, coverage=0.90,
                       dose_ages_y=[2.0/12.0, 4.0/12.0], moa='infection_blocking')


def exp37_bounds(cap_age_m=DEFAULT_CAP_M, beta_max=0.35):
    """Parameter bounds: sus_after constrained to Bangladesh exp27 ±50%; VE replaces fim."""
    b = {
        'log_base_beta':     (float(np.log(0.05)), float(np.log(beta_max))),
        **CONSTRAINED_SUS_BOUNDS,
        'maternal_efficacy': (0.50, 0.99),
        'p_symp_1':          (0.20, 1.0),
        'p_r2':              (0.0,  1.0),
        'p_r3':              (0.0,  1.0),
    }
    return b


def obs_cols(cap_age_m=DEFAULT_CAP_M):
    shape_keys = list(PU.load_targets_uk(cap_age_m=cap_age_m)['feature_keys'])
    return shape_keys + list(VE_OBS.keys())


def make_observations(cap_age_m=DEFAULT_CAP_M):
    """Shape targets (multinomial SE + model noise floor) + VE target."""
    t = PU.load_targets_uk(cap_age_m=cap_age_m)
    obs = {}
    for key, p, se in zip(t['feature_keys'], t['proportions'], t['se']):
        obs[key] = (float(p), float(np.hypot(se, MODEL_SD_PROP)))
    obs.update(VE_OBS)
    return obs


def make_simulator(sim_config):
    SHAPE_COLS = list(PU.load_targets_uk(cap_age_m=sim_config.get('cap_age_m', DEFAULT_CAP_M))['feature_keys'])
    ALL_COLS = SHAPE_COLS + ['ve_overall']

    # Two sim configs: same base, one with vaccine for VE computation.
    sc_vax = dict(sim_config)
    sc_vax['vaccine'] = _VACCINE_CONFIG

    def simulate(params_df: pd.DataFrame) -> pd.DataFrame:
        args = []
        for _, row in params_df.iterrows():
            sp = untransform(row, 'infnum', 'titer', fix_titer_shape=True)
            seed = abs(hash(tuple(np.round(row.values, 6)))) % (2**31)
            # Pair: (novax, vax) — same seed so only vaccine arm differs.
            args.append((sim_config, sp, int(seed),     CAL_WINDOW))
            args.append((sc_vax,    sp, int(seed)+1000, CAL_WINDOW))
        ctx = get_context('spawn')
        with ctx.Pool(processes=min(N_WORKERS, len(args)), maxtasksperchild=4) as pool:
            outs = pool.map(cm._run_one_replicate, args)
        rows = []
        n = len(params_df)
        for i in range(n):
            mo_nv = outs[2*i]       # no-vaccine: shape
            mo_vx = outs[2*i + 1]   # vaccine:    VE
            try:
                if mo_nv.get('total_cases', 0) <= 0:
                    rows.append({c: np.nan for c in ALL_COLS}); continue
                prop = mo_nv['case_proportions']
                row_out = {k: float(prop[j]) for j, k in enumerate(SHAPE_COLS)}
                # Overall VE: pooled test-negative-analog across all age bins.
                cv = np.asarray(mo_vx.get('cases_vax',   [0]*len(SHAPE_COLS)), float)
                cu = np.asarray(mo_vx.get('cases_unvax', [0]*len(SHAPE_COLS)), float)
                pv = np.asarray(mo_vx.get('py_vax',      [0]*len(SHAPE_COLS)), float)
                pu = np.asarray(mo_vx.get('py_unvax',    [0]*len(SHAPE_COLS)), float)
                tpv, tpu = pv.sum(), pu.sum()
                if tpv > 0 and tpu > 0 and cu.sum() > 0 and mo_vx.get('total_cases', 0) > 0:
                    row_out['ve_overall'] = float(1.0 - (cv.sum() / tpv) / (cu.sum() / tpu))
                else:
                    row_out['ve_overall'] = float('nan')
                rows.append(row_out)
            except Exception:
                rows.append({c: np.nan for c in ALL_COLS})
        return pd.DataFrame(rows, index=params_df.index)

    return simulate


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-samples', type=int, default=2000)
    ap.add_argument('--max-iter',  type=int, default=3)
    ap.add_argument('--resume',    action='store_true')
    ap.add_argument('--n-agents',  type=int, default=N_AGENTS)
    ap.add_argument('--beta-max',  type=float, default=0.35)
    ap.add_argument('--cap-age-months', type=float, default=DEFAULT_CAP_M)
    ap.add_argument('--smoke',     action='store_true')
    a = ap.parse_args()

    if a.smoke:
        a.n_samples = 20; a.max_iter = 1; a.n_agents = 8_000

    cap = a.cap_age_months
    sim_config = build_sim_config('infnum', a.n_agents, maternal='titer', cap_age_m=cap)
    sim_config['early_stop_extinct']       = True
    sim_config['early_stop_burn_in_years'] = 2.0
    sim_config['init_prevalence_override'] = 0.005
    sim_config['init_age_dist_override']   = [(0, 80, 1.0)]

    obs    = make_observations(cap)
    bounds = exp37_bounds(cap, a.beta_max)
    cols   = obs_cols(cap)

    print(f"Exp 37 — UK HM infnum: VE + case-shape")
    print(f"  n_agents={a.n_agents}  n_samples={a.n_samples}  max_iter={a.max_iter}  workers={N_WORKERS}")
    print(f"  sus_after bounds (±50% Bangladesh exp27): {CONSTRAINED_SUS_BOUNDS}")
    print(f"  vaccine: {_VACCINE_CONFIG}")
    print(f"  targets ({len(obs)}): {list(obs.keys())}")
    print(f"  obs: {[(k, round(v[0],3), round(v[1],3)) for k,v in obs.items()]}")

    engine = hm.HistoryMatching(
        function=make_simulator(sim_config),
        bounds=bounds,
        observations=obs,
        emulator_type='bayes_linear',
        sampling_strategy='lhs',
        feature_selection=CycleFeatureSelection(cols),
        n_samples=a.n_samples,
        implausibility_threshold=3.0,
        max_iterations=a.max_iter,
        output_dir=str(OUT_DIR),
        run_name='uk_infnum_ve_anchor',
        random_seed=20260707,
    )
    t0 = sc.tic()
    engine.run(resume=a.resume)
    print(f'\nExp 37 done in {sc.toc(t0, output=True):.0f}s')
    try:
        print(engine.get_status_summary())
    except Exception as e:
        print('status summary unavailable:', e)


if __name__ == '__main__':
    main()
