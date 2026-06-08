"""
Exp 05 — β/contact range-check pilot.

The MixingPools FOI scale differs from RandomNet; the smoke showed a sharp
extinction<->saturation threshold. This grid sweeps (base_beta, infant_exposure,
young_reservoir) at fixed immunity/symptom params to locate the regime where the
toddler reservoir sustains an endemic epidemic while infants/repeats stay low.
Goal: pick the prior ranges for the full run.

Usage: uv run python experiments/05_structured_mixing_cohort/pilot.py --n-workers 60
"""
import os, json, argparse, itertools
from pathlib import Path
from multiprocessing import get_context
import numpy as np, sciris as sc, pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
run_mod = sc.importbypath(HERE / 'run.py')

# Fixed (plausible) non-swept params: strong-ish immunity, sharp maternal, peaked severity.
FIXED = dict(sus_after_1=0.9, sus_after_2=0.5, sus_after_3plus=0.3,
             maternal_immunity_efficacy=0.9, maternal_mean_duration_days=200.0,
             p_symp_1=1.0, p_symp_2=0.3, p_symp_3plus=0.1)

GRID_BETA = [0.06, 0.09, 0.12, 0.16, 0.22, 0.30]
GRID_INFEXP = [1.0, 4.0, 10.0]
GRID_RESERVOIR = [15.0, 30.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-agents', type=int, default=40_000)
    ap.add_argument('--n-workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
    cens = fi.loc[fi['event_observed'] == 0, 'age_event_months'].dropna().values
    cens = cens[cens > 0]

    tasks = []
    for i, (b, ie, yr) in enumerate(itertools.product(GRID_BETA, GRID_INFEXP, GRID_RESERVOIR)):
        p = dict(FIXED, base_beta=b, infant_exposure=ie, young_reservoir=yr)
        tasks.append((i, p, args.n_agents, args.seed + i, cens, 0.5))

    nw = args.n_workers or os.cpu_count()
    print(f'Pilot: {len(tasks)} grid cells, {args.n_agents} agents, {nw} workers', flush=True)
    out = HERE / 'outputs' / 'pilot.jsonl'
    if out.exists(): out.unlink()
    t0 = sc.tic()
    ctx = get_context('spawn')
    recs = []
    with ctx.Pool(processes=nw) as pool:
        for rec in pool.imap_unordered(run_mod._run_one, tasks):
            with out.open('a') as f: f.write(json.dumps(rec) + '\n')
            recs.append(rec)
    print(f'Done {len(recs)} in {sc.toc(t0, output=True):.0f}s\n')

    # Summarize the landscape.
    print(f'{"beta":>5} {"infExp":>6} {"resv":>5} | {"ever_inf":>8} {"repeat":>6} {"med_1st":>7} | '
          f'{"ir_symp <6/6-11/12-23":>22} peak')
    for r in sorted([x for x in recs if x['ok']], key=lambda r: (r['par_base_beta'], r['par_infant_exposure'], r['par_young_reservoir'])):
        ir = [r['ir_symp_<6 m'], r['ir_symp_6-11 m'], r['ir_symp_12-23 m']]
        peak = ['<6', '6-11', '12-23'][int(np.argmax(ir))] if max(ir) > 0 else '-'
        print(f"{r['par_base_beta']:>5.2f} {r['par_infant_exposure']:>6.1f} {r['par_young_reservoir']:>5.0f} | "
              f"{r['frac_ever_infected']:>8.2f} {r['repeat_detected_frac']:>6.2f} {r['true_first_median']:>7.1f} | "
              f"{ir[0]:>6.1f}{ir[1]:>7.1f}{ir[2]:>8.1f}   {peak}")
    print('\nTarget: ever_inf<~0.6, repeat~0.10, med_1st~8mo, peak at 6-11mo')


if __name__ == '__main__':
    main()
