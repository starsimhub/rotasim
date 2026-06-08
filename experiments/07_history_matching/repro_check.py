"""Is the SIR non-repro a PLATFORM bug or just my param rounding? Test on capy.

The stored SIR stored params rounded to 6 decimals. We rebuilt inputs from those. Here we
re-run idx 6268 with (a) the rounded stored params and (b) the EXACT full-precision params
from the cached NROY draw (rw._untransform of the original row) -- same seed both. If (b)
reproduces stored but (a) doesn't, it's rounding on the knife-edge, NOT a starsim/platform bug.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, sciris as sc
HERE = Path(__file__).resolve().parent; REPO = HERE.parent.parent
rw = sc.importbypath(HERE / 'run_wave.py')
exp06 = sc.importbypath(REPO / 'experiments' / '06_titer_maternal_peak' / 'run.py')
fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
CENS = fi.loc[fi.event_observed == 0, 'age_event_months'].dropna().values; CENS = CENS[CENS > 0]

IDX = 6268
recs = {json.loads(l)['idx']: json.loads(l) for l in open(HERE / 'outputs' / 'sir_results.jsonl')}
r = recs[IDX]; seed = 20260605 + IDX
stored = [r['ir_symp_<6 m'], r['ir_symp_6-11 m'], r['ir_symp_12-23 m']]

# (a) rounded params straight from the stored record
p_round = {k[4:]: r[k] for k in r if k.startswith('par_')}
# (b) exact params from the cached NROY draw row IDX (what the SIR actually used)
nroy = pd.read_csv(HERE / 'outputs' / 'nroy_draw.csv')
p_exact = rw._untransform(nroy.iloc[IDX])

print(f'idx {IDX}, seed {seed}')
print('param max |exact-rounded| =', max(abs(p_exact[k] - p_round[k]) for k in p_exact))
out_a = exp06._run_one((IDX, p_round, 40_000, seed, CENS, 0.5))
out_b = exp06._run_one((IDX, p_exact, 40_000, seed, CENS, 0.5))
ir = lambda o: [o.get('ir_symp_<6 m'), o.get('ir_symp_6-11 m'), o.get('ir_symp_12-23 m')]
print(f'  stored           : {stored}')
print(f'  (a) rounded params: {ir(out_a)}')
print(f'  (b) EXACT params  : {ir(out_b)}')
print('  => exact reproduces stored?', all(abs((ir(out_b)[i] or -9) - stored[i]) < 1e-4 for i in range(3)))
