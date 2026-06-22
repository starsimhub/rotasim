"""Quick comparison: per-child risk-by-age, icddr,b Dhaka surveillance vs MAL-ED cohort vs a
single MAL-ED-fitted model draw. NOT a calibration -- a consistency/validation check. The model
series is one infnum draw pushed through the Surveillance observer (Bangladesh demographics,
icddr,b bins); replace with the full-posterior overlay when available."""
import numpy as np, pathlib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import process_surveillance_icddrb as IC, process_incidence_maled as P
labels = ['<6m', '6-11m', '12-23m', '24-59m*']
ic = IC.load_targets_icddrb()['ir_proxy']                                  # cases/width (uniform-per-yr denom)
mt = P.load_targets('bangladesh')['ir_by_age']['IR'].to_numpy()            # cohort symptomatic IR (last bin 24-35)
model_ir = np.array([50.5, 105.8, 54.4, 26.4])                             # single MAL-ED infnum draw, Surveillance obs
norm = lambda v: np.asarray(v, float) / np.asarray(v, float)[1]            # to the 6-11m peak
series = {'icddr,b surveillance (cases/width)': norm(ic),
          'MAL-ED cohort (symptomatic IR)': norm(mt),
          'model: MAL-ED infnum, 1 draw': norm(model_ir)}
x = np.arange(4); w = 0.26; cols = ['#c0392b', '#000000', '#2c7fb8']
fig, ax = plt.subplots(figsize=(9, 5))
for j, (lab, v) in enumerate(series.items()):
    ax.bar(x + (j-1)*w, v, w, label=lab, color=cols[j], alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('relative incidence (normalized to 6-11m peak)')
ax.set_title('Per-child risk by age: icddr,b surveillance vs MAL-ED cohort vs model\n'
             'All peak at 6-11m; model over-predicts the older tail (severity selection, not modeled)')
ax.legend(frameon=False, fontsize=9); ax.axhline(0, color='gray', lw=.5)
ax.annotate('*MAL-ED last bin = 24-35m', xy=(0.99, -0.13), xycoords='axes fraction', ha='right', fontsize=7, color='gray')
fig.tight_layout()
fig.savefig(pathlib.Path(__file__).parent / 'figures' / 'icddrb_vs_cohort_relative_incidence.png', dpi=140)
