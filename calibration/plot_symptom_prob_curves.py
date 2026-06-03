"""
Plot the fitted P(symptomatic | age) logistic curves for the two MAL-ED
Bangladesh fits (best symptomatic-IR trial #19 and best first-infection
trial #17), overlaid on the prior UK fits for comparison.

All curves use the same functional form (process_incidence_uk_age.
calculate_symptom_probability, symptom_model='age_only'):
    P = logistic(beta0 + beta1*(a-12) + beta2*(a-12)^2),  a = min(age_mo, 60)
so they are directly comparable regardless of site (reporting_rate /
base_beta differ between sites but do not enter this curve).

Also writes a parameter summary table to symptom_fit_params.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from process_incidence_uk_age import calculate_symptom_probability

# (label, beta0, beta1, beta2, color, linestyle, linewidth)
# Lewnard et al. (PLoS Comput Biol; Mexico City + Vellore, India): published
# betas are in age-YEARS centered at 1 yr. This code's logistic is centered in
# MONTHS, so we rescale to the month form: y = (age_mo-12)/12 ->
#   beta1_mo = beta1_yr/12,  beta2_mo = beta2_yr/144,  beta0 unchanged.
FITS = [
    ('MAL-ED symptomatic-IR fit (#19)', -0.8628, -0.2039, -0.0062, '#1f77b4', '-',  2.4),
    ('MAL-ED first-infection fit (#17)',  1.1258, -0.2726, -0.0643, '#d62728', '-',  2.4),
    ('Lewnard et al. (Mexico/India)',    -1.05, -0.16/12, -0.54/144, '#2ca02c', '-.', 2.2),
    ('UK hybrid (Trial 49)',             -0.9843,  0.2582, -0.0085, '#555555', '--', 1.8),
    ('UK age+infection (simple)',        -1.4215,  0.1710, -0.0045, '#999999', ':',  1.6),
    ('UK age-only',                      -2.5165, -0.1118, -0.0360, '#bbbbbb', ':',  1.6),
]


def curve(beta0, beta1, beta2, ages_months):
    return np.array([
        calculate_symptom_probability(age_months=a, n_infections=1,
                                      symptom_model='age_only',
                                      beta0=beta0, beta1=beta1, beta2=beta2)
        for a in ages_months])


def main():
    ages = np.linspace(0, 60, 241)  # 0-5 years (model caps age at 60 months)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, b0, b1, b2, color, ls, lw in FITS:
        ax.plot(ages, curve(b0, b1, b2, ages), label=label,
                color=color, ls=ls, lw=lw)

    # MAL-ED age-bin boundaries for orientation.
    for x in (6, 12, 24, 36):
        ax.axvline(x, color='k', alpha=0.06, lw=1)
    ax.axvline(12, color='k', alpha=0.18, lw=1)  # logistic centering point

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Age at infection (months)')
    ax.set_ylabel('P(symptomatic | infected)')
    ax.set_title('Fitted age-symptom probability curves\n'
                 'MAL-ED Bangladesh fits vs. prior UK fits (logistic centered at 12 mo)')
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = 'symptom_prob_curves.png'
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')

    # Parameter table: betas + P at a few reference ages.
    ref_ages = [3, 6, 12, 24, 36, 60]
    rows = []
    for label, b0, b1, b2, *_ in FITS:
        p = curve(b0, b1, b2, ref_ages)
        rows.append(dict(fit=label, beta0=b0, beta1=b1, beta2=b2,
                         **{f'P@{a}mo': round(float(pi), 3) for a, pi in zip(ref_ages, p)}))
    df = pd.DataFrame(rows)
    df.to_csv('symptom_fit_params.csv', index=False)
    print('wrote symptom_fit_params.csv')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
