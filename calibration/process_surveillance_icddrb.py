"""icddr,b Dhaka surveillance targets: age distribution of MEDICALLY-ATTENDED rotavirus cases,
pooled 2010-2014 (matched to the MAL-ED Bangladesh window -> same place + time + population).

This is the same-country surveillance counterpart to MAL-ED (cohort) and to the UK (surveillance):
  - MAL-ED Dhaka cohort  : actively-surveilled symptomatic cases, <2y (process_incidence_maled)
  - icddr,b Dhaka        : MEDICALLY-ATTENDED (severe) cases, cross-sectional surveillance  <-- here
  - UK                   : medically-attended surveillance, low-FOI (process_surveillance_uk)

IMPORTANT (population structure): these are POPULATION-level surveillance counts, so each age bin
has a different population denominator. icddr,b denominators by age are not available; the project
assumes a UNIFORM under-5 age structure (2.5%/yr -> <1y 2.5%, 1-2y 2.5%, 2-5y 7.5%, >=5y 87.5%;
see bangladesh_age_data.csv) -- the SAME assumption the ABM uses (its standing structure is
uniform-per-year, 12-23/6-11 pop ratio ~2). So raw case PROPORTIONS conflate per-child risk with
bin width; compare on incidence rates (cases / uniform-per-year denominator) OR ensure the model's
(matching) structure is used. The medically-attended SEVERITY filter (severe ~ young / first
infection) is why almost no cases appear >=24mo here -- model that as an age/order detection filter.

Bins: <6 / 6-11 / 12-23 / 24-59 mo (24-59 pooled; very few cases >2y -> not worth splitting).
"""
import numpy as np

# Dhaka medically-attended cases, pooled 2010-2014 (A. Kraay, icddr,b).
ICDDRB_DHAKA = {
    'counts':       np.array([84, 322, 463, 24], float),
    'bin_labels':   ['<6 m', '6-11 m', '12-23 m', '24-59 m'],
    'feature_keys': ['prop_0_6', 'prop_6_11', 'prop_12_23', 'prop_24_59'],
    'bin_edges_m':  (0.0, 6.0, 12.0, 24.0, 60.0),    # 4 bins, cap at 60mo
    'cap_age_m':    60.0,
    # uniform-per-year population weight per bin (= bin width; the assumed denominator structure)
    'pop_weight_mo': np.array([6.0, 6.0, 12.0, 36.0]),
}


def load_targets_icddrb(site='dhaka'):
    """icddr,b surveillance target. Returns case proportions + multinomial SEs AND a crude
    incidence proxy (cases / uniform-per-year population weight) that divides out bin width."""
    if site != 'dhaka':
        raise ValueError("only 'dhaka' (2010-2014, MAL-ED-matched) is loaded; Matlab pending")
    d = ICDDRB_DHAKA
    counts = d['counts']; total = int(counts.sum())
    prop = counts / total
    se = np.sqrt(prop * (1.0 - prop) / total)
    # incidence proxy under the uniform-per-year assumption: cases / (population ~ bin width).
    ir_proxy = counts / d['pop_weight_mo']
    ir_proxy_rel = ir_proxy / ir_proxy.max()           # relative per-child risk by age (peaks 6-11m)
    return dict(counts=counts, total=total, proportions=prop, se=se,
                ir_proxy=ir_proxy, ir_proxy_rel=ir_proxy_rel,
                bin_labels=d['bin_labels'], feature_keys=d['feature_keys'],
                bin_edges_m=d['bin_edges_m'], cap_age_m=d['cap_age_m'])


if __name__ == '__main__':
    t = load_targets_icddrb()
    print(f"icddr,b Dhaka medically-attended, 2010-2014: N={t['total']}")
    print(f"{'bin':8s}{'n':>6s}{'case_prop':>11s}{'(se)':>8s}{'~rel risk':>11s}")
    for lab, c, p, s, r in zip(t['bin_labels'], t['counts'], t['proportions'], t['se'], t['ir_proxy_rel']):
        print(f"  {lab:8s}{int(c):6d}{p:11.3f}{s:8.3f}{r:11.2f}")
    print("  (case_prop peaks 12-23m by COUNT; per-child risk peaks 6-11m -- 12-23m is 2x wider)")
