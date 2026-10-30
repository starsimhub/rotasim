# Model description

Rotasim is an agent-based model with a daily timestep. Each rotavirus strain is a separate disease in the simulation, so a person can carry several at once. Two connectors link the strains: one sets each person's susceptibility to each strain, and the other handles reassortment.

The default parameter values are starting points for analysis. They have not been calibrated to any setting.

## Strains

A strain is a combination of a G genotype and a P genotype, written like G1P8. The G and P genotypes are the two outer capsid proteins that the immune system mainly responds to. The other genome segments are not modeled.

A scenario lists the strains present at the start. The sim creates every combination of their G and P genotypes, because any of these could arise by reassortment. Strains not in the scenario start with no infections. With `use_preferred_partners=True`, only the common pairings in `rs.PREFERRED_PARTNERS` are created, along with the scenario's own strains.

A strain's transmission rate per day is `base_beta` times its fitness.

## Course of infection

A person infected with a strain is symptomatic and then asymptomatic. Their total duration of infection is drawn from a lognormal distribution with a mean of 13 days. The last 8 days are asymptomatic, with shedding at 10% of the symptomatic level. Rotavirus causes no deaths in the model.

After recovering, a person is susceptible again, with their susceptibility reduced by immunity.

Symptoms don't change transmission beyond the reduced shedding of the second phase. Whether an infection is clinically symptomatic is decided afterwards by the observation models (see below).

## Immunity

`rs.RotaImmunityConnector` sets each person's relative susceptibility to each strain at each step, as one minus their protection. Protection is the larger of acquired and maternal immunity.

### Acquired immunity

Acquired protection against a strain is an efficacy times a factor.

The **efficacy** depends on the closest match among the strains the person has recovered from:

| Match | Meaning | Default efficacy |
|---|---|---|
| Homotypic | The same G and P | 1.0 |
| Partial heterotypic | The same G or the same P | 0.5 |
| Complete heterotypic | Neither shared | 0.3 |
| Naive | No earlier infection | 0.0 |

The **factor** is the largest of:

- **Baseline protection**, which depends on the number of infection episodes the person has completed, and doesn't wane. By default it is `1 - exp(-0.1 * episodes)`. With `use_fixed_susceptibility=True` it is set directly, as a susceptibility after one, two, and three or more episodes (0.67, 0.50, and 0.36 by default). Infections that overlap count as one episode.
- **Waning immunity** from the strain itself, for a homotypic match, or from strains sharing G or P, for a partial match. It starts at 1 on recovery, and decays exponentially, with a mean duration drawn per person from a normal distribution (mean 91 days, standard deviation 14 days).

So someone who has recovered from a strain is fully protected against it at first, and that protection falls towards the baseline over the following months.

### Maternal immunity

Maternal immunity depends on age, and protects against all strains equally. It is off by default, and is turned on by setting `maternal_immunity_efficacy`. There are two versions, set by `maternal_immunity_model`:

- `'erlang'`: everyone of the same age has the same protection, which is the efficacy times the survival function of an Erlang distribution. With `maternal_immunity_n_stages=1` this is exponential decay, and more stages give a plateau followed by a sharper drop, as with a maternal class split into several compartments. A second, slower component can be added, for example for breastfeeding, and the two combine as independent sources of protection.
- `'titer'`: each infant has an initial antibody titer, drawn from a lognormal distribution, which decays at a common rate and gives protection through a Hill curve. The peak in cases then comes from the spread in titers, rather than from an imposed shape.

## Reassortment

`rs.RotaReassortmentConnector` looks at people carrying two or more strains. Each day, each of them reassorts with probability `reassortment_prob` (0.05 by default). They are then infected with one strain, chosen at random from the G,P combinations of the strains they carry that they don't already have. For example, someone with G1P8 and G2P4 can gain G1P4 or G2P8.

This infection is not reduced by their immunity, since the virus is already present in them.

## Vaccination

`rs.RotaVaccination` offers infants between 6 weeks and 1 year of age each dose once, 28 days apart by default, and they take up each dose with a set probability. Protection against a strain is the effectiveness of the latest dose (0.6 and 0.8 for two doses by default), times a relative efficacy for the match between the strain and the vaccine (1.0, 0.6, or 0.3, using the same categories as above). It wanes exponentially, with a mean duration of a year by default. A person's susceptibility is the lower of what their immunity and the vaccine would give.

## Observation models

An infection in the model is not the same as a case seen in a study. `rs.SymptomModel` gives the probability that an infection is symptomatic, depending on infection number, age, or both. `rs.MALEDTargets`, `rs.MALEDCohort`, and `rs.Surveillance` use it to work out what a study would see. These only observe the simulation, so they have no effect on transmission.

## Parameters

### Each strain (`rs.Rotavirus`)

| Parameter | Default | Meaning |
|---|---|---|
| `beta` | 0.16 per day | Transmission rate; in `rs.Sim`, `base_beta` times fitness |
| `init_prevalence` | 0.04 | Share of people infected at the start; in `rs.Sim`, from the scenario |
| `init_age_dist` | All ages | How those infections are spread by age; in `rs.Sim`, mostly in young children |
| `dur_inf` | Lognormal, mean 13 days | Total duration of infection |
| `dur_asymptomatic` | 8 days | Duration of the asymptomatic phase |
| `asymptomatic_shedding_rate` | 0.1 | Relative infectiousness of the asymptomatic phase |
| `waning_rate_dist` | Normal, mean 91 days, standard deviation 14 days | Mean duration of immunity, per person |
| `waning_delay` | 0 days | Time after recovery before immunity wanes |

### Immunity (`rs.RotaImmunityConnector`)

| Parameter | Default | Meaning |
|---|---|---|
| `homotypic_immunity_efficacy` | 1.0 | Efficacy for the same G and P |
| `partial_heterotypic_immunity_efficacy` | 0.5 | Efficacy for the same G or P |
| `complete_heterotypic_immunity_efficacy` | 0.3 | Efficacy for neither shared |
| `naive_immunity_efficacy` | 0.0 | Efficacy with no earlier infection |
| `baseline_immunity_exponential_rate` | 0.1 | How quickly baseline protection builds up |
| `use_fixed_susceptibility` | False | Whether to use the three values below instead |
| `sus_after_1`, `sus_after_2`, `sus_after_3plus` | 0.67, 0.50, 0.36 | Susceptibility by the number of episodes |
| `maternal_immunity_model` | `'erlang'` | How maternal immunity wanes |
| `maternal_immunity_efficacy` | 0.0 | Protection at birth |
| `maternal_immunity_half_life` | 90 days | Sets the mean duration, if that isn't given |
| `maternal_immunity_n_stages` | 1 | Stages of Erlang waning |
| `maternal_immunity_mean_duration` | None | Mean duration of Erlang waning |
| `maternal_immunity_efficacy_slow` | 0.0 | Efficacy of the slower component |
| `maternal_immunity_half_life_slow` | 270 days | Half-life of the slower component |
| `maternal_titer_median` | 8.0 | Median initial titer, relative to the titer that gives half protection |
| `maternal_titer_gsd` | 2.0 | Geometric standard deviation of initial titers |
| `maternal_titer_half_life` | 35 days | Half-life of the titer |
| `maternal_hill_slope` | 3.0 | Slope of the curve from titer to protection |
