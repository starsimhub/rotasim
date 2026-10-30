# Rotasim on Starsim v4: work plan

Working tree only, on the `v4` branch; nothing is committed.

## Scope

Port the model as it stands on `origin/dec_calibration_akraay` (2026-08) to Starsim 4, not the older `main`. The branch is merged, so its `calibration/` experiments are in the tree, unported.

## Definition of done

1. `import rotasim` runs on Starsim 4 with no Starsim 3 names; `pyproject.toml` and CI point at Starsim 4.
2. The branch's model features are all there: asymptomatic phase, immunity (waning, fixed-susceptibility and saturating baselines), maternal immunity (Erlang and titer), reassortment, vaccination, age-assortative network, analyzers.
3. Known bugs fixed and listed in `CHANGELOG.md`.
4. Results compared with the branch on Starsim 3 over 10 seeds, with differences explained.
5. A small test suite replaces the old one, and passes.
6. Minimal docs: README, model description, getting-started guide, changelog. Every code example runs.
7. Stale files removed; `CLAUDE.md` is current.
8. What Starsim 4 could do better for this model is written up in `starsim_issues.md`.

## Progress: 100% done

- [x] Baseline: branch on Starsim 3, 10 seeds, two cases
- [x] Branch model copied into the working tree
- [x] Scenarios and `rs.Sim`
- [x] `Rotavirus`
- [x] Immunity connector
- [x] Reassortment connector
- [x] Age-assortative network; remove `Aging`
- [x] Vaccination
- [x] Analyzers
- [x] Comparison with the baseline
- [x] Tests
- [x] Docs
- [x] Clean up stale files; packaging and CI
- [x] `starsim_issues.md`
