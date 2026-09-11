# Leakage-safe Coat evaluation

`experiments/13_coat_heldout_benchmark.py` separates model fitting from logged
evaluation.

For each user, a fixed-seed simple random sample of that user's logged ratings
is assigned to the logged-evaluation subset. The remaining logged ratings are
used for fitting. The randomized Coat test matrix remains evaluation-only.

If user `u` has `n_u` logged ratings and `h_u` are held out, then a logged
user-item pair has known conditional holdout probability

```text
q_u = h_u / n_u
```

given that the pair was logged. The estimated HT evaluation therefore uses

```text
P(held-out logged observation) = e_ui * q_u
```

where `e_ui` is the supplied learned Coat propensity.

This removes the training-set-reuse problem in the baseline empirical
experiment. It does **not** turn the supplied Coat propensities into oracle
probabilities: the weighted empirical results remain estimated-propensity
analyses.

Propensity-floor results are labeled `clipped-HT` / `clipped-SNIPS` and are
sensitivity diagnostics. In particular, clipped HT is not presented as an
unbiased estimator of the original target estimand.
