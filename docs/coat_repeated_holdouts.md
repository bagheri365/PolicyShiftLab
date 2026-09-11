# Repeated Coat holdouts and randomized uncertainty

`experiments/14_coat_repeated_holdouts.py` repeats the leakage-safe logged
fit/evaluation split over 200 deterministic seeds.

For each split it compares naive held-out logged evaluation,
estimated-propensity HT, and estimated-propensity SNIPS against the randomized
Coat benchmark. It reports Brier error, rank correlations, reversal frequency,
exact-order recovery, and the specific `user_smoothed` versus `item_smoothed`
reversal frequency.

These repeated splits measure **split sensitivity conditional on the observed
Coat dataset**. They are not independent dataset replications.

The experiment also computes a paired user-cluster bootstrap for the randomized
Brier difference between `user_smoothed` and `item_smoothed` using the fixed
logged-fit split with seed 2026. Users are resampled as clusters, carrying all
randomized observations for a sampled user together.

The percentile interval is descriptive uncertainty, not a formal hypothesis
test. The randomized test matrix remains evaluation-only throughout.
