# Repeated Yahoo! R3 holdouts

`experiments/18_yahoo_r3_repeated_holdouts.py` repeats the leakage-safe
within-user logged fit/evaluation split over 200 deterministic seeds, always
restricting the logged data to the 5,400 users who appear in randomized
evaluation.

It reports:

- mean signed, mean absolute, and RMSE Brier gap between held-out logged and
  randomized evaluation;
- mean/median Spearman and Kendall agreement;
- any-reversal frequency;
- mean reversal count;
- exact-order recovery;
- the specific `user_smoothed` versus `user_item_blend` reversal frequency.

These repeated splits measure **split sensitivity conditional on the observed
Yahoo! R3 dataset**. They are not independent dataset replications.

The experiment also computes a paired user-cluster bootstrap for the randomized
top-pair comparison `user_item_blend` versus `item_smoothed`, using the fixed
seed-2026 logged-fit split. The percentile interval is descriptive uncertainty,
not a formal hypothesis test.
