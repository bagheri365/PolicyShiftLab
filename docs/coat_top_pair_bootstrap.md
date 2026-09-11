# Coat top-pair randomized uncertainty

`experiments/15_coat_top_pair_bootstrap.py` applies the existing paired
user-cluster bootstrap to the two best candidate models from the fixed
seed-2026 leakage-safe Coat benchmark:

```text
user_item_blend
user_smoothed
```

The reported quantity is

```text
Brier(user_item_blend) - Brier(user_smoothed)
```

so negative values favor `user_item_blend`.

The experiment uses 5,000 bootstrap replicates and resamples users as clusters,
carrying all randomized observations for each sampled user together. Model
predictions remain fixed from the logged-fit split with seed 2026; randomized
ratings remain evaluation-only.

The percentile interval is a descriptive uncertainty summary, not a formal
hypothesis test and not a claim that the candidate set contains a uniquely best
model.
