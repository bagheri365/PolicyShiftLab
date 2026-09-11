# Yahoo! R3 leakage-safe benchmark

`experiments/17_yahoo_r3_heldout_benchmark.py` mirrors the leakage-safe Coat
design without importing an unjustified propensity model.

The target user population is restricted to the users appearing in Yahoo! R3's
randomized evaluation file. Logged ratings from the other logged-only users are
excluded from the target comparison.

Within the randomized-user cohort:

1. logged ratings are split within user into fit and held-out evaluation rows;
2. fixed candidate probability estimators are fit only on logged-fit rows;
3. the same fitted models are evaluated on held-out logged ratings and on the
   randomized benchmark;
4. Brier-score gaps and model-ranking agreement are reported.

The candidate family is the same simple fixed family used for Coat:

```text
global
user_smoothed
item_smoothed
user_item_blend
```

The binary event is `rating >= 4`.

No propensity weighting is included in this first Yahoo! R3 replication. A
propensity model should only be added after its assumptions and estimation
procedure are explicitly justified.
