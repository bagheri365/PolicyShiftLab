# Coat empirical benchmark

`experiments/12_coat_empirical_benchmark.py` is the first logged-vs-randomized
empirical evaluation.

Candidate probability estimators are fitted **only** from `train.ascii`.
`test.ascii` is never passed to model fitting and is used only as the randomized
evaluation benchmark. The binary event is `rating >= 4`.

The primary comparison reports:

- randomized-test Brier score;
- naive logged Brier score;
- Horvitz-Thompson-style Brier using the supplied learned propensities;
- self-normalized IPW (SNIPS) Brier.

Because `propensities.ascii` contains learned rather than oracle probabilities,
the weighted estimates are labeled estimated-propensity analyses. Propensity
floors at `0.01`, `0.02`, and `0.05` are reported only as sensitivity analyses.

The randomized test subset is interpreted as evaluation under Coat's
per-user randomized item-assignment design over the same users/items, not as a
generic sample representing every possible recommender deployment population.
