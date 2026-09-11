# PolicyShiftLab

PolicyShiftLab is a controlled empirical study of **policy-induced selective
observation in offline recommender-system evaluation**.

The central question is:

> When an existing recommendation policy determines which labels are observed,
> how much can offline probabilistic evaluation and model selection differ from
> evaluation on the target population?

The project focuses on evaluation rather than propensity-weighted model
training.

## Identification regime

The main synthetic benchmark studies selective observation / covariate shift
under the identifying condition

```text
Y ⟂ S | X
```

equivalently,

```text
P_log(Y | X) = P_target(Y | X)
```

while the evaluated population changes:

```text
P_log(X) != P_target(X)
```

Here `S` is the observation/exposure indicator. Under this regime, the
conditional outcome mechanism is stable; this is **not concept drift**.

When propensities are known and overlap holds, PolicyShiftLab uses the
Horvitz-Thompson finite-population estimator for target Brier risk:

```text
(1 / N) * sum_i S_i * loss_i / e_i
```

with `e_i = P(S_i = 1 | X_i)`.

The estimator can be unbiased over repeated selection draws while still having
higher finite-sample variance and RMSE than naive logged evaluation.

## What is implemented

The synthetic benchmark currently includes:

- a latent-factor recommender outcome model with known target probabilities;
- aligned and popularity-distorted logging policies;
- naive logged, propensity-stratified, and oracle-IPW Brier evaluation;
- repeated-selection Monte Carlo bias, variance, RMSE, and coverage analysis;
- strong-to-weak overlap stress tests with IPW effective sample size;
- target, logged, and IPW reliability curves with shared target-derived bins;
- pointwise bootstrap uncertainty bands for reliability curves;
- fixed candidate-model ranking comparisons using Brier score;
- Spearman, Kendall, pairwise reversal, and exact-order recovery diagnostics;
- Monte Carlo ranking-stability analysis across repeated exposure draws;
- overlap-by-ranking-stability sweeps;
- controlled propensity-misspecification experiments;
- a hidden-selection failure case where the identifying assumption is violated.

## Main synthetic findings

The current experiments illustrate four recurring patterns. Exact values are
configuration- and seed-specific; the claims are about observed patterns, not
universal numerical guarantees.

**1. IPW trades bias for variance.** Oracle IPW can substantially reduce bias
without minimizing finite-sample RMSE. Under weak overlap, inverse weights
become unstable and effective sample size falls sharply.

**2. Selective observation can change model choice.** In
`experiments/06_ranking_monte_carlo.py`, naive logged evaluation produced at
least one pairwise ranking reversal in every replication for the current
configuration, with mean Spearman agreement of `0.520`. Oracle IPW improved
mean Spearman agreement to `0.932`, but still had reversals in `0.640` of
replications. Correction improved ranking stability without eliminating
finite-sample model-selection error.

**3. Weak overlap hurts both naive and corrected selection.** In
`experiments/07_overlap_ranking_sweep.py`, expected IPW effective sample size
fell from about `6559` under strong overlap to about `1511` under weak overlap.
Over the same sweep, the naive reversal rate rose from `0.030` to `0.703`, while
the oracle-IPW reversal rate rose from `0.290` to `0.653`. IPW is therefore not
uniformly better on every finite-sample ranking metric.

**4. Identification matters more than weighting machinery.** In
`experiments/09_hidden_selection_failure.py`, a hidden variable affects both
outcome risk and exposure. Full-propensity IPW using `P(S=1 | X, Z)` was nearly
unbiased in the repeated-selection experiment, while weighting with the exact
observed-only propensity `P(S=1 | X)` retained material bias because
`Y ⟂ S | X` no longer holds.

## Calibration diagnostics

Reliability diagrams use one set of score bins derived from the target
prediction distribution and reuse those same intervals for target, logged, and
IPW curves.

![Target, logged, and oracle-IPW reliability curves](figures/target_logged_ipw_reliability.png)

Expected calibration error (ECE) is treated as a **descriptive secondary
diagnostic** because it depends on binning and can be noisy under inverse
weighting. Brier score remains the primary probabilistic evaluation metric.

A conditionally correct predictor can remain correct under pure covariate shift
even when aggregate calibration summaries differ because the evaluated
population composition changes.

## What this project does not claim

PolicyShiftLab does **not** claim that:

- selective observation intrinsically makes a conditionally correct model
  miscalibrated;
- IPW must outperform naive evaluation in every finite sample;
- low ECE is sufficient evidence of good probability estimation;
- propensity correction solves hidden-confounding or unobserved-selection
  problems;
- a ranking reversal must occur for every policy, dataset, or candidate set;
- explicit-rating benchmarks such as Coat or Yahoo! R3 are realistic proxies
  for all modern recommendation products.

The synthetic experiments are designed to expose assumptions and failure modes,
not to tune the data-generating process until a preferred estimator "wins."

## Experiments

Run the synthetic experiments individually:

```bash
python experiments/01_logged_vs_target.py
python experiments/02_monte_carlo_eval.py
python experiments/03_overlap_stress_test.py
python experiments/04_calibration_curves.py
python experiments/05_model_ranking.py
python experiments/06_ranking_monte_carlo.py
python experiments/07_overlap_ranking_sweep.py
python experiments/08_propensity_misspecification.py
python experiments/09_hidden_selection_failure.py
```

## Development

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest
```

## Empirical next step

The next phase is a **support and composition audit of the Coat dataset** before
using its randomized subset as a target-evaluation benchmark. The audit will
check user/item overlap, user-item support, rating support, activity and item
popularity, outcome prevalence, missingness, and propensity support.

Only after that audit will the empirical evaluation define precisely what the
randomized subset approximates and compare logged-data model evaluation against
that target benchmark. Yahoo! R3 is planned as a replication dataset.
