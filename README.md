# PolicyShiftLab

PolicyShiftLab is a controlled study of how policy-induced selective observation
can distort offline probabilistic evaluation and model selection in recommender
systems.

The core question is:

> When an existing recommendation policy determines which labels are observed,
> how different can logged-data evaluation be from evaluation on a target
> population?

## Scope

The main identifiable regime assumes a stable conditional outcome mechanism:

```text
P_log(Y | X) = P_target(Y | X)
```

while the covariate distribution may differ:

```text
P_log(X) != P_target(X)
```

The initial implementation focuses on evaluation, not weighted model training.

## Planned experiments

1. Synthetic recommender benchmark with known outcome and exposure mechanisms.
2. Logged-vs-target probability evaluation.
3. Post-hoc calibration transfer.
4. Model-ranking stability under selective observation.
5. Oracle vs estimated propensity weighting and overlap stress tests.

## Primary metrics

- Brier score
- Log loss
- Reliability diagrams
- Expected calibration error as a descriptive diagnostic

## Development

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Status

This repository is being built incrementally. The first milestone is a
reproducible metric and evaluation layer before adding simulation and dataset
pipelines.
