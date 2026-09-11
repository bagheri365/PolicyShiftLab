# Coat data audit

PolicyShiftLab does not redistribute the Coat dataset.

Download the original Coat data from the source used in the literature and
place the two dense rating matrices here:

```text
data/coat/train.ascii
data/coat/test.ascii
```

The original files encode observed explicit ratings as positive integers and
unobserved user-item pairs as zero. Existing public recommender-system code
using Coat commonly interprets `train.ascii` as the biased/logged matrix and
`test.ascii` as the randomized evaluation matrix.

Run:

```bash
python experiments/10_coat_support_audit.py
```

The audit intentionally runs **before** empirical model evaluation. It checks:

- user and item support in both matrices;
- exact user-item pair overlap;
- fraction of randomized observations whose user and item both have logged
  support;
- rating-value support;
- prevalence of the binary event `rating >= 4`;
- user activity and item popularity ranges;
- whether the randomized set has a constant number of observed items per
  active user;
- the overall randomized observation fraction.

A randomized test subset is not automatically treated as a universal target
population. The empirical README/result must state precisely what target
distribution it approximates and restrict claims to regions with adequate
support.
