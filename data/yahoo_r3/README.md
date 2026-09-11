# Yahoo! R3 local data

PolicyShiftLab does not redistribute Yahoo! R3.

Two local layouts are supported.

Original Webscope-style filenames:

```text
data/yahoo_r3/ydata-ymusic-rating-study-v1_0-train.txt
data/yahoo_r3/ydata-ymusic-rating-study-v1_0-test.txt
```

Common research-copy filenames:

```text
data/yahoo_r3/user.txt
data/yahoo_r3/random.txt
```

Both whitespace-separated and comma-separated triplets are accepted. User/item
IDs may be zero- or one-based.

Run:

```bash
python experiments/16_yahoo_r3_support_audit.py
```

The audit checks user/item overlap, exact pair overlap, rating support, binary
outcome prevalence (`rating >= 4`), user activity, item popularity, and the
fraction of randomized observations lying inside logged user/item support.

`sampling_data.txt` is intentionally excluded from the PolicyShiftLab
replication because it is a derived/sample artifact rather than the primary
logged or randomized benchmark file.

The randomized subset is not automatically treated as a universal target
population; the replication must state exactly what assignment/evaluation
distribution it approximates.
