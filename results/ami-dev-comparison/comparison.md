# Retrieval comparison

Corpus: 425 items. Questions: 24. k=5.

| Method | Recall | Hit rate | MRR | nDCG | All support | Mean search (ms) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.2498 | 0.7917 | 0.5708 | 0.3720 | 0.0833 | 0.25 |
| dense-text | 0.3889 | 0.9583 | 0.7340 | 0.5253 | 0.0833 | 302.17 |

Search latency excludes model loading and index construction. This evaluates retrieval, not generated answers.
Do not compare this custom subset directly with published full-benchmark leaderboards.
