# Retrieval comparison

Corpus: 2015 items. Questions: 100. k=5.

| Method | Recall | Hit rate | MRR | nDCG | All support | Mean search (ms) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.6214 | 0.7500 | 0.6165 | 0.5667 | 0.5000 | 2.09 |

Search latency excludes model loading and index construction. This evaluates retrieval, not generated answers.
Do not compare this custom subset directly with published full-benchmark leaderboards.
