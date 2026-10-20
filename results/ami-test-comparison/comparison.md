# Retrieval comparison

Corpus: 425 items. Questions: 22. k=5.

| Method | Recall | Hit rate | MRR | nDCG | All support | Mean search (ms) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.3711 | 0.9545 | 0.8333 | 0.5624 | 0.0455 | 0.13 |
| dense-text | 0.4815 | 0.9545 | 0.8447 | 0.6471 | 0.1818 | 133.82 |

Search latency excludes model loading and index construction. This evaluates retrieval, not generated answers.
Do not compare this custom subset directly with published full-benchmark leaderboards.
