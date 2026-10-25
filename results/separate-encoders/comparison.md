# Retrieval comparison

Corpus: 2015 items. Questions: 100. k=5.

| Method | Recall | Hit rate | MRR | nDCG | All support | Mean search (ms) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.6214 | 0.7500 | 0.6165 | 0.5667 | 0.5000 | 1.87 |
| dense-text | 0.7714 | 0.9100 | 0.8195 | 0.7411 | 0.6400 | 489.55 |
| multimodal | 0.8229 | 0.9400 | 0.8160 | 0.7627 | 0.7100 | 155.28 |
| separate-encoders | 0.7229 | 0.8400 | 0.7448 | 0.6718 | 0.6100 | 380.68 |

Search latency excludes model loading and index construction. This evaluates retrieval, not generated answers.
Do not compare this custom subset directly with published full-benchmark leaderboards.
