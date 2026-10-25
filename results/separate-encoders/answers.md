| Method | Exact match | Token F1 | Answered | Invalid outputs |
|---|---:|---:|---:|---:|
| BM25 | 14% | 20.68% | 63 | 9 |
| Dense text | 19% | 27.14% | 79 | 8 |
| Native multimodal | 22% | 30.00% | 81 | 6 |
| Separate encoders (SigLIP 2 + text, RRF) | 16% | 22.40% | 74 | 16 |

| Comparison | Difference | 95% interval |
|---|---:|---:|
| Separate encoders (SigLIP 2 + text, RRF) minus BM25 | +2 points | -5 to +9 |
| Separate encoders (SigLIP 2 + text, RRF) minus Dense text | -3 points | -9 to +3 |
| Separate encoders (SigLIP 2 + text, RRF) minus Native multimodal | -6 points | -13 to +1 |
