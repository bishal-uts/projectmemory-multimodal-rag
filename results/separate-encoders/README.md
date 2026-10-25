# Separate-encoder baseline

Added on 24 October 2026, after the three-method MultiModalQA comparison had been run and
reported, on the same 100 development questions and 2,015 candidates. The encoders, the
fusion rule and k were fixed before anything was scored.

## What it is

The conventional multimodal RAG design: one encoder per modality, separate indexes, merged
results.

- **Text and tables** (997 passages, 98 serialized tables): EmbeddingGemma 2 text
  embeddings, identical to the dense-text baseline.
- **Images** (920): SigLIP 2, a dedicated image–text encoder
  (`google/siglip2-so400m-patch14-384`, revision `e8e48729`, Apache-2.0). The question is
  the text query, with no prompt template and SigLIP 2's 64-token limit (2 of 100 questions
  were truncated).
- **Merge**: reciprocal rank fusion with k = 60. Scores from two models are not comparable,
  so only ranks are used. With one list per modality this alternates between the lists, and
  ties go to the text list. Every top five therefore holds three text or table items and two
  images, the same effect as the fixed per-modality quotas common in multimodal RAG
  frameworks.
- **Answers**: the saved top-five rankings are replayed into the same local Gemma 4 E2B
  generator and settings as the other three runs.

The SigLIP 2 vectors were computed on the UTS Cetus HPC (CPU job, 16 threads, 7.8 minutes
for 920 images). Only the images and the question text were copied there, never answers or
relevance labels.

## Results

Retrieval, top five:

| Method | Recall@5 | All support | Hit rate | Right image found (42 questions) |
|---|---:|---:|---:|---:|
| BM25 | 62.1% | 50% | 75% | 21.4% |
| Dense text | 77.1% | 64% | 91% | 23.8% |
| Separate encoders | 72.3% | 61% | 84% | 29.0% |
| Native multimodal | **82.3%** | **71%** | **94%** | **43.3%** |

Recall@5 by whether a question needs an image:

| Method | Needs an image (42) | No image needed (58) |
|---|---:|---:|
| BM25 | 37.2% | 80.2% |
| Dense text | 50.3% | **96.6%** |
| Separate encoders | 47.1% | 90.5% |
| Native multimodal | **63.8%** | 95.7% |

Image encoders alone. This diagnostic ranks only the 920 images for the 42 questions with
labelled images, with no text items and no fusion:

| Image representation | Right image in top 5 | Mean reciprocal rank | Median rank of first right image |
|---|---:|---:|---:|
| Titles (EmbeddingGemma 2 text) | 42.9% | 0.33 | 9.5 |
| Pixels, SigLIP 2 | 48.0% | 0.37 | 6.5 |
| Pixels, EmbeddingGemma 2 (native) | **65.5%** | **0.55** | **2** |

Answers from the fixed local generator:

| Method | Exact match | Token F1 | Answered | Invalid outputs |
|---|---:|---:|---:|---:|
| BM25 | 14% | 20.7% | 63 | 9 |
| Dense text | 19% | 27.1% | 79 | 8 |
| Separate encoders | 16% | 22.4% | 74 | 16 |
| Native multimodal | **22%** | **30.0%** | **81** | **6** |

Paired exact-match differences (exploratory 95% percentile bootstrap, 10,000 resamples,
seed 42, not adjusted for multiple comparisons):

| Comparison | Difference | 95% interval |
|---|---:|---:|
| Separate encoders minus BM25 | +2 points | −5 to +9 |
| Separate encoders minus dense text | −3 points | −9 to +3 |
| Separate encoders minus native multimodal | −6 points | −13 to +1 |

## What this shows

- **Native single-space retrieval beat the separate-encoder pipeline** on every retrieval
  measure, and its answers were 6 points better. The answer interval includes zero, so the
  answer difference is not conclusive at this sample size.
- **The gain is not only the fusion rule.** As an image encoder alone, EmbeddingGemma 2
  placed the right image higher than SigLIP 2 for 20 of the 42 questions, lower for 12,
  with 10 ties.
- **A fixed modality mix costs text questions.** Separate indexes need a rule for how many
  images to include, and this one always adds two. That lowered recall on questions that
  need no image from 96.6% (dense text) to 90.5%. In one shared space the ranking decides:
  native retrieval put no image in the top five for 49 questions, 1.0 per question on
  average.
- **Twice as many invalid outputs** (16 against 6 to 9) occurred when every evidence set
  held two images. This was observed, not explained.

## Caveats

- **Added after the main comparison**, on development questions that had already been used.
  It is a supplementary baseline, not part of the original pre-specified comparison.
- **One encoder and one fusion rule.** A score-normalised or learned fusion, or a prompt
  template for SigLIP 2, might do better. None was tried, to avoid tuning on these
  questions.
- **Generator code version.** The original three runs used `local_generation.py` from
  before the video-only decoder repair (media protocol v1). This run used the repaired file
  (v2). The repair changed video frame sampling only, and MultiModalQA evidence never
  includes video. A control re-ran the first 20 native questions with the repaired file
  (`generator-control.json`): 19 answers were identical, including all 8 with image
  evidence. The one difference is a text-and-table question whose original answer had been
  reused from an older cached generation, so it is run-to-run sampling variation, not the
  code change. The analysis allows exactly this difference and requires every other setting
  and code hash to match.
- Repeating this run with the same code reproduced all 46 overlapping answers of an
  interrupted earlier attempt exactly. That attempt is kept outside the results, in
  `.cache/separate-encoders/aborted-answers-run-1`.

## Files

- `separate-encoders.json`: retrieval report, same schema as `../mmqa-comparison/`
- `comparison.md`: four-method retrieval table
- `retrieval-summary.json`: right-image recall, recall by image need, image-only diagnostics
- `siglip2-mmqa.npz`, `siglip2-mmqa.json`: SigLIP 2 vectors and their provenance record
- `answer-analysis.json`, `answers.md`: audited answer comparison, built on
  `scripts/analyze_final_results.py`
- `generator-control.json`: the 20-question generator-code control
- `../final-mmqa-local-separate-encoders/`: the answer run
- `../control-mmqa-local-multimodal-v2/`: the control run
- `retrieve.log`, `answers.log`, `control.log`: console logs

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/run_separate_encoders.py export
# Copy .cache/separate-encoders/bundle to the cluster as ~/projectmemory-siglip/data, with
# scripts/separate_encoders/siglip_embed.py and siglip_embed.pbs, then run: qsub siglip_embed.pbs
# Copy out/siglip2-mmqa.npz and out/siglip2-mmqa.json back into this folder.
.\.venv\Scripts\python.exe scripts/run_separate_encoders.py retrieve
.\.venv\Scripts\python.exe scripts/run_separate_encoders.py answers
.\.venv\Scripts\python.exe scripts/run_separate_encoders.py control
.\.venv\Scripts\python.exe scripts/run_separate_encoders.py analyze
```
