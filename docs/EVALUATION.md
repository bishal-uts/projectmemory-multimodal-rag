# Evaluation protocol

## Research question

Does using original visual evidence improve evidence retrieval over available
text and metadata, while preserving useful retrieval on ordinary text sources?
Answer generation is evaluated separately using local Gemma 4 E2B. NVIDIA
Nemotron 3 Nano Omni and Gemini provide saved API pilot comparisons. The
[expanded local protocol](FINAL_EXPERIMENT_PROTOCOL.md) specifies the complete
local comparisons and failure handling.

## Controlled retrieval comparison

| Method | Text/table inputs | Image inputs | Purpose |
|---|---|---|---|
| `bm25` | Text and serialized tables | Titles only | Lexical baseline |
| `dense-text` | EmbeddingGemma 2 text embeddings | Titles only | Semantic text baseline |
| `multimodal` | EmbeddingGemma 2 text embeddings | Original pixels | Native image experiment |

This first comparison has no OCR/caption enrichment. A later OCR/caption baseline
would distinguish the value of native embeddings from the value of simply adding
visual information. Keep corpus selection, question selection, k and generation
settings fixed when adding it. Completed differences are conditional on this
sampled pool and baseline design; they do not isolate native embeddings from
every alternative way of making visual information available.

Model revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`.
Pinned libraries: Sentence Transformers 6.1.0, Transformers 5.19.0.
Document text uses Google's `title: ... | text: ...` prefix. Queries use
`SearchQuery`; native media use their modality dictionaries without text prefixes.

An offline tokenizer check of the exact prefixed text inputs found maximum
lengths of 976 tokens for AMI and 1,406 for MMQA, including special tokens.
No prepared text input exceeds the model's 8,192-token context. Longer user
documents and recordings are segmented by the local ingestion pipeline; this
tokenizer check does not establish video/audio duration coverage.

## MultiModalQA sampled development experiment

The pilot selects 100 development questions with seed 42. It retains all supplied
contexts for those questions, including distractors, and searches their pooled
union: 2,015 source records. Tables are serialized as text. Gold support IDs come
from the official annotations, not from a judge model.

This is a **custom sampled development experiment**. Its candidate pool is
different from both per-question supplied contexts and the full published corpus.
Do not compare its scores directly to published leaderboard numbers. Use it for
pipeline development and within-project comparisons. Freeze settings before using
a separate untouched question sample for any future test-set claim; overlapping
source documents are acceptable, but report question overlap and corpus construction.
The completed 100-question local answer comparison extends this existing sample;
it does not convert it into an untouched test set.

The pilot has 28 text-only, 16 table-only, 18 image-only, 14 text/table,
20 image/table and 4 image/text questions, classified by annotated gold supports.
There are 159 support occurrences across the 100 questions. One question has
seven supports, so the maximum possible macro Recall@5 is 99.71% and maximum
All-support@5 is 99%.

Native image retrieval uses pixels; titles are available to text baselines.
Original image files are required. No image URL, title, synthetic image or caption
is silently substituted for a missing file.

## AMI/QMSum application evaluation

The development team is ES2006 (four meetings, 24 specific questions). The held-out
team is ES2011 (four meetings, 22 specific questions). Both teams' transcripts are
prepared, but development evaluation searches only the supplied meeting's chunks
for each query. A question such as “What did they decide?” assumes a known meeting.

Fixed consecutive 10-utterance chunks are created before consulting evidence
labels. A chunk is relevant if it overlaps an official inclusive utterance span.
Whole-meeting summary questions lack localized support labels and are saved for
later generation rather than assigned artificial retrieval gold.

These are derived **transcript chunk** labels. Long gold spans can cover more
than five chunks, placing a ceiling on Recall@5 and All-support@5. QMSum does not
establish image, audio or video retrieval quality, and utterance numbers are not
timestamps. Settings were frozen before the held-out team was scored.

The final transcript protocol was frozen on 15 October 2026 before scoring
ES2011: consecutive 10-utterance chunks, the supplied meeting as candidate scope,
k=5, default BM25 parameters, the pinned EmbeddingGemma 2 revision, official
Document/SearchQuery prompts and float32 embeddings. No choice of these settings
used ES2011 results. The held-out run is stored separately under
`results/ami-test-comparison`; any later tuning must not be presented as untouched
test performance.

In the prepared development subset, gold spans overlap an average of 10.67
chunks per question (maximum 38). Twelve of 24 questions have more than five
gold chunks. The resulting theoretical macro Recall@5 ceiling is 73.20%, and
the All-support@5 ceiling is 50%. This explains part of the difference in metric
scale between AMI and MultiModalQA; their scores should not be averaged together.

## Reported metrics

- Recall@k: fraction of all annotated supporting sources retrieved.
- Hit rate@k: fraction of questions with at least one relevant source retrieved.
- MRR@k: reciprocal rank of the first relevant source, or zero.
- nDCG@k: rank-sensitive gain using binary source relevance.
- All-support@k: fraction of questions retrieving every annotated support source.

Metrics are macro-averaged across questions. They measure evidence retrieval,
not whether a generated answer is correct. Annotation coverage may be incomplete:
an unlabelled retrieved item may still contain useful information.

Reports include per-question rankings, gold IDs, model settings, candidate scope,
index construction time, search latency and dataset fingerprints. Comparison
requires matching corpus evidence, questions and k. Native media hashes distinguish
changed files at the same path. Search latency excludes model initialization and
index construction; the latter is recorded separately. Hardware and cache state
must be stated when presenting speed comparisons.

## Daily-Omni answer evaluation

The seeded 100-question sample uses 94 clips. The original task supplies the clip
and four choices, so it has no corpus-retrieval relevance labels. `evaluate`
rejects it; `evaluate-answers` supplies its known media directly to the generator,
and `score-choices` can also measure externally produced choices.

All 188 selected media files are local. Original video files contained audio;
full-length silent copies preserve frames and duration for the visual condition.
The historical API pilots used the same first five supplied clips/questions.
They do not establish complete provider or modality comparisons.

Only two of the 100 questions fit the local model's strict full-audio limit.
The separate local experiment therefore applies the
[frozen normalization protocol](FINAL_EXPERIMENT_PROTOCOL.md): audio no longer
than 30.001 seconds and silent video no longer than 60 seconds are eligible, with
terminal PCM samples beyond 30 seconds removed and both source/derived hashes
recorded. Seed 42 selects 20 questions from 52 eligible questions, covering 16
clips. Each selected audio loses only three 16 kHz samples (0.1875 ms); originals
and complete silent videos are preserved. This is explicitly a boundary-normalized
subset, not an untouched full-clip benchmark.

The same 20 IDs and choices were evaluated under identical generator settings
and repaired timestamp-based video-decoder protocol version 2:

| Supplied media and report | Choice accuracy | Declared coverage | Invalid responses |
|---|---:|---:|---:|
| [Audio only](../results/final-local-av-audio/report.json) | 10% (2/20) | 10% | 10 |
| [Silent video only](../results/final-local-av-video/report.json) | 15% (3/20) | 45% | 4 |
| [Audio + silent video](../results/final-local-av-combined/report.json) | 10% (2/20) | 60% | 5 |

Valid abstentions number 8, 7 and 3 respectively; these and invalid outputs count
as incorrect. All observed accuracies are below both the 25% expected accuracy
of uniform guessing and the sample's 25% always-A baseline. Combined media has
not demonstrated an accuracy advantage. This descriptive result does not claim
statistical significance. Pre-repair outputs are archived and excluded. The
[paired analysis](../results/final-analysis/comparison.md) reports categories and
uncertainty estimates accounting for shared clips. One exact duplicate exists
in the official source; occurrence IDs preserve its original weighting. Further
deduplication or filtering must not silently change the scoring denominator.

Application queries pair matching audio/video segments and select whole groups
within duration/frame/byte budgets, displaying additions and omissions. Controlled
experiments bypass this convenience policy: MMQA retains its exact top-five
rankings and Daily-Omni receives its prescribed supplied media. Generation still
enforces strict limits and never silently truncates an evaluation input.

## Leakage and reproducibility

Index source evidence only. Questions, reference answers, intermediate answers,
gold support labels and answer choices remain outside `corpus.jsonl`. Evaluation
passes only question text and any supplied meeting scope to the retriever; it
never passes gold support IDs. Source text can naturally contain an answer.

Record sampling seed, selected IDs, source checksums, model commit, chunk size,
embedding precision and k. Do not change these between compared runs. Re-run
baselines after changing prepared evidence or the question set. Report actual
results and failed runs separately; a dependency or missing-media failure is not
a zero-scoring completed experiment.

## Generated answers and citations

Use the same generator, question IDs, evidence count and decoding settings when
comparing retrievers. The completed local comparison uses Gemma 4 E2B QAT Q4_0,
llama.cpp b11429, temperature 1.0, top-p 0.95, top-k 64, seed 42, 4,096 output
tokens, disabled thinking and one formatting retry. All three runs use the same
fixed 100 MMQA IDs and k=5. Their exact-match scores are 14%/19%/22%, declared
coverage 63%/79%/81%, and invalid-response counts 9/8/6 for BM25/dense text/native
multimodal retrieval, respectively. Links and full metrics are in
[generation results](GENERATION_RESULTS.md); interpretation is in the
[technical report](TECHNICAL_REPORT.md).

The five-question NVIDIA, Gemini and local pilots remain historical integration
results. Provider-specific sampling/reasoning settings differ, and their scores
must not be ranked against the 100-question local runs. Record exact model
versions and implementation fingerprints; fixed seeds do not guarantee identical
GPU or hosted outputs.

MMQA reference lists contain all required answer entities, not interchangeable
aliases. Report normalized answer-set exact match, precision and recall, plus
token F1. AMI uses token F1 and ROUGE-L F1 against its reference prose; these
lexical metrics do not establish factual correctness. Daily-Omni uses choice
accuracy, with abstentions counted as incorrect, and answer coverage. Declared
coverage is the fraction with `abstained=false`; the model can still produce a
semantic non-answer or incomplete answer under that flag.

Citation gold precision/recall measure overlap with labelled supporting sources.
They do not prove that each generated claim follows from its citation. The
[first-ten error analysis](ERROR_ANALYSIS.md) inspects a predefined sample for
retrieval misses, omissions, incorrect reasoning and unsupported claims. It is
explicitly AI-assisted qualitative inspection, not independent human annotation
or an entailment-accuracy estimate. Daily-Omni has no retrieval gold, so citation
gold overlap is omitted for that task.

The generator receives only the question, supplied choices when applicable, and
retrieved/supplied evidence. Its structured answer must cite valid evidence IDs.
The expanded experiment's opt-in failure policy records only the specified local
grounded-answer validation failure after its existing retry as an invalid model
response, unanswered and incorrect. All other failures, including startup,
transport, filesystem and abnormal completion errors, stop the run while
preserving completed predictions. The policy and its implementation hash are
recorded in the generator configuration; it was fixed after an interrupted pilot
and before the complete comparisons.

Run manifests include evidence hashes, selected questions, retrieval settings,
model/output settings and generation/retrieval/evaluation implementation hashes.
Each completed prediction is checkpointed for an identical-run resume. Validated
identical-input generation cache hits are reused, not independent new inferences.
Mixed cached/warm/cold timings are not controlled latency measurements, and GPU
readings taken after requests are not peak memory. Public benchmark contamination
remains a limitation; this is not a test of unseen pretraining data.
