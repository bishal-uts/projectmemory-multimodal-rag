# ProjectMemory: cited answers across project materials

## Practical problem

Project decisions are scattered across meeting discussions, notes, slides,
whiteboards, and recordings. A student or team member may remember a decision
but not which source explains it. Filename or transcript searches can miss
information represented visually or acoustically.

ProjectMemory retrieves evidence across these materials and generates an answer
with source citations. AMI's public product-design meetings provide a realistic
project setting without requiring private recordings or manually written
questions. Public multimodal benchmarks provide complementary evaluation data.

## Application and user interaction

A user selects a collection or imports files, then asks a question such as
“What features should the remote control have?” The local browser app retrieves
sources, displays relevant text or media with location metadata, and uses selected evidence to generate a cited answer locally, with optional
hosted providers. Search works
without an API call. A CLI supports the same ingestion, retrieval, and evaluation
workflow.

Google **EmbeddingGemma 2** (`google/embeddinggemma-2`) provides native multimodal
embeddings. BM25 and dense text retrieval provide comparison baselines. The
primary answer model is **local Gemma 4 E2B QAT Q4_0**. NVIDIA Nemotron 3 Nano
Omni and Gemini remain optional API comparisons. Their availability and quotas
depend on the account. A separate generator is necessary because embeddings retrieve evidence
but do not compose answers.

File ingestion handles text, tables, HTML, PDF page text/images, DOCX/PPTX text
and embedded images, images, and physical audio/video clips with source times.
Office layouts are not rendered. Video clips are silent and audio is represented
separately, allowing explicit modality comparisons. Hosted generation processes
the selected evidence externally; the public datasets are the default experiment
material.

## Research questions and evaluation

The main question is: **When does native multimodal retrieval find supporting
evidence that lexical or semantic text retrieval misses?** Compare BM25, dense
text, and native multimodal retrieval on fixed candidates and queries. Report
Recall@5, Hit@5, MRR@5, nDCG@5, and complete-support retrieval. Keep the answer
model fixed when comparing retrieval methods so generation changes do not
confound the result.

| Dataset | Role | Evaluation |
|---|---|---|
| AMI/QMSum | Project-memory transcript retrieval and answers; 24 development and 22 held-out questions | Evidence retrieval; lexical answer token F1 and ROUGE-L |
| MultiModalQA | Text/table/image evidence; fixed 100-question development pilot | Evidence retrieval; custom answer-set precision/recall/exact match and token F1 |
| Daily-Omni | Audio/video reasoning; 100 prepared questions about 94 supplied clips | Choice accuracy and coverage for audio-only, silent-video-only, and combined inputs |

These datasets answer different questions and must not be pooled into one
headline score. Daily-Omni supplies the relevant clip, so its ablations evaluate
answering rather than corpus retrieval. The text baselines use image titles but
no OCR or generated captions; native image retrieval uses pixels without titles.
This evaluates the implemented representations rather than isolating a visual
encoder effect with identical metadata.

Begin hosted answer measurements with the first five questions in the fixed
prepared order. Record per-question predictions and failures, use matching
questions across conditions, and expand after reviewing outputs and quota.
Checkpointed runs resume only when data, configuration, and relevant code hashes
match. Citation ID validation is implemented; determining whether each source
actually supports a claim requires additional evaluation. Details are in the
[evaluation protocol](EVALUATION.md).

## Current delivery and remaining work

Public data preparation, media acquisition, native embedding support, local file
ingestion, the browser interface, local/API generation, and answer scorers are
implemented. All 920 selected MMQA images and all 188 selected Daily-Omni media
records are available. Real image/audio/video embedding checks passed. A real
AMI API answer and the browser search/cached-answer flow also passed integration
checks; these checks are not answer-quality results.

AMI development and held-out retrieval runs are complete. The held-out protocol
was frozen before scoring the 22 ES2011 questions. The full MMQA retrieval
comparison is complete. The expanded 100-question local MMQA answer comparison
and 20-question supplied-media AV ablations are also complete. Consult
[measured status](STATUS.md) and [the final comparison](../results/final-analysis/comparison.md)
for actual results, uncertainty and limitations. Earlier five-question API pilots
are measured separately.
AMI raw recording acquisition and transcript/media alignment remain separate
extension work.

The technical report, fixed-sample error analysis and demonstration guide are
prepared. The assessment report and presentation follow the supplied rubric and
timing and are submitted separately. Independent human citation-support
evaluation, a new untouched question sample and an OCR/caption baseline are
possible research extensions.

## Presentation outcome

Demonstrate a real user question, retrieved sources, a cited answer, and its
original evidence. Present fixed-data comparisons with sample sizes and dataset
scope. Explain both successes and failures, distinguish missing evidence from
incorrect generation, and identify exactly which modalities were measured.
Describe five-question runs as integration pilots, not robust estimates of
model quality or official benchmark results.
