# Verified setup status — 15 October 2026

ProjectMemory now has local file ingestion, a browser application, retrieval,
source citations, local/API answer generation, and separate retrieval/answer
scoring. **Gemma 4 E2B QAT Q4_0 is the default**, installed with llama.cpp b11429.
All five local input checks (text, image, audio transcription, video frames,
combined audio/video) passed on the RTX 4060 Laptop GPU. Its five-question MMQA
BM25 answer pilot completed with 40% exact match; the expanded 100-question
local comparison now scores 14% / 19% / 22% exact match with BM25 / dense text /
native multimodal retrieval respectively; see the [local model guide](LOCAL_MODEL.md)
for measured results and limitations. Both NVIDIA
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` and `gemini-3.8-flash` remain
configured; choose a provider in the app or with `--provider`. NVIDIA lists a
[free prototype endpoint](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning),
with account-dependent availability and quotas.

A real public AMI transcript question completed through NVIDIA with valid
citation IDs. Browser search and the cached-answer flow also passed a manual
check. Real image, audio, silent-video and combined requests also returned valid
responses. Gemini text, image and audio requests also completed, including
five-question MMQA answer pilots with all three retrieval methods. Repeated
HTTP 503 responses interrupted NVIDIA video/combined and Gemini combined runs;
partial runs have no full-condition quality score. See
[answer pilot results](GENERATION_RESULTS.md).

The expanded local AV comparison is also complete: **10% audio, 15% silent-video,
and 10% combined accuracy** on the same 20 questions about 16 clips. All are below
the measured always-A baseline and uniform-guessing expectation of 25%. Invalid
outputs were 10, 4 and 5, respectively. This is supplied-media QA on a documented
duration-filtered, boundary-normalized subset, not AV retrieval evaluation.
[Final analysis](../results/final-analysis/comparison.md) includes paired
uncertainty estimates, subgroup results and audited prediction metrics.

## Data available locally

| Collection | Questions | Searchable/prepared evidence | Media status |
|---|---:|---|---|
| AMI/QMSum | 24 dev + 22 held-out test | 425 transcript chunks from 8 meetings | Raw audio/video locations recorded; recordings not downloaded or aligned |
| MultiModalQA pilot | 100 dev | 997 passages, 98 tables, 920 original images | All 920 images downloaded and verified |
| Daily-Omni pilot | 100 | 94 clips represented by 188 separate audio/video records | All originals present; visual records use verified full-duration silent copies |

MMQA image acquisition transferred **42,633,057 bytes**, including ZIP metadata,
from the 2,362,869,144-byte official archive. Extracted originals occupy
35,099,010 bytes. Every image passed archive CRC and Pillow verification.

The Daily-Omni archive contains **3,910,892,544 bytes** and passed SHA-256
verification against its published archive identifier. The 188 selected original
files occupy 542,939,534 bytes. All 94 original video files contained audio, so
94 full-duration silent copies were prepared and independently probed. Frame
counts, visual codec, dimensions, and video duration were preserved; question
labels were unchanged. No selected video exceeds two minutes. See
[AV status](AV_STATUS.md) for integrity reports and decoder setup.

## Completed retrieval runs

| Dataset/method | Questions | Recall@5 | Hit@5 | MRR@5 | nDCG@5 | All-support@5 |
|---|---:|---:|---:|---:|---:|---:|
| AMI dev / BM25 | 24 | 0.2498 | 0.7917 | 0.5708 | 0.3720 | 0.0833 |
| AMI dev / EmbeddingGemma 2 dense text | 24 | 0.3889 | 0.9583 | 0.7340 | 0.5253 | 0.0833 |
| AMI held-out test / BM25 | 22 | 0.3711 | 0.9545 | 0.8333 | 0.5624 | 0.0455 |
| AMI held-out test / EmbeddingGemma 2 dense text | 22 | 0.4815 | 0.9545 | 0.8447 | 0.6471 | 0.1818 |
| MultiModalQA pilot / BM25 | 100 | 0.6214 | 0.7500 | 0.6165 | 0.5667 | 0.5000 |
| MultiModalQA pilot / EmbeddingGemma 2 dense text | 100 | 0.7714 | 0.9100 | 0.8195 | 0.7411 | 0.6400 |
| MultiModalQA pilot / EmbeddingGemma 2 native multimodal | 100 | 0.8229 | 0.9400 | 0.8160 | 0.7627 | 0.7100 |

Reports: [AMI development comparison](../results/ami-dev-comparison/comparison.md),
[development BM25](../results/ami-dev-comparison/bm25.json),
[development dense text](../results/ami-dev-comparison/dense-text.json),
[AMI held-out comparison](../results/ami-test-comparison/comparison.md),
[test BM25](../results/ami-test-comparison/bm25.json),
[test dense text](../results/ami-test-comparison/dense-text.json), and
[MMQA BM25](../results/mmqa-pilot/bm25.json).

On identical AMI development inputs, dense text improved Recall@5 by 13.91
percentage points and Hit@5 by 16.67 points. The final protocol was frozen before
scoring the 22 held-out ES2011 questions: ten-utterance chunks, supplied-meeting
scope, top five evidence units, and the same pinned model revision/float32
precision. Held-out Recall@5 improved by 11.04 percentage points; Hit@5 was
unchanged. These results support semantic transcript retrieval on this small
subset; they do not establish an image, audio, or video advantage.

The full MMQA comparison completed on all 100 questions and 2,015 sources.
Native multimodal Recall@5 exceeded dense text by 5.14 percentage points. For
the 42 image-dependent questions, recall of gold image sources was 21.43% for
BM25, 23.81% for dense text and 43.25% for native multimodal retrieval.
[MMQA results](MMQA_RESULTS.md) includes per-modality analysis, error cases,
uncertainty estimates and resumed-run timing limitations.
AMI searches within the supplied meeting; MMQA searches the union of supplied
candidate contexts, including distractors. These are custom experiments, not
published leaderboard scores. The [evaluation protocol](EVALUATION.md) records
controls and gold-support ceilings. Shared-machine timings are operational
observations, not isolated hardware benchmarks.

## Application and runtime verification

- Python 3.11.9; PyTorch 2.12.1+cu130; CUDA verified on an NVIDIA RTX 4060 Laptop
  GPU with 8 GB VRAM; Sentence Transformers 6.1.0 and Transformers 5.19.0.
- EmbeddingGemma 2 revision:
  `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- Native text/image encoding and shared-space query ranking passed using real
  downloaded content. [Text/image smoke report](../results/embedding-smoke.json).
- Native audio and video encoding passed on the first three seconds of one real
  Daily-Omni clip using CPU/two threads. Audio and silent video produced valid
  embeddings and could be ranked together. This is a runtime smoke test, not a
  quality score. [AV smoke report](../results/av-embedding-smoke.json).
- SoundFile 0.14.0, TorchCodec 0.17.0+cpu, and local shared FFmpeg 8.1.2 provide
  verified Windows decoding. Process-local library configuration avoids global
  PATH changes.
- The final full suite passed **298 tests and 17 subtests**, with one skipped
  Windows symlink-permission test. Coverage includes local runtime, generators,
  GPU handoff, evaluation resume/replay, AV decoding, analysis, packaging and HTTP
  application behavior. A reproduced Windows rejected-POST connection-reset race
  was fixed and regression-tested. Tiny synthetic media are explicitly test
  fixtures, not benchmark evidence.
- A real AMI question completed through the NVIDIA endpoint with validated
  source IDs. [Hosted text smoke report](../results/nvidia-text-smoke.json).
  Browser source search and cached-answer display also passed manual checks.

Ingestion covers text/Markdown, CSV/TSV, JSON/JSONL, HTML, PDF page text/images,
DOCX/PPTX text and embedded images, images, and physical 20-second audio/video
clips. Office layouts are not rendered. The app supports collection selection,
source previews, citations, uploads, and measured report display. Details are in
[the app guide](APP.md) and [API notes](API.md).

A real 20-second AV application demo completed native retrieval and local
generation. Both source players loaded, audio playback advanced, citation
navigation worked, and Top 1 retrieval correctly added its matching audio record.
The answer described both modalities but cited only the video, a documented
support limitation. [Observed response](../results/demo-av-live.json) and
[rehearsal notes](DEMO_GUIDE.md) distinguish integration checks from accuracy.

## Optional extensions beyond the local project

1. Extend the completed 100-question MMQA retrieval and local-answer comparison
   with an untouched question sample if
   needed. OCR/caption enrichment would be a separate,
   stronger baseline rather than part of the current text baseline.
2. Resume interrupted answer pilots when provider availability improves.
   MMQA BM25 completed 5/5 questions with 60% answer-set exact match and 60%
   coverage. Daily-Omni audio completed 5/5 with 0% accuracy and 20% coverage;
   video completed 3/5 and combined 1/5 before repeated HTTP 503 failures.
   Gemini completed all three MMQA retriever pilots at 80% exact match on the
   same five questions. Its audio-only pilot scored 40%; combined stopped before
   saving a prediction, and video-only was not run after that failure. Do not
   compare partial-condition scores with completed runs or generalize from five
   questions. Use new output directories after the Gemini schema compatibility
   fix, which changed the recorded implementation hash for older NVIDIA runs.
3. Evaluate citation support beyond valid IDs, and distinguish retrieval failures
   from generation failures. Lexical answer metrics alone do not establish
   factual correctness.
4. If extending to AMI recordings, acquire and align those recordings with
   transcript evidence. Generic AV ingestion does not establish that alignment.

Answer evaluation checkpoints each question and resumes only with identical
inputs, configuration, and relevant code hashes. AMI uses token F1/ROUGE-L;
MMQA uses custom answer-set metrics and token F1; Daily-Omni uses choice accuracy
and coverage. See the [evaluation protocol](EVALUATION.md).

Rerun commands are in the [README](../README.md). Data, weights, caches, and
reports remain local artifacts excluded by `.gitignore`.
