# Local Gemma 4 E2B — verified 15 October 2026

Gemma 4 E2B is now the default answer provider. EmbeddingGemma 2 still retrieves
evidence; Gemma generates an answer with source IDs. NVIDIA and Gemini remain
available as optional providers, with saved historical pilots. No API key or
cloud inference is used by `--provider local`.

## Run

The model is already installed in this workspace. Start the app:

```powershell
.\.venv\Scripts\python.exe run.py serve
```

Open the printed address, normally `http://127.0.0.1:8765`, and select
**Gemma 4 E2B · local**. The model starts automatically on the first uncached
answer, on loopback port 8766. Stop the app server with Ctrl+C to stop its owned
model process; closing the browser tab does not stop the server.
The app releases embedding-model GPU memory before local generation and stops
the answer process before the next CUDA retrieval operation. Repeated switching
therefore includes model-loading time.

CLI example:

```powershell
.\.venv\Scripts\python.exe run.py ask --dataset data/prepared/ami --method bm25 --provider local --source-id ES2006a --top-k 3 "What features should the remote control have?" --output results/local-example.json
```

For a new checkout, explicitly run `scripts/setup_local_gemma.py` with the project
Python first. It downloads approximately 5 GB of model/runtime archives to
`.cache/local-gemma4`, verifies publisher SHA-256 values, and records the files in
`runtime.json`. It does not install global packages or change the system PATH.
Answer requests do not download models automatically.

## Pinned implementation

- Model: `google/gemma-4-E2B-it-qat-q4_0-gguf`, revision
  `675cff42a74c774d6cb76f76d8eacb49b48c9b93`.
- Official QAT Q4_0 weights plus the model's audio/vision projector: 4,336,349,920 bytes.
- Runtime: llama.cpp `b11429`, Windows x64 CUDA 12.4 build; 16,384-token context,
  one request at a time, context shifting disabled, offline mode enabled.
- Generation: temperature 1.0, top-p 0.95, top-k 64, seed 42, thinking disabled;
  default maximum output is 4,096 tokens. GPU kernels and prompt-cache state can
  still affect reproducibility; a fixed seed does not guarantee identical text.
- Structured generation uses `response_format.json_schema.schema`, as required
  by the pinned server parser. The README's flat `json_schema` example silently
  ignores the intended schema. Local validation additionally rejects unknown or
  duplicate citations and inconsistent abstentions. One formatting retry is allowed.
- Cache and evaluation manifests include input hashes, model/projector identity,
  runtime identity, generation settings, and implementation fingerprints.

## Verified execution

The [final integration report](../results/local-gemma-smoke.json) contains five
successful, uncached requests on the RTX 4060 Laptop GPU (8,188 MiB). Every
response passed schema and citation-ID validation without a formatting retry.

| Input | Check | Elapsed seconds |
|---|---|---:|
| Text | Answer from AMI meeting transcript chunks | 23.75 |
| Image | Describe an original MMQA image | 2.40 |
| Audio | Transcribe a native 20-second speech clip | 2.38 |
| Video | Describe 20 seconds represented by 20 timestamped frames | 7.90 |
| Audio + video | Describe the visual scene and spoken explanation together | 10.10 |

The first request includes loading and file verification. Later requests reuse
the loaded model. These are integration timings, not a controlled latency
benchmark. GPU memory observed **after** requests ranged from 2,763 to 3,141 MiB;
these readings are not peak memory measurements or a promise for larger inputs.

Two consecutive browser questions also completed with EmbeddingGemma on CUDA
and local Gemma generation, verifying GPU handoff in both directions. They took
95.60 seconds for the cold request and 26.04 seconds for the next request. The
second answer cited its evidence but omitted a requested cost that was present
in the source, so successful execution does not mean complete answer accuracy.
The completed local answer comparison now covers the same 100 MMQA questions
under three retrieval methods. Earlier five-question runs remain historical pilots.

Earlier attempts are retained in `results/local-gemma-smoke-first-attempt.json`,
`local-gemma-smoke-second-attempt.json`, and `local-gemma-smoke-pre-schema-fix.json`.
They exposed ignored schema constraints, invalid citations, output-limit failures,
and audio prompts that incorrectly claimed missing audio. The final audio check
uses an explicit transcription prompt. Successful speech transcription and one
combined example do not establish reliable general sound/event understanding.

## Completed 100-question local comparison

All three runs use the same fixed 100 MMQA development question IDs, 2,015-source
candidate union, top-five evidence limit, and local generator settings. They replay
validated saved retrieval rankings without consulting gold labels to choose evidence.

| Retrieval method and report | Exact answer-set match | Declared answer coverage | Invalid model responses |
|---|---:|---:|---:|
| [BM25](../results/final-mmqa-local-bm25/report.json) | 14% (14/100) | 63% | 9 |
| [Dense text](../results/final-mmqa-local-dense-text/report.json) | 19% (19/100) | 79% | 8 |
| [Native multimodal](../results/final-mmqa-local-multimodal/report.json) | 22% (22/100) | 81% | 6 |

Exhausted grounded-JSON formatting failures count as unanswered and incorrect in
the full denominator. Startup, network and filesystem errors still stop a run.
Coverage is the structured `abstained=false` rate, not an assessment that every
answer is useful or correct. Exact match compares normalized answer items and can
miss semantically correct prose or overlook contradictions outside those items.

These are within-project development results, not leaderboard scores or an API
model ranking. See the [frozen expanded protocol](FINAL_EXPERIMENT_PROTOCOL.md),
[technical report](TECHNICAL_REPORT.md), [complete generation metrics](GENERATION_RESULTS.md),
and [fixed first-ten AI-assisted error inspection](ERROR_ANALYSIS.md).

## Historical five-question pilot

The [local MMQA report](../results/answers-mmqa-local-bm25-5/report.json) evaluates
the same five questions and BM25 top-five evidence as the saved API pilots:

| Generator | Exact answer-set match | Answer coverage |
|---|---:|---:|
| Local Gemma 4 E2B QAT Q4_0 | 40% (2/5) | 100% |
| NVIDIA Nemotron 3 Nano Omni | 60% (3/5) | 60% |
| Gemini 3.8 Flash | 80% (4/5) | 100% |

Local token F1 was 0.50; gold-source citation precision was 0.64 and recall was
0.70. Five questions cannot establish a general model ranking. Sampling,
quantization, formatting, and reasoning settings also differ. These scores must
not be compared directly with the larger 100-question local runs. See the
[generation results](GENERATION_RESULTS.md) for the API conditions and metric caveats.

## Media limits and evaluation

Requests allow at most **30 seconds of audio and 60 seconds of video in total**,
64 images/frames, and 32 MB of source-media bytes. Limits are checked before
inference; evidence is never silently truncated. A request can also exceed the
16K-token context before reaching these media limits. Reduce evidence count when
that happens.

Video is sampled at one-second intervals using decoded presentation timestamps,
with repeated frame indices deduplicated and a maximum frame long edge of 768
pixels. The current decoder selects the frame active at each sample time, labels
its actual timestamp, and emits full-range JPEG frames. Visual records must be silent; provide the separated
audio record when its soundtrack matters. The application already creates
20-second media segments. The application pairs matching audio/video segments and selects whole evidence
groups within these limits, showing added and omitted records. Controlled
evaluations bypass this selection and retain their prescribed evidence. PDF pages are rendered during ingestion.

The smoke clips are cropped prefixes used only for integration checks. Full
Daily-Omni benchmark clips often exceed the audio limit. The expanded local AV
experiment uses a separately documented 20-question subset from 52 eligible
questions. It removes only three terminal PCM samples (0.1875 ms) from each
selected audio file to meet the 30-second boundary, preserves originals and the
complete silent video, and keeps the same questions across all three modalities.
All three conditions completed the same 20 questions under identical generator
configuration and repaired video-decoder protocol version 2:

| Supplied media and report | Choice accuracy | Declared coverage | Invalid responses |
|---|---:|---:|---:|
| [Audio only](../results/final-local-av-audio/report.json) | 10% (2/20) | 10% | 10 |
| [Silent video only](../results/final-local-av-video/report.json) | 15% (3/20) | 45% | 4 |
| [Audio + silent video](../results/final-local-av-combined/report.json) | 10% (2/20) | 60% | 5 |

Every condition scores below the 25% expected accuracy of uniform four-choice
guessing and the 25% always-A baseline on this sample. Combined input increases
declared coverage but shows no accuracy advantage. These results do not establish
reliable AV question answering; accepted media and successful transcription are
separate integration properties. Invalid outputs and valid abstentions both count
as incorrect, with invalid outputs also reported separately.

See the [normalization and comparison protocol](FINAL_EXPERIMENT_PROTOCOL.md),
[AV status](AV_STATUS.md), and [paired analysis](../results/final-analysis/comparison.md).
Archived pre-repair outputs are excluded. No full Daily-Omni accuracy or
general model ranking is claimed from this small normalized subset.

Sources: [official Google model](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf),
[Gemma input limits](https://ai.google.dev/gemma/docs/core/model_card_4),
[runtime release](https://github.com/ggml-org/llama.cpp/releases/tag/b11429), and
[pinned schema parser](https://github.com/ggml-org/llama.cpp/blob/b11429/tools/server/server-common.cpp).
