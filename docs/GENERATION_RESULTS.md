# Generation results — 15 October 2026

The completed main answer experiment compares three retrievers with local Gemma
4 E2B on the same fixed 100 MMQA development questions. Earlier five-question
local/API results are retained below as historical integration pilots. Their
different sample sizes and generator settings do not support a direct API-versus-local
ranking. Input acceptance, answer accuracy, retrieval quality, and citation
support are separate properties.

## Completed local MMQA comparison

All conditions use the same 100 question IDs, 2,015-source candidate union, k=5,
and Gemma 4 E2B QAT Q4_0 generator: revision
`675cff42a74c774d6cb76f76d8eacb49b48c9b93`, llama.cpp b11429,
temperature 1.0, top-p 0.95, top-k 64, seed 42, 4,096 output tokens, disabled
thinking and one formatting retry. Saved rankings were replayed after validating
corpus/media and query fingerprints, source scopes and ranked IDs. This avoids
embedding/generation model swaps without changing the retrieved evidence.

| Retrieval method and report | Completed | Exact answer-set match | Declared coverage | Token F1 | Gold-citation precision | Gold-citation recall | Invalid responses |
|---|---:|---:|---:|---:|---:|---:|---:|
| [BM25](../results/final-mmqa-local-bm25/report.json) | 100/100 | 14% | 63% | 20.68% | 52.68% | 48.00% | 9 |
| [Dense text](../results/final-mmqa-local-dense-text/report.json) | 100/100 | 19% | 79% | 27.14% | 64.87% | 57.00% | 8 |
| [Native multimodal](../results/final-mmqa-local-multimodal/report.json) | 100/100 | 22% | 81% | 30.00% | 69.08% | 59.29% | 6 |

Answer-set precision/recall are 16.50%/15.67% for BM25, 22.50%/21.17% for dense
text, and 25.50%/24.17% for native multimodal retrieval. All metrics retain all
100 questions. A response that still violates the grounded-answer contract after
the existing formatting retry is recorded as `invalid_model_response`, unanswered
and incorrect. It is counted separately from valid abstentions. Infrastructure
failures remain run-stopping errors, not invented wrong answers.

Native multimodal retrieval has the highest exact-match count in this sampled
development experiment. This does not establish general model superiority or
performance on an untouched test set. Coverage reflects the model's structured
abstention flag; exact match compares answer items, not the factual correctness
of every generated sentence. Gold-citation overlap is not claim-level entailment.
Identical-input cache reuse is allowed; saved timings mix cached, warm and cold
requests and are not a controlled latency or peak-memory benchmark.

The [frozen expanded protocol](FINAL_EXPERIMENT_PROTOCOL.md) records selection and
failure handling. The [technical report](TECHNICAL_REPORT.md) develops the
comparison, and the [fixed first-ten AI-assisted inspection](ERROR_ANALYSIS.md)
shows omissions, answer-item representation errors, contradictory prose and
unsupported additions. It is not independent human annotation or an estimate
of entailment accuracy.

## Completed local Daily-Omni comparison

All three conditions completed the same 20 explicitly boundary-normalized
questions, covering 16 clips, under identical local generator settings and
repaired video-decoder protocol version 2. The question and corpus fingerprints
match across conditions. This supplies the known clip and choices directly;
it evaluates answer generation, not corpus retrieval.

| Supplied media and report | Completed | Choice accuracy | Declared coverage | Invalid responses | Valid abstentions |
|---|---:|---:|---:|---:|---:|
| [Audio only](../results/final-local-av-audio/report.json) | 20/20 | 10% (2/20) | 10% (2/20) | 10 | 8 |
| [Silent video only](../results/final-local-av-video/report.json) | 20/20 | 15% (3/20) | 45% (9/20) | 4 | 7 |
| [Audio + silent video](../results/final-local-av-combined/report.json) | 20/20 | 10% (2/20) | 60% (12/20) | 5 | 3 |

All three observed accuracies fall below the **25% expected accuracy of uniform
four-choice guessing**. Always choosing A also scores 25% (5/20) on this subset.
These are descriptive comparisons, not a claim of statistical significance.
Combined input increases the declared answer rate but demonstrates no accuracy
advantage: it matches audio-only and scores one question below silent video.
Invalid model responses and valid abstentions both remain incorrect in the full
20-question denominator. The small quantized local model has not demonstrated
reliable AV reasoning in this experiment, despite successful media integration.

The [paired analysis](../results/final-analysis/comparison.md) includes categories
and uncertainty estimates accounting for shared clips. The
[frozen expanded protocol](FINAL_EXPERIMENT_PROTOCOL.md) and [AV status](AV_STATUS.md)
document eligibility, the removal of only three terminal PCM samples per selected
audio, decoder behavior and archived pre-repair runs. The repaired runs replace
those earlier attempts; archived scores are excluded from the final comparison.
No full Daily-Omni or API-versus-local quality claim follows from this subset.

## Historical NVIDIA Nemotron pilots

Model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, served through NVIDIA NIM. Configuration: 4,096 maximum output tokens, 1,024 reasoning budget, temperature 0.6, top-p 0.95, seed 42, and five requests per minute. The generator receives the question, optional multiple-choice options, and supplied evidence. Reference answers and gold labels are used only by local evaluation.

### Completed answer evaluation

| Condition | Completed | Answer score | Answer coverage | Abstentions |
|---|---:|---:|---:|---:|
| MMQA, BM25 top five evidence sources | 5/5 | 3/5 exact answer sets (60%) | 60% | 2 |
| Daily-Omni, audio only | 5/5 | 0/5 choices correct (0%) | 20% | 4 |

The [MMQA report](../results/answers-mmqa-bm25-5/report.json) also records macro gold-source citation precision of 50%, recall of 60%, and token F1 of 60%. All five questions had their full annotated supporting-source sets among the retrieved evidence, including both abstentions. Those abstentions therefore cannot be explained by missing annotated retrieval sources alone. Three requests contained native image bytes alongside text; this verifies accepted image input, but does not establish that an image was necessary for a particular answer. Tables were supplied as text.

The [Daily-Omni audio report](../results/answers-daily-audio-5/report.json) records four abstentions and one incorrect non-abstaining choice. These five questions have official categories AV Event Alignment (three) and Event Sequence (two). Insufficient modality information is one possible explanation for some abstentions; this small pilot does not establish their cause.

### Partial conditions

| Condition | Completed | Observations on completed items | Full five-question accuracy |
|---|---:|---|---|
| Daily-Omni, silent video only | 3/5 | One correct choice, one incorrect choice, one abstention | Unavailable |
| Daily-Omni, silent video plus audio | 1/5 | One incorrect choice | Unavailable |

Both conditions stopped on HTTP 503 after the adapter's bounded transport retries. Their saved [video predictions](../results/answers-daily-video-5/predictions.jsonl) and [combined predictions](../results/answers-daily-combined-5/predictions.jsonl) remain available. No completed `report.json` was created for either partial condition. Missing responses are not assigned invented scores. The audio condition initially encountered the same error; one identical resume completed it. Video and combined conditions were not repeatedly restarted.

Daily-Omni uses supplied media for each question, so these are modality ablations for answer generation, not retrieval measurements. All three modes use the same five question IDs and the same corpus/query fingerprints. Audio inputs have an audio stream only; video inputs have a video stream and no audio stream. Full clip durations were approximately 30–60 seconds. The largest combined audio/video input was 6,655,307 bytes, below the configured 8 MB raw-media cap. No temporal cropping or extra compression was used for this pilot.

NVIDIA MMQA dense-text and native-multimodal answer pilots were deferred after repeated endpoint failures. The separately completed retrieval benchmarks remain valid. A fixed-generator comparison was subsequently completed using Gemini, below.

### Usage and integration evidence

| Condition | Saved predictions | Generation requests represented | Reported total tokens | Sum of saved item elapsed times |
|---|---:|---:|---:|---:|
| MMQA BM25 | 5 | 5 | 11,438 | 132.3 s |
| Daily-Omni audio | 5 | 6 | 8,019 | 116.8 s |
| Daily-Omni video, partial | 3 | 3 | 17,866 | 76.5 s |
| Daily-Omni combined, partial | 1 | 1 | 6,707 | 30.4 s |

Audio includes one successful JSON-format repair retry. Usage totals cover returned responses represented in saved predictions; they exclude failed requests without usage data. Elapsed totals exclude failed unsaved items and process restart overhead, and should not be interpreted as end-to-end benchmark duration or service-level latency.

The separate [AMI text smoke result](../results/nvidia-text-smoke.json) answered “What features should the remote control have?” from source `ES2006a`, citing two retrieved utterance groups. It is a custom integration check with no reference-answer score. Across AMI, MMQA, and Daily-Omni, live calls accepted native text, image, audio, and video input. Successful schema and citation-ID validation establishes neither answer entailment nor factual correctness.

Each run directory contains a manifest with question/corpus fingerprints, generator settings, and implementation hashes. Resume an unfinished directory only with matching data and implementation; use a new directory after changing the evaluated implementation. Public benchmarks can overlap with model development data. These results do not establish unlimited free access or a generally superior model.

## Historical Gemini 3.8 Flash pilots

Model: `gemini-3.8-flash`, using `generateContent`, structured JSON, temperature 1.0, LOW thinking level, 4,096 maximum output tokens, and five requests per minute. Public evidence, prompts, query IDs, and scoring are shared with the NVIDIA conditions. Provider-specific sampling and reasoning settings differ, so provider comparisons are not a controlled model-quality experiment.

The first AMI attempt was rejected with HTTP 400 because Gemini disallows an empty enum member in the JSON response schema. The wire-level enum was removed, while strict local A–D/empty-label validation remains. A regression test was added and all 24 generation tests passed. The corrected [Gemini AMI smoke result](../results/gemini-text-smoke.json) cites the same two utterance groups as NVIDIA. This schema-only change alters `generation.py`'s implementation fingerprint. Existing NVIDIA runs retain their original fingerprints; resume their partial conditions in fresh output directories if using the updated code.

### Fixed-generator MMQA retrieval comparison

All three conditions completed the same five questions, with the same generator configuration and top-five evidence limit. Dense-text and native multimodal retrieval used the cached `google/embeddinggemma-2` model at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, float32 on CUDA. The question/corpus fingerprints match the NVIDIA MMQA pilot.

| Retrieval method and report | Completed | Exact answer-set match | Coverage | Token F1 | Gold-citation precision | Gold-citation recall |
|---|---:|---:|---:|---:|---:|---:|
| [BM25](../results/answers-mmqa-gemini-bm25-5/report.json) | 5/5 | 80% | 100% | 93.33% | 90% | 100% |
| [Dense text](../results/answers-mmqa-gemini-dense-text-5/report.json) | 5/5 | 80% | 100% | 93.33% | 90% | 100% |
| [Native multimodal](../results/answers-mmqa-gemini-multimodal-5/report.json) | 5/5 | 80% | 80% | 80% | 70% | 80% |

BM25 and dense text each generated “ear piercing” for a reference answer of “ear”, their sole exact-match miss. This illustrates why exact string matching is an incomplete answer-quality measure. The multimodal condition abstained on that question; its top five did not contain the annotated supporting image, whereas the other two methods retrieved it. Exact-match scores tie in this very small pilot; they do not show an answer-quality improvement from native multimodal retrieval.

Four native-multimodal outputs reused validated cached answers because the complete question/evidence input matched an earlier request. This is intentional reproducible reuse, not four independent new generations. One native-multimodal result required a new API call. These answer pilots should be interpreted alongside the much larger retrieval evaluation, rather than extrapolated as a reliable ranking from five questions.

### Daily-Omni modality pilots

| Condition | Completed | Observed result | Full five-question accuracy |
|---|---:|---|---:|
| [Audio only](../results/answers-daily-gemini-audio-5/report.json) | 5/5 | Two correct choices, three abstentions; 40% coverage | 40% |
| [Audio plus silent video](../results/answers-daily-gemini-combined-5/run.json) | 0/5 | First question failed with HTTP 503 after bounded retries | Unavailable |
| Silent video only | Not run | Further live calls stopped after persistent endpoint failures | Unavailable |

The audio condition used the same full clips and five questions as the NVIDIA condition. Combined media passed local duration/size validation, but no successful response or prediction was recorded. No complete modality comparison is available for Gemini. Its verified live paths are text, image, and audio. NVIDIA accepted video in the hosted pilots, and separate local Gemma integration checks also accepted sampled video. No model switch or billing change was used to bypass an error or quota.

### Gemini usage

| Condition | New calls with saved answers | Cache hits | Reported total tokens for new calls | Sum of saved item elapsed times |
|---|---:|---:|---:|---:|
| MMQA BM25 | 5 | 0 | 10,404 | 71.0 s |
| MMQA dense text | 5 | 0 | 9,143 | 113.9 s |
| MMQA native multimodal | 1 | 4 | 2,866 | 70.8 s |
| Daily-Omni audio | 5 | 0 | 7,626 | 65.9 s |

Tokens exclude cached outputs' historical usage to avoid double-counting. Failed requests did not return usage and are excluded, so these totals are not an account billing statement. Saved-item elapsed time can include retrieval and model initialization as well as generation; it is not a clean provider latency comparison. The AMI smoke call is separate from this table. All Gemini runs used separate output directories from NVIDIA.

## Historical local Gemma 4 E2B pilot

`google/gemma-4-E2B-it-qat-q4_0-gguf` at revision
`675cff42a74c774d6cb76f76d8eacb49b48c9b93` is installed with llama.cpp b11429 and
is the default application provider. The [same five-question MMQA BM25 pilot](../results/answers-mmqa-local-bm25-5/report.json)
completed 5/5, with 40% exact answer-set match, 100% coverage, token F1 0.50,
gold-source citation precision 0.64, and recall 0.70. Its question/corpus
fingerprints match the API BM25 pilots above. This is a small comparison with
different generator-specific settings, not a general model ranking.

Separate [native-media integration checks](../results/local-gemma-smoke.json)
completed text, image, audio transcription, video-frame, and combined AV requests.
Audio/video checks use 20-second prefixes and have no accuracy score. Early
failures and the structured-output fix are documented in the
[local model guide](LOCAL_MODEL.md), along with the model/runtime settings,
timings, GPU observations, and full-clip evaluation restrictions.
