# Expanded local evaluation protocol — 15 October 2026

This protocol extends the development pilots. It is fixed before complete local
comparison runs. It does not turn previously inspected development questions into
an untouched test set. The AMI held-out retrieval evaluation remains separate.

## Shared generator and failure policy

Use the installed Gemma 4 E2B QAT Q4_0 model and pinned llama.cpp b11429 runtime.
Keep the existing prompts, temperature 1.0, top-p 0.95, top-k 64, seed 42,
4,096-token output budget, disabled thinking and one formatting repair unchanged.
Do not tune these settings using the expanded experiment's answers.

An initial attempted expansion stopped after 14 MMQA predictions when the model
failed the grounded-answer contract after its prescribed repair. In complete
runs, this exact failure is recorded as `invalid_model_response`: unanswered and
incorrect in the full denominator, with no fabricated answer. It is also counted
separately from valid abstentions. Startup, network, filesystem and other runtime
failures still stop the experiment; they are not converted into wrong answers.

Cache reuse is allowed for identical input and generator fingerprints. Timing
mixes warm, cold and cached requests and is not a performance benchmark.

## Text, table and image answering

- Use all 100 questions in the existing seeded MMQA development subset and its
  unchanged 2,015-source candidate union.
- Compare BM25, dense text and native multimodal retrieval at k=5 using the same
  local generator. Replay each saved report's exact ranked IDs and scores after
  verifying corpus/media hashes, question hashes, source scopes and coverage.
- Replaying rankings avoids repeated GPU model swapping; it does not rerank,
  consult gold labels to choose evidence, or change the evidence budget.
- Report normalized answer-set exact match, token F1, answer coverage, invalid
  responses and gold-source citation overlap. Break down by gold-support
  modality and whether all annotated supports were retrieved.
- Compare paired per-question results with seeded exploratory bootstrap
  intervals. These estimates are conditional on the sampled candidate pool;
  shared sources and development-set use limit generalization.

## Audio/video answering

The original 100-question Daily-Omni sample has only two questions whose full
audio fits the strict 30-second limit. Many other audio files contain 30 seconds
plus three terminal PCM samples. A separate, explicitly normalized experiment
uses the following reproducible rule, applied before looking at answers:

1. Require audio no longer than 30.001 seconds, silent video no longer than
   60 seconds, and the local request's byte/frame limits. Missing or incompatible
   media is ineligible.
2. For PCM WAV audio in (30, 30.001] seconds, remove only the terminal samples
   beyond 30 seconds. Preserve originals, sample rate, channel count, PCM bytes
   before the cutoff and the complete video. Record both hashes and sample counts.
3. Sample 20 questions with seed 42 from the 52 eligible questions. This selects
   16 clips. Each selected audio file loses three samples at 16 kHz (0.1875 ms).
4. Run the same 20 questions and supplied choices in audio-only, silent-video-only
   and combined conditions. Select the frame active at each one-second sampling
   point and label its actual presentation timestamp; deduplicate repeated frames.

This evaluates supplied-media answering, not retrieval, and is not an untouched
full-clip benchmark. No meaningful-duration cropping or question-dependent media
selection is used. Report accuracy, coverage, invalid responses, categories,
always-A accuracy and the 25% expected accuracy of uniform guessing. Paired
intervals resample clips as clusters because some questions share a clip.

The strict two-question subset is retained separately. The normalized subset's
manifest records every inclusion/exclusion and original/derived media hash.

### Decoder correction and final run provenance

The initial AV video attempt exposed a duration/frame boundary mismatch: a
30.03-second container's last frame begins at 29.988292 seconds. Seeking a new
frame at second 30 could fail. Decoder protocol v2 probes presentation timestamps,
selects the active frame at each requested time, and decodes in one pass with
JPEG-compatible pixel format. Invalid timestamps, decode errors and missing
outputs remain fatal. It does not alter source media or silently drop failed frames.

All 16 selected clips passed independent decoding (489 valid JPEGs), and all
three AV conditions were rerun with protocol v2. Earlier attempts are archived
outside the final comparison. MMQA was completed before this video-only repair;
its three conditions share the earlier implementation, while all AV conditions
share the updated one. Analysis enforces identical configuration within each
experiment and retains/discloses the separate hashes across tasks. No benchmark
questions, prompts, generation settings or answer labels changed for this repair.

## Application versus controlled experiment

For user uploads, the application groups matching audio/video intervals and
selects whole evidence groups within local duration, frame and byte limits. It
displays which records were added or omitted. Sources are not cropped by this
selection step. Controlled MMQA and Daily-Omni experiments bypass this policy
and retain the evidence specified above.

The resulting report must distinguish retrieval quality, answer quality,
citation-ID validity and actual claim support. Neither lexical metrics nor a
valid citation ID alone proves factual correctness. Existing five-question API
pilots remain labelled as pilots; they are not compared directly to these larger
local runs with different sample sizes.
