# ProjectMemory demonstration

Audience: students and project teams searching mixed-media collections. The main
demo shows a natural-language question, retrieval of original audio/video,
local generation, and evidence the audience can inspect. It uses **only the
first 20 seconds** of Daily-Omni clip `-jR3C_yA_G0`; it is an application demo,
not a benchmark question or a measured accuracy result.

## Observed rehearsal — 15 October 2026

The browser retrieved both 0–20 second media records using native embeddings
and generated an answer with local Gemma. It described the spiral-galaxy image,
the overlaid “Spiral Arm Density Wave Theory” title, and the spoken account of
early ideas about orbiting spiral-arm material. The original answer cited only
the silent-video record despite discussing speech: citation IDs are valid, but
audio citation support is incomplete. No human correctness score is assigned.

Both media elements loaded without errors; audio playback advanced beyond six
seconds. Source-citation navigation and HTTP byte-range serving passed. A Top 1
request added the matching audio record to the retrieved video, producing the
same two-record input and reusing the cached answer. The first observed search
took 145.31 seconds and the first generation 23.14 seconds; these are single-run
startup observations, not controlled performance measurements.

The [saved response and verification](../results/demo-av-live.json) preserve the
actual output; its HTTP snapshot reused the browser generation cache. The
[browser screenshot](../results/demo-av-browser.jpg) records the displayed answer.
This integration success does not override the low aggregate AV accuracy.

## Prepare and launch

Run from the project directory after the existing datasets, EmbeddingGemma
weights, local Gemma runtime, and FFmpeg have been installed. Preparation uses
the original video **with its soundtrack**, identified by `original_path` in
the Daily-Omni corpus. It performs no inference or downloads.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_demo.py
.\.venv\Scripts\python.exe run.py validate --dataset data/prepared/demo-av
.\.venv\Scripts\python.exe run.py serve
```

Open [the local app](http://127.0.0.1:8765). Stop other GPU experiments before
the live demo. The first search can load EmbeddingGemma and build the index;
the first uncached answer also starts local Gemma. Allow time for startup and
GPU handoff. The interface displays elapsed time; no fixed response-time
guarantee is implied. Stop the app with Ctrl+C when finished.

## Three-minute AV walkthrough

1. Select **demo-av**, **EmbeddingGemma · multimodal**, **CUDA GPU**,
   **Gemma 4 E2B · local**, and **Top 3**. Use **All sources**.
2. Ask exactly: **“Describe the visible scene and the spoken explanation.”**
3. Select **Search sources**. Inspect the audio and video cards and their
   **0–20 sec** locations. There are only two evidence records, so Top 3
   cannot return three. Both come from the same original file and interval.
4. Select **Generate answer**. Read the actual returned answer and its cited
   source IDs. Select a citation to locate its evidence card; use playback or
   **Open source** to check the claim. The video preview is intentionally
   silent; listen to the separate audio card. Playback controls are independent.

The intended outcome is an explanation supported by visible and audible
evidence. Preparation does **not** establish that the answer is correct or
complete. Report what the live run actually says, including omissions or an
abstention. Source-ID validation confirms a reference exists; it does not
verify every claim. With only two candidates, this demonstrates integration,
not competitive retrieval quality over a large collection.

## Optional examples and an honest limitation

- **AMI text:** choose `ami`, BM25, meeting/source `ES2006a`, local Gemma,
  Top 3. Ask **“What features should the remote control have?”** Inspect the
  retrieved transcript utterances and compare them with the generated answer.
- **MMQA image:** choose `mmqa`, multimodal retrieval, CUDA, local Gemma,
  Top 3. Ask the existing dataset question **“What type of piercing does
  Manuel Lazzari have?”** Inspect whether the relevant original photograph was
  retrieved before assessing the answer. This is a demonstration question;
  do not treat a convenient example as an overall accuracy measurement.
- **Missing evidence:** in `demo-av`, ask **“What happens 45 seconds after
  this clip starts?”** Only seconds 0–20 are available. An appropriate answer
  acknowledges missing evidence; a specific unsupported continuation is an
  error to discuss. Abstention is a behavior to test, not a promised outcome.

Video generation uses timestamped frames sampled at 1 FPS, so brief actions
between frames can be missed. The local model also has request limits of
30 seconds of audio and 60 seconds of video in total. Larger collections need
bounded evidence selection. A citation and successful playback are not proof
of accurate audio understanding or temporal reasoning.

## Rehearsal, offline use, and fallback

After stopping other GPU work, this CLI command runs the same AV question and
saves the **observed** response. Run it before presenting; check the evidence
yourself. It performs local inference and may take time on a cold start.

```powershell
.\.venv\Scripts\python.exe run.py ask --dataset data/prepared/demo-av --method multimodal --device cuda --batch-size 2 --dtype float32 --provider local --top-k 3 "Describe the visible scene and the spoken explanation." --output results/demo-av-live.json
```

For offline rehearsal, keep all model weights, runtime files, source media,
and `.cache/` contents locally. In a new PowerShell session, set these before
starting the app; missing Hugging Face assets then fail instead of downloading.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe run.py serve
```

Repeating the exact question with unchanged sources/settings can reuse the
generation cache; the UI marks a **cached response** when it does. Cache hits
are not guaranteed after code, model, prompt, or evidence changes, and retrieval
can still need the embedding model. Do not depend on a cache to fix a missing
runtime or unavailable GPU.

If live inference fails, show the previously saved response explicitly as a
**recorded rehearsal**, and inspect its source files or the saved media in
`data/prepared/demo-av/assets/`. Reading the JSON requires no inference:

```powershell
Get-Content -LiteralPath results/demo-av-live.json -Raw
```

If that snapshot has not been created, demonstrate source playback and the
existing results dashboard only. State the live failure; do not invent a
response or present previously recorded timings as a new run.
