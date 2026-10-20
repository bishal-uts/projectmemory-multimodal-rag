# Audio and video pipeline

Updated 15 October 2026 (Australia/Sydney).

`projectmemory.av.segment_media(path, output_dir, chunk_seconds=20,
max_duration_seconds=None)` returns ordinary retrieval corpus records for
physically clipped files. It reads local files with FFmpeg/ffprobe and emits:

- silent H.264 MP4 clips for a temporal video stream;
- separate mono 16 kHz PCM WAV clips if an audio stream actually exists;
- WAV clips only for audio inputs.

Fixed boundaries do not depend on questions or answers. Every record contains
the source SHA-256, original source path, source stream index, source duration,
start/end source-relative seconds, and measured encoded duration. Delayed
streams retain their real start offset. Video boundaries have frame precision;
the measured file duration can differ slightly from the requested source interval.
The first video and first audio stream are used. Embedded album artwork does
not count as temporal video. A configured duration limit intentionally selects
only a source prefix and is recorded explicitly.

FFmpeg receives argument lists with local-only file protocols, a timeout, and
two encoder threads. Generated paths are content/configuration hashed, kept
inside the output directory, probed, and atomically published. Inputs are not
modified. FFmpeg and ffprobe must be installed and available on PATH.

## Verification

The segmentation tests generate tiny synthetic media fixtures solely to verify
file handling and timing. They are not benchmark examples or quality evidence.
The tests cover audio/video separation, silent sources, delayed audio,
partial final segments, sample rates, bounded outputs and invalid parameters.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_av.py -q --basetemp .cache/pytest-av
```

Use a temporary directory inside the workspace under the Windows sandbox;
atomic rename from the sandbox's redirected system temporary folder can fail.

The real-media runtime smoke uses one matching Daily-Omni audio/video pair,
physically crops its first three seconds, then encodes the waveform and up to
three visual frames with the pinned EmbeddingGemma 2 model:

```powershell
.venv\Scripts\python.exe scripts/smoke_av_embeddings.py --device cpu --threads 2
```

Its report is `results/av-embedding-smoke.json`. This verifies decoder and model
compatibility only. It does not measure retrieval accuracy or answer quality,
and the cropped prefix must not be substituted for full-video QA evaluation.

**Completed:** native audio and native video encoding and query search both
passed on the first three seconds of real clip `-jR3C_yA_G0`, using float32 on
CPU with two threads, pinned model revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. The full smoke took 89.6 seconds,
including initialization and source segmentation. This is a runtime result,
not evidence that the retrieved source answers the question.

## Data and dependencies

The prepared Daily-Omni sample contains 100 questions and 94 unique videos,
each represented by separate video and audio media. Its official media archive
is 3,910,892,544 bytes. The existing bounded downloader extracts only the 188
declared sample files:

```powershell
.venv\Scripts\python.exe run.py fetch-media --dataset data/prepared/daily-omni --max-mb 4500
```

**Completed:** all 188 original media files were fetched (542,939,534 extracted
bytes) and successfully probed. Every one of the 94 original `_video.mp4` files
contained embedded audio. `prepare_video_only_corpus` therefore created full
silent MP4 copies using video stream copy, with no frame re-encoding, and
updated the prepared corpus paths. Original media paths and hashes remain in
metadata, original files remain on disk, and question labels were unchanged.
All 94 copies passed independent silent-stream checks; frame counts and video
durations were checked during preparation. The original separate audio/video
durations differed by at most 0.030 seconds.

Reports: `results/av-media-probe.json`, `results/av-silent-preparation.json`,
`results/av-media-silent-probe.json`, and `results/av-data-integrity.json`.
If preparing this dataset from scratch again, run the silent-video preparation
after downloading media, before audio/video ablations.

TorchCodec 0.17.0 CPU and SoundFile 0.14.0 are installed in the project virtual
environment. TorchCodec's [official compatibility table](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)
supports PyTorch 2.11 or newer for this release. Native media input formats are
documented by [Sentence Transformers](https://www.sbert.net/docs/input_formats.html).

On Windows, TorchCodec needs **shared FFmpeg DLLs**, even when an existing
static `ffmpeg.exe` already works. The project-local Gyan FFmpeg 8.1.2 shared
build is installed at `.cache/ffmpeg/ffmpeg-8.1.2-full_build-shared/bin`.
Its archive SHA-256 was verified against the distributor's published checksum:
`cba748035c21ce1431d0823c7a3a711f38616f89f87a265dceddf9b7f6749d2d`.
[Gyan's builds](https://www.gyan.dev/ffmpeg/builds/) are linked from the
[official FFmpeg download page](https://ffmpeg.org/download.html).
`configure_media_runtime()` exposes this directory only to the current
process. Alternatively set `PROJECTMEMORY_FFMPEG_BIN` to another shared-build
bin directory before loading the model. System installation and user PATH
were not modified. Actual TorchCodec audio/video decoding succeeded with
PyTorch 2.12.1+cu130 and TorchCodec 0.17.0+cpu; CPU decoding and GPU model
inference can coexist.

## Local generation and evidence budgets

The application retrieves ordinary audio/video records, pairs matching source
segments (including streams with delayed starts), and selects whole evidence
groups within the local generator's 30-second total audio, 60-second total video,
64-image/frame and 32 MB source-media limits. It displays added counterparts and
omitted records. This selection does not crop source files. Controlled answer
experiments bypass application selection and preserve their prescribed evidence;
the generator still rejects inputs exceeding its strict limits.

The local video adapter uses one decode pass after inspecting presentation
timestamps. At each one-second sample time it selects the latest active frame,
deduplicates repeated frame indices and labels the actual frame timestamp. The
last active frame can cover a fractional container-duration tail. Selected
frames become full-range JPEGs with a maximum long edge of 768 pixels. A decode
error stops the request rather than sending a partial video. This repair replaces
independent per-second seeking, which could request a nonexistent final frame
when the container duration exceeded the last frame timestamp.

Separate [local integration checks](../results/local-gemma-smoke.json) accepted
text, image, audio transcription, sampled video and combined AV. Their media
inputs are 20-second prefixes with no reference-answer scoring. Successful speech
transcription and accepted video input do not establish general sound/event QA
accuracy. The [local guide](LOCAL_MODEL.md) records timings and memory readings;
these are integration observations, not controlled latency or peak-memory results.

## Daily-Omni local answer experiment

Daily-Omni supplies each question's clip and four choices. It has no
collection-retrieval relevance labels, so Recall@k is not reported. Local
audio-only, silent-video-only and combined conditions use the same questions and
generator to isolate the supplied modalities.

The [strict full-audio subset](../data/prepared/daily-omni-local/manifest.json)
contains only two eligible questions from the original 100. A separately recorded
[boundary-normalized subset](../results/final-local-av-audio/source_manifest.json)
selects 20 questions with seed 42 from 52 eligible questions, covering 16 clips.
Eligibility requires audio no longer than 30.001 seconds, silent video no longer
than 60 seconds, and request byte/frame compatibility. Only terminal PCM samples
beyond 30 seconds are removed: each selected audio loses three samples at 16 kHz,
or 0.1875 ms. Originals, retained PCM bytes and complete silent videos are
preserved, with hashes and sample counts in the manifest. This is not an untouched
full-clip benchmark and involves no question-dependent temporal cropping.
The normalized manifest is included in the distributable archive. The strict
subset's local manifest requires running the data preparation steps.

All three conditions completed under identical generator configuration,
question/corpus fingerprints and repaired video-decoder protocol version 2:

| Supplied media and report | Completed | Choice accuracy | Declared coverage | Invalid responses | Valid abstentions |
|---|---:|---:|---:|---:|---:|
| [Audio only](../results/final-local-av-audio/report.json) | 20/20 | 10% (2/20) | 10% (2/20) | 10 | 8 |
| [Silent video only](../results/final-local-av-video/report.json) | 20/20 | 15% (3/20) | 45% (9/20) | 4 | 7 |
| [Audio + silent video](../results/final-local-av-combined/report.json) | 20/20 | 10% (2/20) | 60% (12/20) | 5 | 3 |

All observed accuracies are below the 25% expected accuracy of uniform
four-choice guessing and the 25% (5/20) always-A baseline. Combined media produces
more declared answers but no demonstrated accuracy advantage. This is a
descriptive comparison on a small sample, not a statistical-significance claim.
The project therefore demonstrates working AV input handling, but does not
establish reliable AV question answering. Invalid responses after the existing
format retry and valid abstentions both count as incorrect; the two kinds of
non-answer remain separately reported.

The [frozen expanded protocol](FINAL_EXPERIMENT_PROTOCOL.md) records all three
conditions and failure handling. The [paired analysis](../results/final-analysis/comparison.md)
includes category breakdowns, trivial choice baselines and clip-cluster
uncertainty estimates. The pre-repair audio run and interrupted video attempt
remain under `results/pre-decoder-fix-local-av-*` for provenance and are excluded
from final results. These results concern the normalized subset, not the complete
Daily-Omni dataset or an untouched full-clip benchmark.

Historical five-question API pilots remain in [generation results](GENERATION_RESULTS.md).
The completed 100-question local MMQA comparison there concerns text/table/image
answering; it does not measure AV quality. See the [technical report](TECHNICAL_REPORT.md)
and [fixed-sample answer error analysis](ERROR_ANALYSIS.md) for interpretation.
