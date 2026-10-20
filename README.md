# ProjectMemory: answers from multimodal project materials

An Advanced NLP application for finding evidence and answering questions across
meeting transcripts, documents, tables, images, audio, and video. The local app
combines **Google EmbeddingGemma 2** (`google/embeddinggemma-2`) for retrieval with
**Gemma 4 E2B QAT Q4_0** (`google/gemma-4-E2B-it-qat-q4_0-gguf`) for local cited
answers. NVIDIA Nemotron 3 Nano Omni and Gemini remain selectable API providers
for comparison. Local Gemma requires no API key or cloud inference.

Local text, image, speech transcription, sampled video, and combined audio/video
checks have passed on this laptop. See the [local model guide](docs/LOCAL_MODEL.md)
for setup, measured results, and limits. In the expanded 100-question local
MMQA comparison, answer-set exact match was **14% with BM25, 19% with dense text,
and 22% with native multimodal retrieval**. These strict development-set scores
show both a retrieval benefit and substantial remaining answer-quality limits.
Earlier five-question API pilots have a different denominator and remain separate.
On the completed 20-question supplied-media AV subset, audio/video/combined
accuracy was **10% / 15% / 10%**, below the 25% uniform-guessing expectation.
Input support does not establish reliable AV understanding. See the
[final comparison and uncertainty estimates](results/final-analysis/comparison.md).
See the [technical report](docs/TECHNICAL_REPORT.md),
[submission index](docs/SUBMISSION_INDEX.md), and [demo guide](docs/DEMO_GUIDE.md).
The assessment report and presentation are submitted separately and are not part
of this repository.

The app, file ingestion, retrieval, and answer evaluation are implemented. A
public AMI transcript question has completed through both hosted providers with valid
source IDs, and both answers displayed in the browser. The full retrieval
comparison and small answer pilots have measured reports. Some audio/video API
runs encountered HTTP 503 and remain incomplete; working integration alone is
not evidence of accurate answers. See [answer results](docs/GENERATION_RESULTS.md).

## Start the local app

A project-local `.venv` and optional API configuration are present on this machine.
From the project directory:

```powershell
.\.venv\Scripts\python.exe run.py doctor
.\.venv\Scripts\python.exe run.py serve
```

Open `http://127.0.0.1:8765`. Select a collection, search for sources, or generate
an answer with citations. Upload supported files to build your own collection.
See the [app guide](docs/APP.md) for controls and limits. Stop the server with
Ctrl+C. The app is a local development interface.

**Gemma 4 E2B · local** is the default answer model. Its installed llama.cpp
runtime starts on demand on `127.0.0.1:8766`; requests never download missing
model files. For first-time local model setup, run:

```powershell
.\.venv\Scripts\python.exe scripts/setup_local_gemma.py
```

This explicitly downloads about **5 GB** of pinned model, multimodal projector,
and Windows CUDA runtime archives, verifies their SHA-256 hashes, and records
them in `.cache/local-gemma4/runtime.json`. The runtime is llama.cpp **b11429**.
Model files and extracted binaries require additional disk space. The setup is
for this Windows NVIDIA GPU environment. `doctor` reports installation status
without starting a model or downloading files.

Choosing NVIDIA or Gemini sends the question and selected text/media evidence
to that hosted provider. Use public benchmark collections for these experiments. Keep
`NVIDIA_API_KEY` or `GEMINI_API_KEY` in the local, ignored `.env` file. Existing
credentials should not be overwritten. Provider details, limits, caching, and
data-handling references are in [generation notes](docs/API.md).

For a fresh installation with Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[embeddings,ingestion,analysis,dev]"
```

FFmpeg and ffprobe are required for audio/video ingestion. On Windows, native
model decoding also requires compatible shared FFmpeg libraries; see the
[verified AV setup](docs/AV_STATUS.md). CUDA is optional for retrieval; the pinned
local generator setup targets an NVIDIA GPU. This machine has a
verified RTX 4060 Laptop GPU with PyTorch 2.12.1+cu130.

## Import and ask about your files

```powershell
.\.venv\Scripts\python.exe run.py ingest "path/to/notes.pdf" "path/to/meeting.mp4" --output data/prepared/uploads --chunk-seconds 20
.\.venv\Scripts\python.exe run.py search --dataset data/prepared/uploads "What decision was made?" --method multimodal --device cuda
.\.venv\Scripts\python.exe run.py ask --dataset data/prepared/uploads "What decision was made?" --method multimodal --device cuda --provider local --output results/upload-answer.json
```

Omit `--device cuda` to use CPU for retrieval. Local answer generation uses the
installed CUDA runtime. The app and RAG pipeline unload the embedding model
before local generation, and stop their managed answer runtime before the next
GPU search. Avoid unrelated GPU jobs while using the app.

Local answers accept at most **30 seconds of audio and 60 seconds of video in
total per question**. Video is sampled at 1 frame/second, at most 768 pixels on
the long edge; separate audio uses the native audio encoder. Longer source files
can be imported as 20-second clips. The app and `ask` command pair matching audio/video segments and select whole
evidence groups within these limits, displaying added and omitted records.
Direct evaluation requests over the limits fail explicitly; sources are never
silently cropped.

| Input | Prepared representation |
|---|---|
| TXT, Markdown, CSV/TSV, JSON/JSONL, HTML | Overlapping text chunks; HTML scripts/styles excluded |
| PDF | Extracted page text plus page images, at most 1,800 pixels on the long edge |
| DOCX and PPTX | Text plus embedded images; original page/slide layout is not rendered |
| Common image files | Original visual content converted to a supported image format |
| Audio and video | Physical 20-second clips with source timestamps; silent video and separate audio |

The browser accepts up to 50 MB per file; the CLI permits up to 500 MB per input.
Source cards retain file/page/utterance/time information where available. A
validated citation ID establishes that the source was supplied, not that it
entails every generated claim. Imported files have no invented reference answers.

## Public evaluation data

| Collection | Prepared subset | Purpose |
|---|---|---|
| AMI via QMSum | 8 meetings, 425 transcript chunks, 46 evidence-labelled questions | Project-memory retrieval and answers |
| MultiModalQA | 100 questions; 997 texts, 98 tables, 920 original images | Text/table/image retrieval comparison and answer pilot |
| Daily-Omni | 100 questions, 94 clips, 188 separate audio/video records | Supplied-media audio/video answer ablations |

All selected MMQA images and Daily-Omni media are available locally. Daily-Omni
visual records point to full-duration silent copies, with audio supplied
separately. AMI raw recordings are not downloaded or timestamp-aligned; its
current evidence is the annotated transcript. No manual question curation is
required. To reproduce the public subsets and media acquisition:

```powershell
.\.venv\Scripts\python.exe run.py prepare ami --limit 100
.\.venv\Scripts\python.exe run.py prepare mmqa --limit 100 --seed 42
.\.venv\Scripts\python.exe run.py prepare daily-omni --limit 100 --seed 42
.\.venv\Scripts\python.exe run.py fetch-media --dataset data/prepared/mmqa --max-mb 150
.\.venv\Scripts\python.exe run.py fetch-media --dataset data/prepared/daily-omni --max-mb 4500
.\.venv\Scripts\python.exe run.py prepare-video-only --dataset data/prepared/daily-omni --output data/media/daily_omni/silent
.\.venv\Scripts\python.exe run.py validate --dataset data/prepared/daily-omni
```

MMQA downloads only selected image payloads from the official ZIP. Daily-Omni
requires the full 3.91 GB TAR even when selecting fewer clips. Reapply
`prepare-video-only` after rebuilding its prepared corpus. Manifests record the
selection, provenance, checksums, and acquisition status.

## Compare retrieval methods

```powershell
.\.venv\Scripts\python.exe run.py compare --dataset data/prepared/ami --methods bm25 dense-text --split dev --output results/ami-dev-comparison
.\.venv\Scripts\python.exe run.py compare --dataset data/prepared/mmqa --methods bm25 dense-text multimodal --device cuda --batch-size 1 --output results/mmqa-comparison
```

All methods use identical candidate IDs. BM25 and dense text use passages,
serialized tables, and image titles; **they do not include image OCR or generated
captions**. Native multimodal retrieval uses original image pixels, without image
titles, and fails if required media is missing. Interpret results as comparisons
of these implemented evidence representations.

The model revision is pinned to
`914f7f89142e33e77833254d9c9b90c3cef7303b`. Initial setup downloads roughly
1.5 GB of weights plus working space. Float32 is the default; float16 is
unsupported. Cached embeddings allow interrupted index builds to resume.
See [MMQA results](docs/MMQA_RESULTS.md) for the full comparison's current status,
and [measured project status](docs/STATUS.md) for existing AMI results.

EmbeddingGemma native text/image and real audio/video runtime checks have passed. These checks
establish compatibility, not retrieval quality:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_embeddings.py
.\.venv\Scripts\python.exe scripts/smoke_av_embeddings.py --device cpu --threads 2
```

## Start answer evaluation with five questions

Run a small pilot before expanding to the complete subset. Local inference has
no API quota; hosted-provider availability is account-dependent. API requests
are paced at five per minute by default, which does not state a free allowance.

```powershell
.\.venv\Scripts\python.exe run.py evaluate-answers --dataset data/prepared/ami --method bm25 --split dev --provider local --limit 5 --output results/ami-local-answer-pilot
.\.venv\Scripts\python.exe run.py evaluate-answers --dataset data/prepared/mmqa --method bm25 --provider local --limit 5 --output results/mmqa-local-answer-pilot
.\.venv\Scripts\python.exe run.py evaluate-answers --dataset data/prepared/daily-omni --media-mode audio --provider nvidia --limit 5 --output results/daily-audio-pilot
.\.venv\Scripts\python.exe run.py evaluate-answers --dataset data/prepared/daily-omni --media-mode video --provider nvidia --limit 5 --output results/daily-video-pilot
.\.venv\Scripts\python.exe run.py evaluate-answers --dataset data/prepared/daily-omni --media-mode combined --provider nvidia --limit 5 --output results/daily-combined-pilot
```

The Daily-Omni commands above retain the hosted provider because full benchmark
clips can exceed local Gemma's duration limits. A local AV comparison needs an
explicit, fixed duration-compatible subset and the same questions/evidence for
every provider. Cropping a benchmark clip changes its evidence and must not be
presented as the existing full-clip experiment.

Selection is the first five questions in the fixed prepared order, not a new
random sample. Predictions checkpoint after each question. Resume using the same
command and output directory: dataset fingerprints, model settings, selection,
and relevant code hashes must match. Use a new directory when changing the
configuration, including the question limit.

AMI uses lexical token F1 and ROUGE-L. MMQA uses custom answer-set precision,
recall, exact match, and token F1. Daily-Omni uses answer-choice accuracy and
coverage on the same questions across audio-only, silent-video-only, and combined
conditions. Daily-Omni supplies the relevant full clip, so it evaluates answering,
not corpus retrieval. These are custom pilots, not official leaderboard scores.
See the [evaluation protocol](docs/EVALUATION.md) for controls and limitations.

## Reports and checks

- [Measured status](docs/STATUS.md), [MMQA retrieval results](docs/MMQA_RESULTS.md), [answer pilot results](docs/GENERATION_RESULTS.md), and [AV verification](docs/AV_STATUS.md).
- [Project proposal](docs/PROJECT_PROPOSAL.md), [API notes](docs/API.md), and [AMI data notes](docs/AMI_DATA.md).
- `data/prepared/*/manifest.json` and `download_manifest.json`: selection and provenance.
- `results/`: the measured reports cited throughout these documents, including
  superseded attempts kept for provenance. Downloaded data, prepared datasets,
  model weights and caches are not in the repository; recreate them with the
  commands above.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .cache/pytest-verification
```

Only `corpus.jsonl` is indexed. Query files and raw annotations contain reference
answers and must never be included in the searchable corpus.

## Primary sources

- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2) and [Google multimodal guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers).
- [Google Gemma 4 E2B QAT GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) and [pinned llama.cpp runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b11429).
- [NVIDIA Nemotron 3 Nano Omni free prototype endpoint](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning).
- [AMI](https://groups.inf.ed.ac.uk/ami/corpus/), [QMSum](https://github.com/Yale-LILY/QMSum), [MultiModalQA](https://github.com/allenai/multimodalqa), and [Daily-Omni](https://github.com/Lliar-liar/Daily-Omni).

Preserve each dataset's attribution and licence requirements when redistributing
material. This project does not relicense downloaded sources. Files under
`results/` quote questions and source identifiers from these datasets and remain
subject to their original licences.
