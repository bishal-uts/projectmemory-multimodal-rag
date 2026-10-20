# Local ProjectMemory app

The browser interface connects the prepared collections, retrieval, local/API
generation, local file ingestion, and measured evaluation reports in one place.
It uses Python's standard library HTTP server and has no frontend build step.

Start it from the project directory:

```powershell
.venv\Scripts\python.exe run.py serve
```

Open the printed local URL, normally `http://127.0.0.1:8765`. The app binds only
to localhost; it is a development and research interface, not a publicly
deployed service. Stop it with Ctrl+C in the terminal that started it.
The owned local answer runtime also stops when the app exits.

Local Gemma needs a one-time explicit setup:

```powershell
.venv\Scripts\python.exe scripts/setup_local_gemma.py
```

This installs Gemma 4 E2B QAT Q4_0 and llama.cpp b11429 under the project cache.
It downloads about 5 GB and verifies the pinned files. Once installed, the
runtime starts automatically when a question requires inference. The app never
downloads a model as part of answering a question.

1. Choose a prepared collection and retrieval method. BM25 works immediately
   without a model or API key. Optionally restrict retrieval to a meeting or
   uploaded source file.
2. Enter a question and choose **Search sources** to inspect the evidence.
   EmbeddingGemma can take time to load or build an index on its first search.
3. Choose **Gemma 4 E2B · local**, the default, then **Generate answer**. Evidence
   stays on this computer and no API key is needed. Status distinguishes a
   missing installation, an installed model, and a running model. The first
   uncached answer includes model startup time. NVIDIA and Gemini remain
   selectable; those options send selected evidence to their API. Their answer
   button requires a configured key in the local `.env` file. Use **Refresh results**
   after adding a previously missing key. Restart the server after replacing or
   removing an existing key, because the process keeps its configured credential
   and active generator until it exits. Keys never
   appear in browser inputs or API status responses.
4. Inspect the returned source cards, native media previews, page/utterance/time
   locations, and answer citations. A citation ID check does not prove that its
   source supports the generated claim.

Local Gemma accepts up to **30 seconds of audio and 60 seconds of video total
per question**, across all retrieved clips. Video uses timestamped frames at
1 fps, with a maximum long edge of 768 pixels; audio is supplied separately to
the native audio encoder. Imported AV uses 20-second clips. The application
pairs audio/video from the same segment and selects whole evidence groups within
the limits, displaying added/omitted records. Direct benchmark requests retain
their prescribed evidence and reject excessive duration. Neither path silently
crops sources. Sparse video frames can miss brief actions, and successful
input processing does not establish answer accuracy.

Imports accept supported documents, images, audio, and video up to **50 MB per
file**. Choose several files to import them sequentially. They are saved under
`data/uploads/` and prepared into the **Your uploaded files** collection. Text is
chunked, PDF pages retain images and text, Office documents retain text and
embedded images, and audio/video is segmented into clips. The upload control
imports into a fixed workspace collection; it does not accept output paths.
File extraction requires the ingestion dependencies described in the project
guide. Images and AV have little searchable text in the BM25 baseline; use
multimodal retrieval for their original content. Audio/video collections select
multimodal retrieval automatically; changing back to text/keyword search displays
a limitation notice. Multiple-choice benchmark questions are omitted from the
free-text example buttons because their choices and supplied clip belong to the
separate evaluation protocol.

Retrieval defaults to CPU. Select CUDA only when the local installation supports
it and no competing GPU benchmark is running. The app serializes indexing,
search, generation, and ingestion, and keeps only one dense retriever active
to avoid loading several embedding models into GPU memory. Local Gemma uses
the installed CUDA runtime. Before a GPU retrieval the app stops its owned
answer process; before local generation it unloads dense embedding models.
This does not coordinate with unrelated terminal processes.

The results tables read existing JSON reports under `results/`; they do not
invent or launch experiments. Retrieval and answer quality have separate tables
showing actual saved metrics and sample sizes. Only compare runs using the same
corpus, questions, split, evidence count, and generator. Hover a row to see the
report filename and identifying fingerprints or model. Answer coverage counts
declared non-abstentions; this can include semantic non-answers. Reference scores and gold-source citation precision are useful
proxies, not a complete factuality assessment.

The local server validates Host/Origin and a per-session request token for
state-changing requests. Media previews resolve evidence IDs through a selected
collection and only read files under this project's `data/` directory. Original
documents are offered as downloads. These controls support local use; they do
not make this development server suitable for public hosting.
