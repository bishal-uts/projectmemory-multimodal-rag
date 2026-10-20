# Local and hosted answer generation

The default generator is local Gemma 4 E2B. NVIDIA's hosted Nemotron 3 Nano Omni
endpoint and Google's Gemini API remain available for comparison. EmbeddingGemma
2 remains the retrieval model; a separate generator reads retrieved evidence
and produces a cited answer. No paid account or billing setting is created.

## Local Gemma 4 E2B

`LocalGemmaGenerator` uses Google's **Gemma 4 E2B QAT Q4_0** GGUF and matching
multimodal projector, revision `675cff42a74c774d6cb76f76d8eacb49b48c9b93`.
The project pins the Windows CUDA build of **llama.cpp b11429**. Run
`scripts/setup_local_gemma.py` once with the project's Python to download and
SHA-256-verify about 5 GB of model/runtime archives. All artifacts and versions
are recorded in `.cache/local-gemma4/runtime.json`.
[Google's model repository](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf),
[runtime release](https://github.com/ggml-org/llama.cpp/releases/tag/b11429).

Select `--provider local` or **Gemma 4 E2B · local** in the app. No API key is
required or sent. The runtime starts only for an uncached generation request,
serves the alias `gemma-4-e2b` at `http://127.0.0.1:8766`, and exits with the app
or CLI process. Requests use numeric loopback only, disable proxies and
redirects, and have no cloud fallback. Missing files produce a setup error;
answering does not initiate a download. Existing NVIDIA/Gemini keys are retained.

Text, JPEG/PNG images, native WAV/MP3/FLAC audio, and silent video evidence are
accepted. PDFs must first be ingested into page text and images. Video is an
explicit sequence of timestamped JPEG frames sampled at **1 fps**, with at most
**768 pixels** on the long edge. Audio uses the native encoder. Per-request
limits are **30 seconds of audio**, **60 seconds of video**, and **64 combined
images/frames** across all evidence. These are conservative application limits;
the adapter rejects excess input without silent truncation. For longer sources,
import 20-second clips and retrieve fewer clips. The supplied full Daily-Omni
clips are not automatically shortened for local evaluation.

The runtime has a 16,384-token context with 4,096 output tokens by default and
thinking disabled. A JSON response schema and local citation validation enforce
the shared answer shape. Cache keys include model/projector/runtime identities,
adapter code, media processing, and complete evidence hashes. Changes in these
settings require a new evaluation output directory. Local inference has no
requests-per-minute throttle. EmbeddingGemma and the answer model take turns
using the GPU; unrelated GPU processes are outside this lifecycle control.

Protocol tests establish adapter behavior; live runtime and quality results
must be reported separately in the project's measured results.

## NVIDIA Nemotron 3 Nano Omni

Use the exact catalog ID
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`. This is the **Omni** variant with
text, image, audio, and video inputs. Other Nemotron model variants may have
different input support. NVIDIA lists a free prototype endpoint for this model;
the hosted service is governed by its API trial terms. Availability and quotas
depend on the account and can change. [Catalog and hosted example](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning).

Set `NVIDIA_API_KEY` locally (the adapter also accepts `NVIDIA_NIM_API_KEY` or
`NGC_API_KEY`). Never paste the credential into a prompt, commit it, or include
it in a command output. The adapter sends a Bearer authorization header to the
fixed `https://integrate.api.nvidia.com/v1/chat/completions` endpoint. It uses
Python's standard library, so no additional SDK installation is needed.

```python
from pathlib import Path
from projectmemory.nvidia import NvidiaGenerator

generator = NvidiaGenerator(cache_dir=Path(".cache/generation"))
answer = generator.answer(
    "What colour did the team choose?",
    [{"id": "meeting:chunk:1", "modality": "text",
      "text": "The team agreed to use blue for the remote control."}],
)
print(answer["answer"], answer["citations"])
```

The implementation uses the hosted API's documented `reasoning_budget`, with
1,024 reasoning tokens and 4,096 total output tokens by default. It requests
JSON in the prompt and validates the response locally. It does not assume that
the hosted API supports self-hosted structured-generation or video-preprocessing
flags. Pending HTTP 202 responses are polled through the documented status
endpoint. [Hosted request schema](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-infer),
[status endpoint](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-statuspolling).

If the provider returns malformed JSON or invalid citations, the adapter makes
at most one formatting retry with the same question and evidence. It does not
send back the failed answer or private reasoning. A failed second response
stops the operation; a successful result records the retry count and combined
token usage. This is separate from bounded retries for transport errors.

Local JPEG/PNG images, WAV/MP3/FLAC audio, and MP4 video are encoded into media
content parts. This adapter's default is **8 MB of raw media per request**, configurable
up to 12 MB; this is our application limit, not a claim about the provider's
maximum. Compress or segment larger clips before generation. PDF pages should
be rendered to images or extracted as text. The model card describes video
inputs up to two minutes and English language support. Do not assume a video
request includes the soundtrack: for an audio-plus-video experiment, provide
the separate audio evidence as well. [Model card](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning/modelcard),
[media input examples](https://docs.nvidia.com/nim/vision-language-models/1.7.0/examples/nemotron-3-nano-omni-30b-a3b-reasoning/api.html).

## Gemini alternative

`GeminiGenerator` defaults to `gemini-3.8-flash`. It accepts
`GEMINI_API_KEY` or `GOOGLE_API_KEY` and uses `generateContent` with a JSON
response schema. The model supports text, images, audio, video, and PDFs.
`gemini-3.5-flash-lite` can be explicitly selected for a separate comparison;
the code never switches models silently. [Model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash),
[REST API reference](https://ai.google.dev/api/generate-content).

Google lists standard free-tier access for eligible models/accounts. Check the
current model row and the project's tier before running an evaluation; an API
key alone does not prove that requests are free. Free-tier data handling differs
from paid use, so use the public benchmark corpus for these experiments.
[Pricing and data handling](https://ai.google.dev/gemini-api/docs/pricing).

Gemini sends up to 12 MB of raw media inline per request. Larger files use the
Files API, with a 100 MB application limit per file and 150 MB per request.
Uploaded files are deleted in a `finally` cleanup after inference; if deletion
fails, successful answers include a warning. Google documents automatic expiry
after 48 hours. [Files API](https://ai.google.dev/gemini-api/docs/files).

## Grounding, reproducibility, and limits

- All generators accept `answer(question, evidence, options=None)` and return
  `answer`, `answer_items`, `citations`, `abstained`, `answer_label`, `sources`,
  `model`, `model_version`, `usage`, and `cache_hit`. Multiple-choice questions
  use two to four options and an A–D label.
- Only evidence IDs, modalities, titles, text, and selected media are sent.
  Gold answers, relevance labels, and arbitrary dataset metadata are excluded.
  Citation validation rejects invented IDs, but **does not prove that a cited
  source actually supports the answer**; citation entailment needs separate
  evaluation. An empty evidence list abstains locally without an API request.
- Full input/media hashes, model IDs, prompts, and generation settings identify
  the local response cache. Responses are validated again on cache hits.
  Credentials and raw media are not saved in this cache. Cached outputs and
  local source paths are stored, so clear or protect the cache as appropriate.
  A cached answer records the original model version and usage; `api_called`
  is false on a cache hit.
- The default API throttle is five generation requests per minute. It is not
  an assertion about free-tier limits. At most two retries follow transient
  errors; long `Retry-After` delays stop the command. Provider errors are
  redacted, and failed or malformed answers are not cached as successful results.
- Benchmark results must identify the generator, retrieval method, evidence
  representation, split, and prompt configuration. Keep the generator fixed
  when comparing retrievers. A public benchmark may appear in model training or
  published evaluation; do not claim it is a contamination-free holdout.

These adapters have mocked protocol tests. A public AMI transcript question
completed successfully through both NVIDIA and Gemini on 15 October 2026, with
two valid evidence citations in each response. Both results also displayed in
the local app. That verifies integration; individual smoke tests are not
benchmark scores. See [measured answer pilots](GENERATION_RESULTS.md) for
completed evaluations, provider failures, and partial conditions.
