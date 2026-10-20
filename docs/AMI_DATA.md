# AMI + QMSum: project-memory demonstration collection

The adapter prepares a usable transcript retrieval dataset without manual question
writing. It downloads eight small QMSum Product meeting JSON files. It does not
download audio, video, images, documents, or model weights.

```powershell
$env:PYTHONPATH = "src"
python -m projectmemory.cli prepare ami --output data/prepared/ami --limit 100
python -m projectmemory.cli validate --dataset data/prepared/ami
python -m projectmemory.cli evaluate --dataset data/prepared/ami --method bm25 --split dev --output results/ami-dev-bm25.json
```

The Python API accepts `prepare(Path("data/prepared/ami"), limit=0)` to include all
specific questions; positive limits select a deterministic seeded subset without
shrinking the retrieval corpus. `chunk_utterances=10` is the default chunk size.
The adapter uses Python 3.11 standard-library dependencies only.

## Source and split provenance

| Local split | QMSum source split | Meetings | Team |
| --- | --- | --- | --- |
| `dev` | `Product/val` | ES2006a, ES2006b, ES2006c, ES2006d | ES2006 |
| `test` | `Product/test` | ES2011a, ES2011b, ES2011c, ES2011d | ES2011 |

The prepared collection contains **425 chunks and 46 specific questions** (24 dev,
22 test), plus 8 whole-meeting summary questions kept separately. Raw JSON files
total 643,739 bytes, so the transcript baseline is inexpensive to reproduce.

The adapter pins [QMSum revision 83d7768](https://github.com/Yale-LILY/QMSum/tree/83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e),
records source URLs and SHA-256 checksums in `manifest.json`, and preserves the
existing team split. These eight meetings are a small project subset, not the full
QMSum benchmark. Tune on `dev`; leave `test` for the final experiment.

## Generated files and evaluation meaning

- `corpus.jsonl`: only source transcript utterances grouped in consecutive chunks.
  Each chunk retains its meeting ID and zero-based inclusive utterance bounds.
- `queries.jsonl`: QMSum specific questions, reference answers, original evidence
  spans, and IDs of every chunk overlapping a gold span.
- `generation_queries.jsonl`: whole-meeting questions for later answer generation.
  They have no localized gold evidence spans and are excluded from retrieval scores.
- `manifest.json`: source revisions, checksums, split counts, chunking and sampling.
- `download_manifest.json`: optional AMI media locations and observed availability.
- `raw/`: original QMSum JSON files. These contain labels as well as source text.
  Never recursively index this directory or either query file.

QMSum's [schema documentation](https://github.com/Yale-LILY/QMSum#dataset) supplies
reference summaries and utterance spans. The adapter derives **chunk-level**
relevance from those spans. It neither invents timestamps nor labels visual
evidence. Recall@k measures how many overlapping chunks were recovered; Hit@k
measures whether at least one was recovered. Long gold spans can cover many chunks,
so a low Recall@5 can coexist with finding useful evidence.

Queries refer to a known meeting, often implicitly. The metadata field
`candidate_source_ids` names that meeting. For the original task setting, evaluate
against chunks from that meeting. Searching all eight meetings is a harder derived
task and must be labelled accordingly. This subset does not establish the benefit
of image/video retrieval over transcripts or provide official leaderboard scores.

## Optional original media: verified availability

On **15 October 2026**, read-only HTTP requests verified every selected meeting's
`audio/<meeting>.Mix-Headset.wav` and `video/<meeting>.Corner.avi` with HTTP 200 and
Content-Length. Exact URLs and sizes are included in the manifest. These are one
mixed audio track and one compressed camera view per meeting, approximately **886
MB combined** across all eight, before images/documents and derived clips.

The [official download page](https://groups.inf.ed.ac.uk/ami/download/) also offers
other camera and microphone streams, annotations, slides, shared documents and
whiteboard data. Directory inspection found these qualifications:

- All eight have a slide browser at `slides/`; its index is a frameset, not a ZIP
  archive or a direct image list. Follow the browser's `left.html` and slide pages
  to the underlying image/OCR files when adding slide ingestion.
- ES2006a–d and ES2011a list whiteboard stroke videos and JPEG previews.
- ES2011b–d list `NoUseOfWhiteboard.txt` and stroke XML, with no preview image.
- ES2006a–d expose shared `.ppt` and `.doc` files. ES2011a–d `shared-doc/` directories
  returned HTTP 404. Do not assume that availability-table flags guarantee files.

The manifest accurately preserves those missing resources. It does not insert
placeholder images, empty media items, or gold summaries into the searchable
corpus. Downloading optional media does not automatically ingest it: subsequent
work must decode/segment it and connect source IDs. QMSum utterance indices need
alignment with AMI timed annotations before creating timestamp citations. The
manifest links the official manual annotation archive for that later step.

## Attribution

The [AMI project](https://groups.inf.ed.ac.uk/ami/corpus/) releases its signals and
transcription, and some annotations, under
[CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml). Retain attribution
and identify derived transcripts/chunks. QMSum's repository includes an
[MIT license](https://github.com/Yale-LILY/QMSum/blob/main/LICENSE); retain its notice
alongside the underlying AMI attribution when redistributing relevant material.
The prepared manifest records provenance but is not a replacement for license
notices when publishing a dataset package.

Cite the AMI corpus (Carletta et al.) and Zhong et al., *QMSum: A New Benchmark for
Query-based Multi-domain Meeting Summarization*, NAACL 2021, in the project report.
