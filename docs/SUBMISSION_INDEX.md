# ProjectMemory deliverables

The technical project consists of a local application, reproducible evaluation,
and supporting documentation. The assessment report (2,806 words, 11-point type)
and the seven-slide, 5-minute-10-second presentation follow the supplied rubric;
they are submitted separately and are not part of this repository.

Final verification: **298 tests and 17 subtests passed**, with one Windows
symlink-permission skip. The live AV walkthrough, source playback and recorded
response are complete. Model and preview processes were stopped after testing.

| Deliverable | Location |
|---|---|
| Application setup and commands | [README](../README.md) |
| Technical report | [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) |
| Frozen expanded experiment protocol | [FINAL_EXPERIMENT_PROTOCOL.md](FINAL_EXPERIMENT_PROTOCOL.md) |
| Measured comparisons and uncertainty | [Final comparison](../results/final-analysis/comparison.md) |
| Fixed-sample qualitative inspection | [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) |
| Rehearsal and live demonstration | [DEMO_GUIDE.md](DEMO_GUIDE.md) |
| Observed AV demo | [Saved response](../results/demo-av-live.json), [browser screenshot](../results/demo-av-browser.jpg) |
| Installed model/runtime and limits | [LOCAL_MODEL.md](LOCAL_MODEL.md) |
| Observed package versions | [TESTED_ENVIRONMENT.json](TESTED_ENVIRONMENT.json) |
| Exportable measured figures | [Answer comparison](../results/final-analysis/answer-comparison.png), [paired differences](../results/final-analysis/paired-differences.png) |

The final archive is generated with:

```powershell
.\.venv\Scripts\python.exe scripts/package_submission.py --write
.\.venv\Scripts\python.exe scripts/package_submission.py --verify dist/ProjectMemory-submission.zip
```

It contains application source, tests, scripts, documentation and explicitly
selected public-data evaluation reports. It excludes `.env`, credentials, user
uploads, datasets, model weights, caches and the virtual environment. A fresh
machine must obtain dependencies, models and public datasets using the included
instructions. Packaged report paths are made relative; the inventory retains
both original and packaged SHA-256 hashes.

The archive is an optional offline snapshot; this repository is the public
project code.

## Claims that the submission should preserve

- This is an application and evaluation project, not a newly trained model.
- EmbeddingGemma 2 retrieves evidence; Gemma 4 E2B generates answers.
- The 100-question MMQA experiment is a sampled development experiment. Only
  the separate 22-question AMI retrieval evaluation is labelled held-out.
- The 20-question AV study uses supplied clips and choices, with explicitly
  documented terminal audio-sample normalization. It does not measure AV retrieval.
- Cached generations are recorded; timings are not controlled speed benchmarks.
- Exact match, model-declared coverage and gold citation overlap are imperfect
  proxies. The qualitative inspection is AI-assisted, not independent human
  annotation or a quantified factuality study.
- Report low scores and invalid outputs alongside successful demonstrations.
