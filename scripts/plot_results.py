r"""Render the measured MMQA retrieval comparison without loading any models.

Usage: .venv\Scripts\python.exe scripts/plot_results.py
Requires matplotlib. Reads the three existing retrieval reports and checks that
they have identical corpus/question fingerprints and the same cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bm25", "dense-text", "multimodal")
LABELS = ("BM25", "Dense text", "Native\nmultimodal")
COLORS = ("#64748b", "#2563eb", "#0f766e")


def load_measurements(folder: Path) -> tuple[list[float], list[float], int, int, int]:
    reports = [json.loads((folder / f"{method}.json").read_text(encoding="utf-8"))
               for method in METHODS]
    signatures = {(r["corpus_fingerprint"], r["query_fingerprint"], r["k"])
                  for r in reports}
    if len(signatures) != 1 or reports[0]["k"] != 5:
        raise ValueError("Plot requires matching corpus, questions, and k=5 across reports")
    question_ids = [{row["id"] for row in report["queries"]} for report in reports]
    if any(ids != question_ids[0] for ids in question_ids[1:]):
        raise ValueError("Reports contain different scored question IDs")
    overall = [report["metrics"]["recall@5"] * 100 for report in reports]
    images = []
    image_question_ids = []
    for report in reports:
        recalls = {}
        for row in report["queries"]:
            gold_images = {source for source in row["relevant_ids"]
                           if source.startswith("mmqa:image:")}
            if gold_images:
                retrieved = {hit["id"] for hit in row["hits"][:5]}
                recalls[row["id"]] = len(retrieved & gold_images) / len(gold_images)
        if not recalls:
            raise ValueError("Reports contain no MMQA gold image sources")
        images.append(mean(recalls.values()) * 100)
        image_question_ids.append(set(recalls))
    if any(ids != image_question_ids[0] for ids in image_question_ids[1:]):
        raise ValueError("Reports contain different image-question subsets")
    return overall, images, len(question_ids[0]), len(image_question_ids[0]), reports[0]["corpus_size"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/mmqa-comparison")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    overall, images, n_questions, n_image_questions, corpus_size = load_measurements(args.results_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator, PercentFormatter
    except ImportError as exc:
        raise SystemExit("This optional figure needs matplotlib; no model dependencies are required.") from exc

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.7), sharey=True)
    figure.subplots_adjust(left=0.09, right=0.97, top=0.75, bottom=0.25, wspace=0.22)
    figure.suptitle("Higher retrieval recall with native images in this pilot", x=0.09,
                   y=0.96, ha="left", fontsize=17, fontweight="bold", color="#0f172a")
    figure.text(0.09, 0.89, "EmbeddingGemma 2 | MultiModalQA development pilot",
                fontsize=12, color="#475569")
    for axis, values, title in zip(axes, (overall, images), (
            f"All supporting sources\n{n_questions} questions",
            f"Gold image sources only\n{n_image_questions} questions")):
        bars = axis.bar(range(3), values, color=COLORS, width=0.60, zorder=3)
        axis.set_ylim(0, 100)
        axis.set_xticks(range(3), LABELS)
        axis.set_title(title, pad=15, loc="left", color="#0f172a")
        axis.yaxis.set_major_locator(MultipleLocator(20))
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
        axis.tick_params(axis="both", length=0, pad=8, colors="#334155")
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#cbd5e1")
        axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values],
                       padding=6, fontsize=12, fontweight="bold", color="#0f172a")
    axes[0].set_ylabel("Recall@5", labelpad=10, color="#334155")
    figure.text(0.09, 0.115,
                f"Custom development pilot: pooled candidate corpus of {corpus_size:,} sources; the same questions and k=5 for all methods.",
                fontsize=9, color="#475569")
    figure.text(0.09, 0.076,
                "Text baselines use image titles, without OCR/captions; native mode uses original images. Retrieval, not answer accuracy.",
                fontsize=9, color="#475569")
    output = args.output_dir or args.results_dir
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        destination = output / f"retrieval-comparison.{suffix}"
        figure.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.20)
        print(destination.resolve())
    plt.close(figure)


if __name__ == "__main__":
    main()
