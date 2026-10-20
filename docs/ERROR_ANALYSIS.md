# Fixed-sample answer error analysis

This is **AI-assisted qualitative inspection**, not independent human annotation
or an estimate of entailment accuracy. The inspected sample is the **first ten
questions in the fixed prepared MMQA order**, selected before this inspection;
examples were not selected because they succeeded or failed. Their IDs match the
first ten predictions in the completed [100-question local BM25 run](../results/final-mmqa-local-bm25/report.json).

Links to prepared questions, corpus records and original images below require
the public data preparation steps in the README. Those bulk assets are excluded
from the distributable archive; saved predictions and reports are included.

The run uses BM25 top-five evidence and the pinned Gemma 4 E2B QAT Q4_0 generator.
For each example, the inspection compares the reference answer, saved generated
answer, retrieved source records, and annotated support IDs. The original images
were also viewed where needed: the dataset-labelled Manuel Lazzari portrait and
the *Fantastic Beasts* book cover. No new generation, prompt tuning, retrieval
changes, or score corrections were performed for this audit.

The full run scored **14/100 exact answer-set matches**, **63/100 formally
answered**, and **9 invalid model responses** after the prescribed formatting
retry. Invalid responses count as unanswered and incorrect under the recorded
evaluation policy. The fixed first-ten sample has four exact matches, ten
`abstained=false` outputs, and no invalid responses. All ten answers were reused
from validated identical-input generation caches; they are not ten additional
inference calls. The sample is not representative enough to replace the full-run
results or rank generators.

## Fixed first-ten inspection

“Support coverage” means **annotated supporting source IDs retrieved / annotated
supporting source IDs**, not claim-level support. Nine examples contain every
annotated source; the eighth contains one of two. A missing annotation match does
not necessarily mean the answer is unavailable, and a matching citation does not
prove every generated sentence is correct. Questions and long predictions below
are abbreviated; the saved predictions contain the complete output.

| # / Query ID | Short question | Reference | Saved prediction | Support coverage | Evidence-backed interpretation |
|---|---|---|---|---|---|
| 1 — `mmqa:024a2b24f4f7da220a879b0479ef1166` | Which years had Swingfly singles from the album containing “Winner”? | 2006; 2008; 2009 | `answer_items`: 2008 | **1/1** | **Required answers omitted despite complete retrieval.** The Swingfly table identifies *God Bless the IRS* and lists matching singles in all three years. The model returns only the year of “Winner”, failing the album-level aggregation. Its cited table is correct, but that does not establish completeness. |
| 2 — `mmqa:08e66ea55258dfb1e76421a0cd580108` | What type of piercing does Manuel Lazzari have? | ear | Prose says the evidence does not state it; `answer_items`: “abstained”; `abstained=false` | **1/1** | **Visual evidence missed and abstention fields inconsistent in meaning.** The supplied portrait visibly shows ear jewelry. The model gives a non-answer despite receiving the required image, then labels that response as answered. The formal coverage metric therefore counts it as covered. This is not an appropriate missing-evidence abstention. |
| 3 — `mmqa:0964e81cc1919b2bfcecc9dc976d93ec` | Which Taylor Swift country number-one came from the album that peaked at number five? | Should’ve Said No | `answer_items`: Should’ve Said No | **2/2** | **Correct answer; partial citation coverage.** The discography passage provides the album’s number-five peak, and the song passage provides the country number-one result. Both were retrieved, but only the song passage was cited. Exact match is one while gold citation recall is one-half. |
| 4 — `mmqa:0b037ac2ba0c3eefbeb083e481ad2848` | Did *Sometimes They Come Back… for More* come out in 2000? | No | Prose: “No … released in 1998”; `answer_items` contains a sentence about the director and 1998 release | **1/1** | **Supported answer prose, wrong answer-item granularity.** Both the film passage and the annotated Chase Masterson table give 1998. The sentence in `answer_items` does not normalize to “No”, so exact match is zero. The cited film passage is useful evidence despite not being a gold support ID, illustrating the limits of gold-source citation precision. |
| 5 — `mmqa:0cfc13b8cb739d42a29ed76f1bd83b5b` | Which is deeper: Lake Dibawah or Lake Toba? | Lake Toba | `answer_items`: Lake Toba; prose first says Dibawah is deeper, then corrects itself | **1/1** | **Correct extracted answer with contradictory prose.** The table gives 309 m for Dibawah and 529 m for Toba. Exact match rewards the final answer item but does not penalize the opening false comparison. A perfect score on this item does not certify the complete response. |
| 6 — `mmqa:0d21c0f0123e225fd3341324ff69914b` | Is Corey Davis older or younger than Titus Davis? | younger | `answer_items` contains a sentence stating Corey is younger, plus both birth dates | **2/2** | **Supported relation, exact-match representation failure.** The Titus passage explicitly identifies Corey as his younger brother; the two birth dates agree. The generated sentence expresses the reference relation, but the whole sentence is compared with the single reference word, producing exact match zero. No semantic rescore was substituted. |
| 7 — `mmqa:119c9a9bf38ca9f84a3900d7c7fd1fe5` | Which 2016 ceremony listed Shane Pow in Top 10 Most Popular Male Artistes? | 22nd Star Awards Ceremony | `answer_items`: 22nd Star Awards Ceremony | **1/1** | **Correct table lookup.** The retrieved and cited Shane Pow accolades table contains the matching year/category row. The generated answer item matches the reference. |
| 8 — `mmqa:12b5572cf9bbb103a607a9e0b880c94a` | Who wrote the *Blues for Greeny* track originally by Little Willie John? | Mertis John Jr. | `answer_items`: Peter Green; prose mentions both names | **1/2** | **Partial retrieval plus an answer-selection error.** The missing gold song passage names Mertis John Jr., but the retrieved track table already gives the same writer for “Need Your Love So Bad”. The companion album passage describes a tribute to Peter Green; it does not make him the writer of this track. Missing one support alone cannot explain the wrong answer item. |
| 9 — `mmqa:16db0b1e30934e4c5a7acb33d0274b5e` | Who wrote the fictional textbook *Fantastic Beasts and Where to Find Them*? | Newt Scamander | Prose identifies Newt; `answer_items` contains two explanatory sentences; prose additionally says he published the book | **1/1** | **Correct author entity, non-exact answer items, and an unsupported additional claim.** The cited passage identifies the fictional writer; the cover shows his name. Neither establishes the generated publisher claim. The verbose two-item answer fails the fixed exact-set metric, while the added claim requires separate qualitative scrutiny. |
| 10 — `mmqa:1afabde176377143c6118e3828837ce9` | Which drama starred Lee Jong-suk and Kim Woo Bin? | School 2013 | `answer_items`: School 2013 | **1/1** | **Correct answer with additional corroborating evidence.** The cited *School 2013* passage names both actors. The additional cited filmography supports Lee Jong-suk’s participation but is not an annotated gold support, so gold citation precision is one-half despite the direct support in the first citation. |

## What these examples establish

- **Retrieval and generation failures are different.** Example 1 has the needed
  table but omits two answers. Example 2 receives the needed image but does not use
  its visible detail. Example 8 lacks one annotated source while already holding
  an answer-bearing table; its wrong writer should not be attributed solely to
  a retrieval miss.
- **Exact answer-set match measures the structured answer items.** Examples 4
  and 6 express the reference answer in prose but lose exact match because their
  items are full sentences. Example 9 combines that issue with an unsupported
  addition. These observations explain failures without changing the frozen
  metric or replacing the reported 14% with an informal “correctness” score.
- **An exact match can coexist with a bad sentence.** Example 5’s contradictory
  opening is invisible to its otherwise correct answer item. Example 3 also
  shows that a correct answer need not cite every retrieved gold support.
- **Coverage is the declared abstention flag, not a semantic judgment.** Example
  2 shows a non-answer with `abstained=false`. The reported 63% full-run coverage
  should therefore be described as formal answer coverage, not a guarantee that
  63 questions received useful substantive answers.
- **Source-ID overlap is not entailment evaluation.** Examples 4 and 10 cite
  relevant sources outside the gold list, while example 9 cites a valid gold
  source yet adds an unsupported publisher claim. Citation validation and
  annotation overlap cannot replace checking the generated claims.

There is **no appropriate abstention or exhausted-format failure among these ten
preselected outputs**. The audit does not substitute a later example to fill
those categories. The nine full-run formatting failures remain visible in the
saved predictions and aggregate report; they are not nine manually adjudicated
hallucinations. Infrastructure failures are excluded from that failure policy
and would interrupt the run instead of becoming incorrect answers.

Possible future improvements include more reliable extraction into short
`answer_items`, semantic abstention checks, and claim-level citation review.
Those are future experiments: they were not applied to this completed run.
The native/dense comparison should use its own completed reports and the same
frozen generator protocol; this inspection does not imply results for those
conditions.

## Reproducibility and inspected evidence

- [Prepared questions](../data/prepared/mmqa/queries.jsonl), [prepared source corpus](../data/prepared/mmqa/corpus.jsonl), and [saved predictions](../results/final-mmqa-local-bm25/predictions.jsonl).
- [Saved BM25 rankings](../results/mmqa-comparison/bm25.json), [replay provenance](../results/final-mmqa-local-bm25/retrieval_source.json), and [run configuration](../results/final-mmqa-local-bm25/run.json).
- Corpus fingerprint: `2ac934df42e9fbb82ba4abeb7576470af2735daba665642046db0d397a90dd4f`.
- Full 100-question fingerprint: `e4c8eb70fd4d3d5143426d55dfd367a193d560d59459a9d0b48d0fcad66d2016`.
- Viewed original images: [dataset-labelled portrait](../data/media/mmqa/images/897f52c139bd67b0453648ef089f753b.jpg) and [book cover](../data/media/mmqa/images/ecd35368d7c6b8d6bc5fc8abec916292.JPG).
- Key text/table evidence IDs, also present in the corpus: `mmqa:table:ef0eb37ea4210dfb249c1662e97c6084` (Swingfly), `mmqa:table:facf9e1204475d1e5ca3487e5c83303c` (lake depths), `mmqa:table:57725e115967972b4ae91e1d85398d4c` (track writers), and `mmqa:text:800c2f54328e963bfbf4773211c69951` (fictional textbook writer).

Inspected on 15 October 2026. This document preserves the fixed sample and the
original metrics; its interpretations are qualitative, AI-assisted, and subject
to review against the linked evidence.
