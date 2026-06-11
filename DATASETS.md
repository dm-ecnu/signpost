# Datasets

The paper (`sections/05_experiments.tex`, `\label{sec:datasets}`) evaluates on
six workloads: five domain corpora the paper groups under the **UltraDomain**
umbrella — **Agriculture**, **Medical**, **Novel**, **Legal**, **Mix** — plus
the **MuSiQue** multi-hop benchmark used for answer quality only.

> **Provenance honesty note.** The paper prose calls all five domain corpora
> "UltraDomain corpora". The dataset-preparation code
> (`scripts/prepare_datasets.py`) shows the actual upstreams: **Agriculture,
> Legal, and Mix** are pulled from the `TommyChien/UltraDomain` HuggingFace
> dataset, while **Medical and Novel** are pulled from the
> `GraphRAG-Bench/GraphRAG-Bench` HuggingFace dataset (the GraphRAG-Bench
> Medical/Novel corpus + question splits). They are documented per their real
> source below. Per-corpus row/document/question counts are produced by that
> script and recorded in `datasets/manifest.json` after a run.

## Citation keys

The dataset sources are **not cited with BibTeX keys** in
`references.bib` — neither UltraDomain nor MuSiQue nor GraphRAG-Bench has an
entry in the paper's bibliography (verified by grep). The canonical references
below are given for reproducibility, not copied from the paper's `.bib`. The one
related key that *does* exist in `references.bib` is `trivedi2023ircot` (IRCoT,
Trivedi et al. 2023) — that is the IRCoT method paper, **not** the MuSiQue
dataset paper (MuSiQue is Trivedi et al., TACL 2022). Do not conflate them.

See `CITATIONS.bib` for the resolution status of every key.

## Per-dataset detail

| Dataset | Upstream (as fetched by code) | Type | Paper scale | Obtain |
|---|---|---|---|---|
| Agriculture | `TommyChien/UltraDomain` | Long-document domain QA | 12 docs, 9,156 chunks, 100 Qs, **75,088 objects** | HF resolve URL below |
| Legal | `TommyChien/UltraDomain` | Long-document domain QA | sensitive case in RQ5 | HF resolve URL below |
| Mix | `TommyChien/UltraDomain` | Mixed shorter sources | 61 docs, 1,287 chunks, 130 Qs, **28,966 objects** | HF resolve URL below |
| Medical | `GraphRAG-Bench/GraphRAG-Bench` | Domain QA corpus + questions | reported in Table `tab:quality` | HF resolve URLs below |
| Novel | `GraphRAG-Bench/GraphRAG-Bench` | Domain QA corpus + questions | reported in Table `tab:quality` | HF resolve URLs below |
| MuSiQue | MuSiQue (Trivedi et al., TACL 2022) | Multi-hop QA benchmark | answer-quality only; `S_LLM`=4.8, J@7=36.0 for Signpost | see below; expected at `datasets/processed/musique/` |

"Objects" is the retriever-candidate set `|I_C| + |I_G|` (chunk + graph
objects), per Table `tab:workloads`. The two corpora analyzed in depth are
**Agriculture** (only 12 documents but 75,088 objects — stresses vertical
navigation and provenance) and **Mix** (28,966 objects — stresses local lookup
and semantic association).

### UltraDomain (Agriculture, Legal, Mix)

- **Official name:** UltraDomain (long-document, multi-domain RAG benchmark).
- **One line:** Long, domain-specific documents with open-ended questions whose
  answers require navigating within long sources.
- **How to obtain:** HuggingFace dataset `TommyChien/UltraDomain`. The code
  fetches per-domain JSONL files directly:
  - `https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/agriculture.jsonl`
  - `https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/legal.jsonl`
  - `https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/mix.jsonl`
  (see `scripts/prepare_datasets.py`, `ULTRADOMAIN_FILES`). Each row carries a
  `context` (the document), `input` (the question), and `answers`.
- **License:** Not asserted by the upstream dataset card at time of writing;
  treat as research-use, check the HuggingFace dataset page before redistribution.
- **Citation key:** none in `references.bib` (no UltraDomain entry).
- **Splits/scale used:** Agriculture 12 docs / 9,156 chunks / 100 Qs / 75,088
  objects; Mix 61 docs / 1,287 chunks / 130 Qs / 28,966 objects (Table
  `tab:workloads`). Legal/Mix/Agriculture all run through `prepare_ultradomain`,
  which deduplicates rows by `context_id` into documents.

### GraphRAG-Bench (Medical, Novel)

- **Official name:** GraphRAG-Bench.
- **One line:** A GraphRAG evaluation benchmark; the paper uses its Medical and
  Novel domain corpora plus their question splits.
- **How to obtain:** HuggingFace dataset `GraphRAG-Bench/GraphRAG-Bench`. The
  code fetches corpus + question JSON files directly:
  - `https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Corpus/medical.json`
  - `https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Questions/medical_questions.json`
  - `https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Corpus/novel.json`
  - `https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Questions/novel_questions.json`
  (see `scripts/prepare_datasets.py`, `GRAPHRAG_BENCH_FILES`,
  `prepare_graphrag_bench`).
- **License:** Check the upstream HuggingFace dataset page; not asserted here.
- **Citation key:** none in `references.bib`.
- **Splits/scale used:** quality numbers reported in Table `tab:quality`. The
  public repo currently exposes only the Medical and Novel configs (the thesis
  computer-science subset is not in the public splits — noted in the script's
  manifest).

### MuSiQue (multi-hop)

- **Official name:** MuSiQue (Multihop Questions via Single-hop question
  Composition).
- **One line:** A 2–4 hop multi-hop QA benchmark whose answers require composing
  several supporting passages; used by the paper for explicit multi-hop answer
  quality only.
- **How to obtain:** Canonical source is the AllenAI MuSiQue release,
  `https://github.com/StonyBrookNLP/musique` (data download script + Zenodo
  archive linked there); the dataset accompanies Trivedi et al., *MuSiQue:
  Multihop Questions via Single-hop Question Composition*, TACL 2022. The repo
  does **not** auto-download MuSiQue — `scripts/prepare_datasets.py` has no
  MuSiQue entry; the harness expects pre-built files at
  `datasets/processed/musique/{chunks,questions,...}.jsonl` (see
  `docs/baselines/baseline_control_requirements_and_handoff.zh.md` and
  `docs/h200_remaining_datasets_tmux_runbook.zh.md`).
- **License:** CC BY 4.0 (per the upstream MuSiQue release); verify on the repo
  before redistribution.
- **Citation key:** none in `references.bib` for the MuSiQue dataset.
  (`trivedi2023ircot` exists but is the IRCoT *method* paper, a different work.)
- **Splits/scale used:** answer-quality rows in Table `tab:quality` (e.g.,
  Signpost `S_LLM`=4.8, J@7=36.0). HiPRAG did not complete on MuSiQue (context
  overflow), shown as "--" in the paper.

## Silver evidence (in-repo)

Reference answers exist for every workload, **but human gold supporting spans do
not**. Evidence-reachability diagnostics (SilverHit@5, SilverRecall@5, MRR,
ClaimCoverage@5; Table `tab:silver`) are therefore computed over **frozen silver
evidence** constructed in-repo, not over human gold spans. The paper is explicit
that these are silver diagnostics, "not human gold-span labels".

- **Builder:** `scripts/build_silver_evidence.py` (calls
  `signpost.evaluation.silver_builder.build_for_question`). It reads
  `questions.jsonl` (needs `question_id` / `question` / `answer`) and
  `chunks.jsonl` (needs `chunk_id` / `file_name` / `start_line` / `end_line` /
  `content`), and emits `llm_target_units.jsonl` + `llm_silver_chunks.jsonl`.
- It uses a model **different from the evaluated backbone** (e.g. `ecnu-max`) to
  avoid self-grading, and is resumable (already-built `question_id`s are
  skipped).
- A pure-Python offline test of the builder runs in the quickstart
  (`tests/test_silver_builder.py`).
