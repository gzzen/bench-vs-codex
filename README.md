# PaperQA vs Codex: MMLongBench-Doc PDF benchmark

This runner asks PaperQA and the installed Codex CLI the same questions about
the same local PDFs. Questions, reference answers, evidence sources, evidence
pages, answer formats, and document types are read directly from
`../MMLongBench-Doc/data/samples.json`; the PDFs are read from
`../MMLongBench-Doc/data/documents`. `cases.json` is no longer used.

The runner does not use Harbor.

## Setup

- Python 3.11 or newer for `cli.py` (standard library only).
- A local MMLongBench-Doc checkout at `../MMLongBench-Doc`. Pass `--data` to
  use another data directory or question JSON file.
- A PaperQA checkout at `../paper-qa` with its `.venv` installed (`uv sync
  --no-dev` in that checkout), or pass `--paperqa-dir`. This is only required
  when PaperQA is one of the selected systems.
- Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY` there. The runner
  loads it for both systems and the independent answer extractor; an already
  exported key takes precedence. `.env` is ignored by Git. PaperQA uses its
  `deepseek` preset, including the local sparse embedding model. Document text
  is sent to DeepSeek during answering.
- Install the Codex CLI. The repository's `.codex/config.toml` selects
  `deepseek-flash` through the DeepSeek Responses API using `DEEPSEEK_API_KEY`.
  The runner ignores the global Codex config, disables web search, and runs
  with `--sandbox read-only`.

## Quick start

Validate a few dataset records without making model calls:

```bash
python3 cli.py --dry-run --limit 5
```

Run only questions whose evidence sources do not contain `Figure` or `Chart`:

```bash
python3 cli.py --profile without-figures-and-charts --dry-run
python3 cli.py --profile without-figures-and-charts --limit 5
```

Run one system on plain-text questions:

```bash
python3 cli.py --profile plain-text --system codex --limit 1
```

Validate questions from one document type:

```bash
python3 cli.py --include-doc-type "Financial report" --dry-run --limit 5
```

The bundled `research-reports` profile selects one document type from
`run_configs.json`:

```bash
python3 cli.py --profile research-reports --dry-run --limit 5
```

Choose 10 random questions from the plain-text profile:

```bash
python3 cli.py --profile plain-text --limit 10 --random
```

Run the complete dataset with both systems:

```bash
python3 cli.py
```

That command schedules all 1,082 questions in the local dataset, so use a
profile or `--limit` for
smoke tests.

## Run profiles

[`run_configs.json`](run_configs.json) contains named profiles. Use
`--config` to select another profiles file and `--profile` to select a profile.
Each profile can set:

- `systems`: any nonempty combination of `paperqa` and `codex`.
- `limit`: maximum number of questions after filtering, or `null`.
- `random`: whether to randomize matching questions before applying `limit`.
- `filters.case_ids`: exact IDs such as `mmlongbench-0001`.
- `filters.include_documents` / `exclude_documents`: exact PDF filenames.
- `filters.include_doc_types` / `exclude_doc_types`: exact document-type
  labels, matched case-insensitively.
- `filters.include_question_contains` / `exclude_question_contains`:
  case-insensitive question substrings.
- `filters.include_evidence_sources` / `exclude_evidence_sources`:
  case-insensitive substrings matched against `evidence_sources`.

Include filters require any listed value to match. Exclude filters reject the
question when any listed value matches. All filter categories are ANDed, and
`limit` is applied after filtering. For example:

```json
{
  "profiles": {
    "text-and-tables": {
      "description": "Skip questions relying on figures or charts.",
      "systems": ["paperqa", "codex"],
      "limit": null,
      "filters": {
        "exclude_evidence_sources": ["Figure", "Chart"],
        "exclude_question_contains": ["in the image"]
      }
    }
  }
}
```

Convenience overrides are available for `--system`, `--limit`, `--random`,
`--case`, `--include-doc-type`, `--exclude-doc-type`,
`--include-evidence-source`, and `--exclude-evidence-source`. The override
filters add to the selected profile's filters.

## Execution and results

Each run gets a new ignored `runs/<timestamp>-<id>/` directory. Its `run.json`
records the effective profile, dataset path, question file, and selected case
count. PaperQA gets a per-case paper folder containing only that PDF, a separate
index, and separate PaperQA state. Codex gets its own per-case directory
containing the same PDF.

The CLI runs cases and systems sequentially. Each system receives the original
dataset question exactly as written, with no answer-length instruction. After
each response, a separate DeepSeek `deepseek-flash` call extracts a short
answer using only the question and response. It does not receive the PDF or
reference answer. Another DeepSeek call judges whether the extracted answer
means the same thing as the dataset's reference answer. The judge sees the
question and two answers, but not the PDF or the system's raw response.

`results.jsonl` has one record per case and system with the question, reference
answer, `evidence_sources`, `evidence_pages`, answer format, document type,
dataset index, raw system answer, complete response, extracted answer, model
settings, timings, errors, `exact_match`, and `semantic_match`. The CLI table
shows scores as `EM` and `SM` (`yes`, `no`, or `-` if unavailable). Process logs
from PaperQA and Codex update a single terminal line while each system runs and
are kept beside the case folders. Use `--work-dir` and `--output` to choose
other locations.

PaperQA answers that report that no papers were available are recorded as
execution errors rather than correct abstentions. Such records bypass answer
extraction and semantic judging, so `EM` and `SM` remain unavailable (`-`).

The extractor's `answer` is scored after normalization of letter case,
whitespace, and surrounding punctuation. Extraction can introduce mistakes;
inspect `raw_answer` and `full_answer` when reviewing results. A valid
paraphrase can still score as an exact-match miss. The semantic judge is an LLM
assessment and can be wrong; review borderline cases manually.

## Code layout

- `cli.py`: argument parsing, run setup, and CLI result rendering.
- `benchmark.py`: reusable execution core and result recording.
- `benchmark_data.py`: MMLongBench-Doc loading, path resolution, profiles, and
  filters.
- `benchmark_evaluation.py`: normalization, answer extraction, and semantic
  judging.
- `benchmark_systems.py`: PaperQA and Codex command adapters.
- `paperqa_bridge.py`: JSON bridge into the PaperQA checkout.

Run the stubbed tests with:

```bash
python3 -m unittest discover -s tests
```
