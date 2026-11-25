# PaperQA vs Codex: local PDF benchmark

This runner asks PaperQA and the installed Codex CLI the same five questions about
the same PDFs from MMLongBench-Doc. Each question has one checked-in reference
answer in `cases.json`. These five questions are marked as plain-text evidence
in the source dataset. The runner does not use Harbor.

## Setup

- Python 3.11 or newer for `benchmark.py` (standard library only).
- A local MMLongBench-Doc checkout at `../MMLongBench-Doc`, or pass `--dataset`.
- A PaperQA checkout at `../paper-qa` with its `.venv` installed (`uv sync
  --no-dev` in that checkout), or pass `--paperqa-dir`.
- Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY` there. The runner
  loads it for both systems and the independent answer extractor; an already
  exported key takes precedence. `.env`
  is ignored by Git. PaperQA uses its `deepseek` preset, including the local
  sparse embedding model. Document text is sent to DeepSeek during answering.
- Install the Codex CLI. The repository's `.codex/config.toml` selects
  `deepseek-flash` through the DeepSeek Responses API using `DEEPSEEK_API_KEY`.
  The runner ignores the global Codex config, disables web search, and runs
  with `--sandbox read-only`. Codex may ignore project settings until the
  repository is trusted; the runner reads the local config and passes its
  non-secret provider fields explicitly, so no trust change is needed for the
  benchmark. DeepSeek API requests are still required.

Check that the five PDFs are available without making model calls:

```bash
python3 benchmark.py --dry-run
```

Run one case as a smoke test:

```bash
python3 benchmark.py --case limes-phone
```

Run all five:

```bash
python3 benchmark.py
```

Each run gets a new ignored `runs/<timestamp>-<id>/` directory. PaperQA gets a
per-case paper folder containing only that PDF, a separate index, and separate
PaperQA state. Codex gets its own per-case directory containing the same PDF.
The CLI runs cases and systems sequentially and prints brief start/finish lines
for each system before the final comparison table. Each system receives the
question from `cases.json` exactly as written, with no answer-length instruction.
After each response, a separate DeepSeek `deepseek-flash` call extracts a short
answer using only the question and that response. It does not receive the PDF
or reference answer. Another DeepSeek call judges whether the extracted answer
means the same thing as the reference answer for that question. The judge sees
the question and two answers, but not the PDF or the system's raw response.

`results.jsonl` has one record per case and system with the raw system answer,
complete response, extracted answer, separate model, extraction, and judge times,
settings, errors, `exact_match`, and `semantic_match`. The CLI table shows these
scores as `EM` and `SM` (`yes`, `no`, or `-` if unavailable). Process logs are kept beside the
case folders. Use
`--work-dir` and `--output` to choose other locations.

The extractor's `answer` is scored after normalization of letter case,
whitespace, and surrounding punctuation. Extraction can introduce mistakes;
inspect `raw_answer` and `full_answer` when reviewing results. A valid
paraphrase can still score as an exact-match miss. The semantic judge is an LLM
assessment and can be wrong; review borderline cases manually. Both systems use DeepSeek, but PaperQA's
preset and Codex's agent workflow differ, so this still compares complete systems.

Run the stubbed tests with:

```bash
python3 -m unittest discover -s tests
```
