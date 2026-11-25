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
  loads it for both systems; an already exported key takes precedence. `.env`
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
for each system before the final comparison table. `results.jsonl` has one record per
case and system with the short answer, full response, any error, elapsed seconds, settings,
and an exact-match score. Process logs are kept beside the case folders. Use
`--work-dir` and `--output` to choose other locations.

PaperQA citation markers are removed from its short answer before scoring; its
complete cited response remains in `full_answer`. Exact match then ignores
letter case, whitespace differences, and surrounding punctuation only. A valid
paraphrase or answer with extra context can score as a miss; inspect the saved
answers manually. Both systems use DeepSeek, but PaperQA's preset and Codex's
agent workflow differ, so this still compares complete systems.

Run the stubbed tests with:

```bash
python3 -m unittest discover -s tests
```
