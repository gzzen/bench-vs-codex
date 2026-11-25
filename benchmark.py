"""Main runner for the sequential PaperQA versus Codex benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

from benchmark_data import (
    DEFAULT_CONFIG,
    DEFAULT_DATA,
    ROOT,
    Case,
    filter_cases,
    load_cases,
    load_run_config,
    pdf_path,
    questions_path,
)
from benchmark_evaluation import (
    SUMMARIZER_MODEL,
    Judge,
    Summarizer,
    extract_answer,
    judge_semantic_match,
    load_project_env,
    score,
)
from benchmark_systems import (
    DEFAULT_PAPERQA,
    Runner,
    codex_settings,
    run_codex,
    run_paperqa,
)


def run_cases(
    cases: list[Case], dataset: Path, work_dir: Path, output: Path,
    runners: dict[str, Runner], summarizer: Summarizer, judge: Judge,
    planned_settings: dict[str, dict] | None = None,
) -> list[dict]:
    """Run each system, extract an answer, judge it, and persist each result."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("w", encoding="utf-8") as stream:
        for case_number, case in enumerate(cases, start=1):
            pdf = pdf_path(dataset, case)
            case_dir = work_dir / case.id
            case_dir.mkdir()
            for system, runner in runners.items():
                print(f"[{case_number}/{len(cases)}] {case.id} / {system} running...", flush=True)
                started = time.perf_counter()
                raw_answer, full_answer, error = None, None, None
                settings = dict((planned_settings or {}).get(system, {}))
                try:
                    raw_answer, actual_settings, full_answer = runner(case, pdf, case_dir)
                    settings.update(actual_settings)
                    if not isinstance(raw_answer, str) or not raw_answer.strip():
                        raise ValueError("Runner returned an empty answer")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    raw_answer = None
                runner_elapsed = round(time.perf_counter() - started, 3)

                answer, summarizer_error, summarizer_elapsed = None, None, None
                summarizer_settings = {"model": SUMMARIZER_MODEL}
                if raw_answer is not None:
                    print(f"[{case_number}/{len(cases)}] {case.id} / {system} extracting...", flush=True)
                    summarizer_started = time.perf_counter()
                    try:
                        answer, actual_summarizer_settings = summarizer(case.question, raw_answer)
                        summarizer_settings.update(actual_summarizer_settings)
                        if not isinstance(answer, str) or not answer.strip():
                            raise ValueError("Summarizer returned an empty answer")
                    except Exception as exc:
                        summarizer_error = f"{type(exc).__name__}: {exc}"
                        answer = None
                    summarizer_elapsed = round(time.perf_counter() - summarizer_started, 3)

                semantic_match, judge_error, judge_elapsed = None, None, None
                judge_settings = {"model": SUMMARIZER_MODEL}
                if answer is not None:
                    print(f"[{case_number}/{len(cases)}] {case.id} / {system} judging...", flush=True)
                    judge_started = time.perf_counter()
                    try:
                        semantic_match, actual_judge_settings = judge(
                            case.question, answer, case.reference_answer,
                        )
                        judge_settings.update(actual_judge_settings)
                        if not isinstance(semantic_match, bool):
                            raise ValueError("Judge returned a non-boolean verdict")
                    except Exception as exc:
                        judge_error = f"{type(exc).__name__}: {exc}"
                        semantic_match = None
                    judge_elapsed = round(time.perf_counter() - judge_started, 3)
                record = {
                    "case_id": case.id, "document": case.document, "question": case.question,
                    "reference_answer": case.reference_answer, "system": system,
                    "evidence_sources": list(case.evidence_sources),
                    "evidence_pages": list(case.evidence_pages),
                    "answer_format": case.answer_format,
                    "doc_type": case.doc_type,
                    "dataset_index": case.dataset_index,
                    "answer": answer, "raw_answer": raw_answer, "full_answer": full_answer,
                    "error": error, "summarizer_error": summarizer_error,
                    "judge_error": judge_error,
                    "elapsed_seconds": runner_elapsed,
                    "summarizer_elapsed_seconds": summarizer_elapsed,
                    "judge_elapsed_seconds": judge_elapsed,
                    "summarizer_settings": summarizer_settings,
                    "judge_settings": judge_settings,
                    "settings": settings,
                    "exact_match": score(answer, case.reference_answer) if answer else None,
                    "semantic_match": semantic_match,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                outcome = "failed" if error or summarizer_error or judge_error else "done"
                elapsed = f"model {runner_elapsed:.1f}s"
                if summarizer_elapsed is not None:
                    elapsed += f", extraction {summarizer_elapsed:.1f}s"
                if judge_elapsed is not None:
                    elapsed += f", judge {judge_elapsed:.1f}s"
                print(
                    f"[{case_number}/{len(cases)}] {case.id} / {system} "
                    f"{outcome} ({elapsed})",
                    flush=True,
                )
    return records


def print_table(records: list[dict]) -> None:
    print(f"{'Case':20} {'System':8} {'EM':5} {'SM':5} {'Model s':>8}  Answer / error")
    for row in records:
        value = row["error"] or row["summarizer_error"] or row["judge_error"] or row["answer"]
        value = " ".join(value.split())
        if len(value) > 70:
            value = value[:67] + "..."
        em = "-" if row["exact_match"] is None else ("yes" if row["exact_match"] else "no")
        sm = "-" if row["semantic_match"] is None else ("yes" if row["semantic_match"] else "no")
        print(f"{row['case_id']:20} {row['system']:8} {em:5} {sm:5} {row['elapsed_seconds']:8.1f}  {value}")


def select_cases(
    cases: list[Case], limit: int | None = None, randomize: bool = False,
) -> list[Case]:
    """Select cases in dataset order or randomly, then apply the limit."""
    selected = list(cases)
    if randomize:
        random.shuffle(selected)
    return selected[:limit] if limit is not None else selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--dataset", dest="data", type=Path, default=DEFAULT_DATA,
                        help="MMLongBench-Doc data directory or questions JSON")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Named run profiles JSON")
    parser.add_argument("--profile", default="all", help="Run profile name")
    parser.add_argument("--case", action="append", help="Run one case ID (repeatable)")
    parser.add_argument("--system", action="append", choices=("paperqa", "codex"),
                        help="System to run (repeatable; overrides the profile)")
    parser.add_argument("--limit", type=int, help="Maximum number of filtered questions")
    parser.add_argument("--random", action="store_true",
                        help="Randomly reorder matching questions before applying --limit")
    parser.add_argument("--include-evidence-source", action="append", default=[],
                        help="Keep cases whose evidence sources contain this text (repeatable)")
    parser.add_argument("--exclude-evidence-source", action="append", default=[],
                        help="Skip cases whose evidence sources contain this text (repeatable)")
    parser.add_argument("--paperqa-dir", type=Path, default=DEFAULT_PAPERQA)
    parser.add_argument("--codex-bin", default=shutil.which("codex") or "codex")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_run_config(args.config, args.profile)
        if args.system:
            config = replace(config, systems=tuple(dict.fromkeys(args.system)))
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be a positive integer")
            config = replace(config, limit=args.limit)
        if args.random:
            config = replace(config, randomize=True)
        if args.include_evidence_source or args.exclude_evidence_source:
            filters = replace(
                config.filters,
                include_evidence_sources=config.filters.include_evidence_sources
                + tuple(args.include_evidence_source),
                exclude_evidence_sources=config.filters.exclude_evidence_sources
                + tuple(args.exclude_evidence_source),
            )
            config = replace(config, filters=filters)
        all_cases = load_cases(questions_path(args.data))
        available_ids = {case.id for case in all_cases}
        requested_ids = set(config.filters.case_ids) | set(args.case or [])
        unknown_ids = requested_ids - available_ids
        if unknown_ids:
            raise ValueError(f"Unknown case IDs: {', '.join(sorted(unknown_ids))}")
        if args.case:
            config = replace(
                config,
                filters=replace(config.filters, case_ids=tuple(dict.fromkeys(args.case))),
            )
        cases = filter_cases(all_cases, config.filters)
        cases = select_cases(cases, config.limit, config.randomize)
        if not cases:
            raise ValueError("No questions matched the selected run profile")
        for case in cases:
            pdf_path(args.data, case)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for case in cases:
            sources = ", ".join(case.evidence_sources) or "none"
            print(
                f"{case.id:20} {case.document} | {case.question} | "
                f"{case.reference_answer} | sources={sources}"
            )
        print(
            f"Profile {config.name}: validated {len(cases)} PDF paths and "
            f"selected systems ({', '.join(config.systems)}); "
            f"selection={'random' if config.randomize else 'ordered'}; no model calls made."
        )
        return 0
    load_project_env()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("Set DEEPSEEK_API_KEY in .env or the process environment")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    work_dir = (args.work_dir or ROOT / "runs" / run_id).resolve()
    output = (args.output or work_dir / "results.jsonl").resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        parser.error(f"Work directory is not empty: {work_dir}")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "run.json").write_text(
        json.dumps({
            "config": asdict(config),
            "data": str(args.data.resolve()),
            "questions": str(questions_path(args.data).resolve()),
            "case_count": len(cases),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paperqa_dir = args.paperqa_dir.resolve()
    runner_factories = {
        "paperqa": lambda case, pdf, directory: run_paperqa(case, pdf, directory, paperqa_dir),
        "codex": lambda case, pdf, directory: run_codex(case, pdf, directory, args.codex_bin),
    }
    planned_settings = {
        "paperqa": {"preset": "deepseek", "model": "deepseek/deepseek-flash", "embedding": "sparse"},
        "codex": {
            "model": codex_settings()["model"], "provider": codex_settings()["model_provider"],
            "sandbox": "read-only", "web_search": "disabled",
            "user_config_ignored": True, "binary": args.codex_bin,
        },
    }
    records = run_cases(
        cases,
        args.data,
        work_dir,
        output,
        {system: runner_factories[system] for system in config.systems},
        extract_answer,
        judge_semantic_match,
        planned_settings={
            system: planned_settings[system] for system in config.systems
        },
    )
    print_table(records)
    print(f"Results: {output}")
    return 1 if any(
        record["error"] or record["summarizer_error"] or record["judge_error"]
        for record in records
    ) else 0


if __name__ == "__main__":
    sys.exit(main())
