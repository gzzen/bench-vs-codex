"""Command-line interface for the PaperQA versus Codex benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from benchmark import run_cases, select_cases
from benchmark_data import (
    DEFAULT_CONFIG,
    DEFAULT_DATA,
    ROOT,
    filter_cases,
    load_cases,
    load_run_config,
    pdf_path,
    questions_path,
)
from benchmark_evaluation import (
    extract_answer,
    judge_semantic_match,
    load_project_env,
)
from benchmark_systems import (
    DEFAULT_PAPERQA,
    codex_settings,
    run_codex,
    run_paperqa,
)


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


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--include-doc-type", action="append", default=[],
                        help="Keep cases with this exact document type (repeatable)")
    parser.add_argument("--exclude-doc-type", action="append", default=[],
                        help="Skip cases with this exact document type (repeatable)")
    parser.add_argument("--include-evidence-source", action="append", default=[],
                        help="Keep cases whose evidence sources contain this text (repeatable)")
    parser.add_argument("--exclude-evidence-source", action="append", default=[],
                        help="Skip cases whose evidence sources contain this text (repeatable)")
    parser.add_argument("--paperqa-dir", type=Path, default=DEFAULT_PAPERQA)
    parser.add_argument("--codex-bin", default=shutil.which("codex") or "codex")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
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
        if (
            args.include_doc_type or args.exclude_doc_type
            or args.include_evidence_source or args.exclude_evidence_source
        ):
            filters = replace(
                config.filters,
                include_doc_types=config.filters.include_doc_types
                + tuple(args.include_doc_type),
                exclude_doc_types=config.filters.exclude_doc_types
                + tuple(args.exclude_doc_type),
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
            doc_type = case.doc_type or "unknown"
            print(
                f"{case.id:20} {case.document} | {case.question} | "
                f"{case.reference_answer} | type={doc_type} | sources={sources}"
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
