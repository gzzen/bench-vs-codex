"""Execution core for the sequential PaperQA versus Codex benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

from benchmark_data import (
    Case,
    pdf_path,
)
from benchmark_evaluation import (
    SUMMARIZER_MODEL,
    Judge,
    Summarizer,
    score,
)
from benchmark_systems import Runner


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
                if semantic_match is not None:
                    elapsed += f", correct: {'yes' if semantic_match else 'no'}"
                print(
                    f"[{case_number}/{len(cases)}] {case.id} / {system} "
                    f"{outcome} ({elapsed})",
                    flush=True,
                )
    return records


def select_cases(
    cases: list[Case], limit: int | None = None, randomize: bool = False,
) -> list[Case]:
    """Select cases in dataset order or randomly, then apply the limit."""
    selected = list(cases)
    if randomize:
        random.shuffle(selected)
    return selected[:limit] if limit is not None else selected

if __name__ == "__main__":
    from cli import main

    sys.exit(main())
