"""Small, sequential PaperQA versus Codex benchmark for local PDFs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / "MMLongBench-Doc"
DEFAULT_PAPERQA = ROOT.parent / "paper-qa"
PROMPT_SUFFIX = "Give only the short answer, with no explanation."


def load_project_env(path: Path = ROOT / ".env") -> None:
    """Load the project's DeepSeek key; an exported key takes precedence."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if not separator or key != "DEEPSEEK_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if value and not os.environ.get(key):
            os.environ[key] = value


@dataclass(frozen=True)
class Case:
    id: str
    document: str
    question: str
    reference_answer: str


def load_cases(path: Path) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Case manifest must be a nonempty JSON list")
    cases = []
    ids = set()
    for item in data:
        case = Case(**item)
        if not all((case.id, case.document, case.question, case.reference_answer)):
            raise ValueError("Case fields cannot be empty")
        if case.id in ids or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", case.id):
            raise ValueError(f"Invalid or duplicate case id: {case.id}")
        if Path(case.document).name != case.document or not case.document.endswith(".pdf"):
            raise ValueError(f"Document must be a PDF filename: {case.document}")
        ids.add(case.id)
        cases.append(case)
    return cases


def pdf_path(dataset: Path, case: Case) -> Path:
    path = dataset / "data" / "documents" / case.document
    if not path.is_file():
        raise FileNotFoundError(f"Missing PDF for {case.id}: {path}")
    return path


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.casefold()).strip().strip(" \t\n\r.,;:!?\"'`“”‘’()[]{}")


def score(answer: str, reference: str) -> bool:
    return normalize_answer(answer) == normalize_answer(reference)


def question_prompt(case: Case) -> str:
    return f"{case.question.strip()}\n\n{PROMPT_SUFFIX}"


def _run_command(command: list[str], cwd: Path, log_prefix: Path, *, env: dict[str, str] | None = None, stdin: str | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, env=env, input=stdin, text=True, capture_output=True,
        check=False,
    )
    log_prefix.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        detail = completed.stderr.strip()[-1200:] or completed.stdout.strip()[-1200:]
        raise RuntimeError(f"exit {completed.returncode}: {detail}")
    return completed.stdout


def run_paperqa(case: Case, pdf: Path, case_dir: Path, checkout: Path) -> tuple[str, dict, str]:
    python = checkout / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"PaperQA Python missing: {python}")
    papers = case_dir / "paperqa" / "papers"
    papers.mkdir(parents=True)
    shutil.copy2(pdf, papers / pdf.name)
    index = case_dir / "paperqa" / "index"
    env = os.environ.copy()
    env["PQA_HOME"] = str(case_dir / "paperqa" / "home")
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    output = _run_command([
        str(python), str(ROOT / "paperqa_bridge.py"),
        "--papers", str(papers), "--index", str(index),
        "--question", question_prompt(case),
    ], checkout, case_dir / "paperqa" / "process", env=env)
    # PaperQA may print diagnostics to stdout. The bridge's JSON is its last line.
    result = json.loads(output.strip().splitlines()[-1])
    return result["answer"], {
        "preset": "deepseek", "model": result["model"],
        "summary_model": result["summary_model"], "agent_model": result["agent_model"],
        "embedding": result["embedding"], "answer_length": "one short phrase or sentence",
        "status": result["status"], "index_directory": str(index),
    }, result["formatted_answer"]


def codex_settings() -> dict:
    config = ROOT / ".codex" / "config.toml"
    return tomllib.loads(config.read_text(encoding="utf-8"))


def codex_provider_overrides(config: dict) -> list[str]:
    provider_name = config["model_provider"]
    provider = config["model_providers"][provider_name]
    overrides = []
    for key in ("name", "base_url", "wire_api", "env_key", "supports_standalone_web_search"):
        value = provider[key]
        overrides.extend(["--config", f"model_providers.{provider_name}.{key}={json.dumps(value)}"])
    return overrides


def run_codex(case: Case, pdf: Path, case_dir: Path, codex_bin: str) -> tuple[str, dict, str]:
    config = codex_settings()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Set DEEPSEEK_API_KEY in the project .env or environment")
    codex_dir = case_dir / "codex"
    codex_dir.mkdir()
    shutil.copy2(pdf, codex_dir / pdf.name)
    answer_path = case_dir / "codex-answer.txt"
    prompt = (
        f"Read the local PDF file {pdf.name} to answer this question. "
        "Use only this PDF as evidence.\n\n" + question_prompt(case)
    )
    command = [
        codex_bin, "exec", "--skip-git-repo-check", "--ignore-user-config",
        "--strict-config", "--model", config["model"],
        "--config", f"model_provider={config['model_provider']}",
        "--config", "web_search=disabled", "--disable", "standalone_web_search",
        *codex_provider_overrides(config), "--sandbox", "read-only",
        "--ephemeral", "--color", "never", "--output-last-message", str(answer_path), "-",
    ]
    _run_command(command, codex_dir, case_dir / "codex-process", stdin=prompt)
    if not answer_path.is_file():
        raise RuntimeError("Codex finished without writing its final message")
    answer = answer_path.read_text(encoding="utf-8").strip()
    return answer, {
        "model": config["model"], "provider": config["model_provider"],
        "sandbox": "read-only", "web_search": "disabled",
        "user_config_ignored": True, "ephemeral": True, "binary": codex_bin,
    }, answer


Runner = Callable[[Case, Path, Path], tuple[str, dict, str]]


def run_cases(
    cases: list[Case], dataset: Path, work_dir: Path, output: Path,
    runners: dict[str, Runner], planned_settings: dict[str, dict] | None = None,
) -> list[dict]:
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
                answer, settings, full_answer, error = (
                    None, dict((planned_settings or {}).get(system, {})), None, None
                )
                try:
                    answer, actual_settings, full_answer = runner(case, pdf, case_dir)
                    settings.update(actual_settings)
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("Runner returned an empty answer")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    answer = None
                record = {
                    "case_id": case.id, "document": case.document, "question": case.question,
                    "reference_answer": case.reference_answer, "system": system,
                    "answer": answer, "full_answer": full_answer, "error": error,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "settings": settings, "exact_match": score(answer, case.reference_answer) if answer else None,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                outcome = "failed" if error else "done"
                print(
                    f"[{case_number}/{len(cases)}] {case.id} / {system} "
                    f"{outcome} ({record['elapsed_seconds']:.1f}s)",
                    flush=True,
                )
    return records


def print_table(records: list[dict]) -> None:
    print(f"{'Case':20} {'System':8} {'EM':5} {'Seconds':>8}  Answer / error")
    for row in records:
        value = row["answer"] if row["error"] is None else row["error"]
        value = " ".join(value.split())
        if len(value) > 70:
            value = value[:67] + "..."
        em = "-" if row["exact_match"] is None else ("yes" if row["exact_match"] else "no")
        print(f"{row['case_id']:20} {row['system']:8} {em:5} {row['elapsed_seconds']:8.1f}  {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=ROOT / "cases.json")
    parser.add_argument("--case", action="append", help="Run one case ID (repeatable)")
    parser.add_argument("--paperqa-dir", type=Path, default=DEFAULT_PAPERQA)
    parser.add_argument("--codex-bin", default=shutil.which("codex") or "codex")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.manifest)
        if args.case:
            unknown = set(args.case) - {case.id for case in cases}
            if unknown:
                raise ValueError(f"Unknown case IDs: {', '.join(sorted(unknown))}")
            cases = [case for case in cases if case.id in args.case]
        for case in cases:
            pdf_path(args.dataset, case)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for case in cases:
            print(f"{case.id:20} {case.document} | {case.question} | {case.reference_answer}")
        print(f"Validated {len(cases)} PDF paths; no model calls made.")
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
    paperqa_dir = args.paperqa_dir.resolve()
    records = run_cases(cases, args.dataset, work_dir, output, {
        "paperqa": lambda case, pdf, directory: run_paperqa(case, pdf, directory, paperqa_dir),
        "codex": lambda case, pdf, directory: run_codex(case, pdf, directory, args.codex_bin),
    }, planned_settings={
        "paperqa": {"preset": "deepseek", "model": "deepseek/deepseek-flash", "embedding": "sparse"},
        "codex": {
            "model": codex_settings()["model"], "provider": codex_settings()["model_provider"],
            "sandbox": "read-only", "web_search": "disabled",
            "user_config_ignored": True, "binary": args.codex_bin,
        },
    })
    print_table(records)
    print(f"Results: {output}")
    return 1 if any(record["error"] for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
