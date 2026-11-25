"""PaperQA and Codex system adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib
from typing import Callable

from benchmark_data import Case, ROOT


DEFAULT_PAPERQA = ROOT.parent / "paper-qa"
Runner = Callable[[Case, Path, Path], tuple[str, dict, str]]


class PaperQAEmptyDocsError(RuntimeError):
    """PaperQA could not make any readable paper available to its agent."""


def _run_command(
    command: list[str],
    cwd: Path,
    log_prefix: Path,
    *,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> str:
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
        "--question", case.question,
    ], checkout, case_dir / "paperqa" / "process", env=env)
    # PaperQA may print diagnostics to stdout. The bridge's JSON is its last line.
    result = json.loads(output.strip().splitlines()[-1])
    response_text = "\n".join(
        str(result.get(key) or "") for key in ("answer", "formatted_answer")
    )
    if "having no papers" in response_text.casefold():
        raise PaperQAEmptyDocsError(
            "PaperQA produced no readable papers "
            f"(status={result.get('status', 'unknown')}): {result['answer']}"
        )
    return result["answer"], {
        "preset": "deepseek", "model": result["model"],
        "summary_model": result["summary_model"], "agent_model": result["agent_model"],
        "embedding": result["embedding"],
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
    command = [
        codex_bin, "exec", "--skip-git-repo-check", "--ignore-user-config",
        "--strict-config", "--model", config["model"],
        "--config", f"model_provider={config['model_provider']}",
        "--config", "web_search=disabled", "--disable", "standalone_web_search",
        *codex_provider_overrides(config), "--sandbox", "read-only",
        "--ephemeral", "--color", "never", "--output-last-message", str(answer_path), "-",
    ]
    _run_command(command, codex_dir, case_dir / "codex-process", stdin=case.question)
    if not answer_path.is_file():
        raise RuntimeError("Codex finished without writing its final message")
    answer = answer_path.read_text(encoding="utf-8").strip()
    return answer, {
        "model": config["model"], "provider": config["model_provider"],
        "sandbox": "read-only", "web_search": "disabled",
        "user_config_ignored": True, "ephemeral": True, "binary": codex_bin,
    }, answer
