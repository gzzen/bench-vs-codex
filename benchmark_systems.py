"""PaperQA and Codex system adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import tomllib
from typing import Callable

from benchmark_data import Case, ROOT


DEFAULT_PAPERQA = ROOT.parent / "paper-qa"
Runner = Callable[[Case, Path, Path], tuple[str, dict, str]]


class PaperQAEmptyDocsError(RuntimeError):
    """PaperQA could not make any readable paper available to its agent."""


_live_log_lock = threading.Lock()
_ansi_csi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _show_live_log(label: str, text: str) -> None:
    """Replace the single live-log line with the latest subprocess output."""
    if not sys.stdout.isatty():
        return
    printable = "".join(ch if ch.isprintable() else " " for ch in _ansi_csi.sub("", text))
    message = " ".join(printable.split())
    if not message:
        return
    prefix = f"{label}: "
    width = max(shutil.get_terminal_size((120, 24)).columns - 1, len(prefix) + 1)
    available = width - len(prefix)
    if len(message) > available:
        message = (
            "..." + message[-(available - 3):]
            if available > 3 else message[-available:]
        )
    with _live_log_lock:
        sys.stdout.write(f"\r{prefix}{message}\033[K")
        sys.stdout.flush()


def _clear_live_log() -> None:
    if not sys.stdout.isatty():
        return
    with _live_log_lock:
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()


def _run_command(
    command: list[str],
    cwd: Path,
    log_prefix: Path,
    *,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> str:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    label = log_prefix.parent.name

    def capture(stream, stream_name: str, parts: list[str]) -> None:
        try:
            for text in stream:
                parts.append(text)
                _show_live_log(f"{label} {stream_name}", text)
        finally:
            stream.close()

    readers = [
        threading.Thread(
            target=capture, args=(process.stdout, "stdout", stdout_parts), daemon=True,
        ),
        threading.Thread(
            target=capture, args=(process.stderr, "stderr", stderr_parts), daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()
    try:
        if stdin is not None and process.stdin is not None:
            process.stdin.write(stdin)
            process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join()
        _clear_live_log()

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    log_prefix.with_suffix(".stdout.log").write_text(stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(stderr, encoding="utf-8")
    if return_code:
        detail = stderr.strip()[-1200:] or stdout.strip()[-1200:]
        raise RuntimeError(f"exit {return_code}: {detail}")
    return stdout


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
