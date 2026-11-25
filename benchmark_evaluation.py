"""Answer normalization, extraction, and semantic judging."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Callable
import urllib.request

from benchmark_data import ROOT


SUMMARIZER_MODEL = "deepseek-flash"
SUMMARIZER_URL = "https://api.deepseek.com/chat/completions"
SUMMARIZER_INSTRUCTIONS = (
    "Extract the answer from the response text for exact-match scoring. "
    "Use only the response text; the question is context, not a source to solve anew. "
    "Remove citations, references, explanations, and hedging. Preserve names, numbers, and units. "
    "If the response says it cannot answer, return 'I cannot answer.' "
    "Return only the answer, without labels, quotes, or commentary."
)
JUDGE_INSTRUCTIONS = (
    "Compare a candidate answer with a reference answer for the given question. "
    "Judge whether they mean the same thing as answers to that question. "
    "Accept harmless rewording, abbreviations, and formatting differences. "
    "Reject different entities, numbers, units, missing required parts, or contradictions. "
    "Use only the supplied question and answers; do not solve the question yourself. "
    "Reply with exactly YES or NO."
)

Summarizer = Callable[[str, str], tuple[str, dict]]
Judge = Callable[[str, str, str], tuple[bool, dict]]


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


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.casefold()).strip().strip(" \t\n\r.,;:!?\"'`“”‘’()[]{}")


def score(answer: str, reference: str) -> bool:
    return normalize_answer(answer) == normalize_answer(reference)


def _deepseek_completion(instructions: str, inputs: dict, max_tokens: int) -> str:
    """Call DeepSeek directly, without using PaperQA or Codex."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set DEEPSEEK_API_KEY in the project .env or environment")
    payload = {
        "model": SUMMARIZER_MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(
        SUMMARIZER_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        reason = result["choices"][0].get("finish_reason")
        raise ValueError(f"DeepSeek returned an empty response (finish_reason={reason})")
    return content.strip()


def extract_answer(question: str, response_text: str) -> tuple[str, dict]:
    """Extract a scoreable answer from a system response."""
    answer = _deepseek_completion(
        SUMMARIZER_INSTRUCTIONS, {"question": question, "response": response_text}, 512,
    )
    return answer, {
        "model": SUMMARIZER_MODEL, "temperature": 0, "max_tokens": 512,
        "thinking": "disabled",
        "endpoint": SUMMARIZER_URL,
    }


def judge_semantic_match(question: str, answer: str, reference_answer: str) -> tuple[bool, dict]:
    """Judge answer equivalence without seeing the source PDF or system response."""
    verdict = _deepseek_completion(
        JUDGE_INSTRUCTIONS,
        {"question": question, "candidate_answer": answer, "reference_answer": reference_answer},
        256,
    ).strip(' \t\r\n.\"\'')
    if verdict.casefold() not in ("yes", "no"):
        raise ValueError(f"DeepSeek judge returned an invalid verdict: {verdict[:80]!r}")
    return verdict.casefold() == "yes", {
        "model": SUMMARIZER_MODEL, "temperature": 0, "max_tokens": 256,
        "thinking": "disabled",
        "endpoint": SUMMARIZER_URL,
    }
