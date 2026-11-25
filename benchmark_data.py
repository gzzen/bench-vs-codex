"""MMLongBench-Doc loading and run-profile filtering."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "MMLongBench-Doc" / "data"
DEFAULT_CONFIG = ROOT / "run_configs.json"


@dataclass(frozen=True)
class Case:
    id: str
    document: str
    question: str
    reference_answer: str
    evidence_sources: tuple[str, ...] = ()
    evidence_pages: tuple[int, ...] = ()
    answer_format: str | None = None
    doc_type: str | None = None
    dataset_index: int | None = None


@dataclass(frozen=True)
class RunFilters:
    case_ids: tuple[str, ...] = ()
    include_documents: tuple[str, ...] = ()
    exclude_documents: tuple[str, ...] = ()
    include_question_contains: tuple[str, ...] = ()
    exclude_question_contains: tuple[str, ...] = ()
    include_evidence_sources: tuple[str, ...] = ()
    exclude_evidence_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    name: str
    description: str
    systems: tuple[str, ...]
    limit: int | None
    randomize: bool
    filters: RunFilters


def load_cases(path: Path) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Question file must be a nonempty JSON list")
    mmlongbench_format = "doc_id" in data[0]
    if mmlongbench_format and not (
        {"question", "answer"} <= set(data[0])
    ):
        raise ValueError("MMLongBench-Doc records require question and answer fields")
    cases = []
    ids = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Case {index} must be a JSON object")
        if mmlongbench_format:
            case = _case_from_mmlongbench(item, index)
        else:
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


def _literal_list(value, field: str, index: int) -> list:
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Case {index} has invalid {field}: {value!r}") from exc
    else:
        raise ValueError(f"Case {index} has invalid {field}: {value!r}")
    if not isinstance(parsed, list):
        raise ValueError(f"Case {index} has invalid {field}: {value!r}")
    return parsed


def _case_from_mmlongbench(item: dict, index: int) -> Case:
    missing = {"doc_id", "question", "answer"} - set(item)
    if missing:
        raise ValueError(
            f"Case {index} is missing MMLongBench-Doc fields: "
            f"{', '.join(sorted(missing))}"
        )
    sources = _literal_list(item.get("evidence_sources", "[]"), "evidence_sources", index)
    if not all(isinstance(source, str) and source.strip() for source in sources):
        raise ValueError(f"Case {index} evidence_sources must contain strings")
    pages = _literal_list(item.get("evidence_pages", "[]"), "evidence_pages", index)
    if not all(isinstance(page, int) and not isinstance(page, bool) for page in pages):
        raise ValueError(f"Case {index} evidence_pages must contain integers")
    return Case(
        id=f"mmlongbench-{index:04d}",
        document=str(item["doc_id"]),
        question=str(item["question"]),
        reference_answer=str(item["answer"]),
        evidence_sources=tuple(sources),
        evidence_pages=tuple(pages),
        answer_format=str(item["answer_format"]) if item.get("answer_format") else None,
        doc_type=str(item["doc_type"]) if item.get("doc_type") else None,
        dataset_index=index,
    )


def questions_path(data: Path) -> Path:
    """Resolve a data directory (or file) to MMLongBench-Doc's question JSON."""
    data = data.expanduser()
    if data.is_file():
        return data
    for name in ("samples.json", "questions.json"):
        candidate = data / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing MMLongBench-Doc questions file under: {data}")


def documents_dir(data: Path) -> Path:
    data = data.expanduser()
    if data.is_file():
        data = data.parent
    for candidate in (data / "documents", data / "data" / "documents"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing MMLongBench-Doc documents directory under: {data}")


def pdf_path(data: Path, case: Case) -> Path:
    path = documents_dir(data) / case.document
    if not path.is_file():
        raise FileNotFoundError(f"Missing PDF for {case.id}: {path}")
    return path


def _string_tuple(value, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of nonempty strings")
    return tuple(item.strip() for item in value)


def load_run_config(path: Path = DEFAULT_CONFIG, profile: str = "all") -> RunConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Run configuration must be a JSON object")
    if "profiles" in data:
        profiles = data["profiles"]
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("Run configuration profiles must be a nonempty object")
        if profile not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(f"Unknown run profile {profile!r}; available: {available}")
        selected = profiles[profile]
        if not isinstance(selected, dict):
            raise ValueError(f"Run profile {profile!r} must be a JSON object")
    else:
        selected = data

    systems = selected.get("systems", ["paperqa", "codex"])
    if not isinstance(systems, list) or not systems:
        raise ValueError(f"Run profile {profile!r} systems must be a nonempty list")
    systems = tuple(systems)
    if any(system not in ("paperqa", "codex") for system in systems):
        raise ValueError("Run systems must be 'paperqa', 'codex', or both")
    if len(set(systems)) != len(systems):
        raise ValueError(f"Run profile {profile!r} contains duplicate systems")

    limit = selected.get("limit")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise ValueError(f"Run profile {profile!r} limit must be a positive integer or null")
    randomize = selected.get("random", False)
    if not isinstance(randomize, bool):
        raise ValueError(f"Run profile {profile!r} random must be a boolean")

    raw_filters = selected.get("filters", {})
    if not isinstance(raw_filters, dict):
        raise ValueError(f"Run profile {profile!r} filters must be a JSON object")
    filter_fields = set(RunFilters.__dataclass_fields__)
    unknown_filters = set(raw_filters) - filter_fields
    if unknown_filters:
        raise ValueError(
            f"Unknown filters in run profile {profile!r}: "
            f"{', '.join(sorted(unknown_filters))}"
        )
    filters = RunFilters(**{
        field: _string_tuple(raw_filters.get(field, []), f"filters.{field}")
        for field in sorted(filter_fields)
    })
    description = selected.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"Run profile {profile!r} description must be a string")
    return RunConfig(
        name=profile,
        description=description,
        systems=systems,
        limit=limit,
        randomize=randomize,
        filters=filters,
    )


def _matches_any(values: tuple[str, ...] | list[str], needles: tuple[str, ...]) -> bool:
    folded_values = [value.casefold() for value in values]
    return any(
        needle.casefold() in value
        for needle in needles
        for value in folded_values
    )


def filter_cases(cases: list[Case], filters: RunFilters) -> list[Case]:
    selected = []
    for case in cases:
        if filters.case_ids and case.id not in filters.case_ids:
            continue
        if filters.include_documents and case.document.casefold() not in {
            value.casefold() for value in filters.include_documents
        }:
            continue
        if filters.exclude_documents and case.document.casefold() in {
            value.casefold() for value in filters.exclude_documents
        }:
            continue
        if filters.include_question_contains and not _matches_any(
            [case.question], filters.include_question_contains
        ):
            continue
        if filters.exclude_question_contains and _matches_any(
            [case.question], filters.exclude_question_contains
        ):
            continue
        if filters.include_evidence_sources and not _matches_any(
            case.evidence_sources, filters.include_evidence_sources
        ):
            continue
        if filters.exclude_evidence_sources and _matches_any(
            case.evidence_sources, filters.exclude_evidence_sources
        ):
            continue
        selected.append(case)
    return selected
