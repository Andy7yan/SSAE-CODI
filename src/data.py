from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

DATASET_REPO = "zen-E/GSM8k-Aug"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _datasets_cache_dir(hf_home: Path) -> Path:
    return Path(hf_home).resolve() / "datasets"


def _saved_split_path(hf_home: Path, split: str) -> Path:
    return Path(hf_home).resolve() / "saved-datasets" / f"zen-E--GSM8k-Aug--{split}"


def _available_splits(dataset_dict: DatasetDict) -> list[str]:
    return sorted(str(name) for name in dataset_dict.keys())


def _build_missing_split_error(split: str, available_splits: list[str]) -> ValueError:
    available_display = ", ".join(available_splits) if available_splits else "(none)"
    return ValueError(
        f"Split '{split}' is not available in {DATASET_REPO}. "
        f"Available splits: {available_display}. "
        "If you need a test set, use the 'test' split from openai/gsm8k."
    )


def _extract_raw_text(row: dict[str, Any]) -> str:
    for key in ("raw_text", "text", "sample", "content"):
        if key in row and row[key] is not None:
            return str(row[key])

    if {"question", "cot", "answer"}.issubset(row):
        question = str(row["question"])
        cot = str(row["cot"])
        answer = str(row["answer"])
        return f"{question}||{cot}####{answer}"

    available_columns = ", ".join(sorted(row.keys()))
    raise ValueError(
        f"Unable to determine raw text for {DATASET_REPO}. "
        f"Expected a text-like column or question/cot/answer columns, found: {available_columns}"
    )


def _parse_raw_text(raw_text: str) -> tuple[str, str, float]:
    if "||" not in raw_text:
        raise ValueError(f"Invalid sample format in {DATASET_REPO}: missing '||' delimiter.")

    question, remainder = raw_text.split("||", 1)
    if "####" not in remainder:
        raise ValueError(f"Invalid sample format in {DATASET_REPO}: missing '####' delimiter.")

    cot, answer_text = remainder.split("####", 1)
    question = question.strip()
    cot = cot.strip()
    answer = float(answer_text.strip().replace(",", ""))
    return question, cot, answer


def _parse_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    raw_text = _extract_raw_text(row)
    question, cot, answer = _parse_raw_text(raw_text)
    return {
        "question": question,
        "answer": answer,
        "cot": cot,
        "raw_text": raw_text,
        "index": index,
    }


def _parse_split(dataset: Dataset) -> Dataset:
    parsed_rows = [_parse_row(dict(row), index) for index, row in enumerate(dataset)]
    return Dataset.from_list(parsed_rows)


def load_gsm8k_aug(hf_home: Path, split: str) -> Dataset:
    resolved_hf_home = Path(hf_home).resolve()
    saved_split_path = _saved_split_path(resolved_hf_home, split)

    if saved_split_path.exists():
        return load_from_disk(str(saved_split_path))

    datasets_cache_dir = _datasets_cache_dir(resolved_hf_home)
    _ensure_dir(datasets_cache_dir)

    dataset_dict = load_dataset(DATASET_REPO, cache_dir=str(datasets_cache_dir))
    available_splits = _available_splits(dataset_dict)
    if split not in dataset_dict:
        raise _build_missing_split_error(split, available_splits)

    parsed_dataset = _parse_split(dataset_dict[split])
    _ensure_dir(saved_split_path.parent)
    parsed_dataset.save_to_disk(str(saved_split_path))
    return parsed_dataset
