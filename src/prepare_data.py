"""Download a HuggingFace dataset split and export it to JSONL for CODI inference.

Each output row: {"sample_id": str, "question": str, "answer": str}

Usage:
    # Standard GSM8K test set (evaluation):
    python src/prepare_data.py \
        --output /srv/scratch/$USER/ssae-codi/data/gsm8k_test.jsonl \
        --split test

    # Standard GSM8K train set (latent generation for SAE):
    python src/prepare_data.py \
        --output /srv/scratch/$USER/ssae-codi/data/gsm8k_train.jsonl \
        --split train

    # Custom dataset (e.g. GSM8K-Aug variant):
    python src/prepare_data.py \
        --repo <hf-repo-id> --config <dataset-config> --split train \
        --question-field question --answer-field answer \
        --output /srv/scratch/$USER/ssae-codi/data/gsm8k_aug_train.jsonl

Output path is printed to stdout so PBS scripts can capture it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from io_utils import append_jsonl, ensure_dir, write_json


def _load_repo_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def _extract_answer_text(raw_answer: str) -> str:
    """Strip GSM8K calculator annotations (#### <number>) from answer text.

    GSM8K answers look like:
        "She has 3 apples.\\n#### 3"
    We keep the full text (reasoning + number) as the reference.
    """
    return raw_answer.strip()


def prepare_data(
    output_path: str | Path,
    repo: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "test",
    question_field: str = "question",
    answer_field: str = "answer",
    max_samples: int = -1,
    hf_token: str | None = None,
) -> Path:
    """Download a HuggingFace dataset split and write it as JSONL.

    Args:
        output_path: Destination JSONL file.
        repo: HuggingFace dataset repository ID.
        dataset_config: Dataset config/subset name (e.g. "main" for gsm8k).
        split: Dataset split ("train" or "test").
        question_field: Column name that contains the question text.
        answer_field: Column name that contains the answer text.
        max_samples: Maximum number of samples to export; -1 = all.
        hf_token: Optional HuggingFace auth token.

    Returns:
        Resolved output path.
    """
    from datasets import load_dataset  # type: ignore

    out_path = Path(output_path)
    ensure_dir(out_path.parent)

    # Remove existing file to avoid accidental appending.
    if out_path.exists():
        out_path.unlink()

    load_kwargs: dict = {"path": repo, "name": dataset_config, "split": split, "trust_remote_code": True}
    if hf_token:
        load_kwargs["token"] = hf_token

    dataset = load_dataset(**load_kwargs)

    if max_samples != -1:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    id_width = len(str(len(dataset)))
    for idx, row in enumerate(dataset):
        question = str(row[question_field]).strip()
        answer = _extract_answer_text(str(row[answer_field]))
        sample_id = f"{split}_{idx:0{id_width}d}"
        append_jsonl(out_path, {"sample_id": sample_id, "question": question, "answer": answer})

    meta_path = out_path.with_suffix(".meta.json")
    write_json(
        meta_path,
        {
            "repo": repo,
            "dataset_config": dataset_config,
            "split": split,
            "question_field": question_field,
            "answer_field": answer_field,
            "n_samples": len(dataset),
            "output_path": str(out_path.resolve()),
        },
    )

    print(f"Wrote {len(dataset)} samples → {out_path}")
    return out_path.resolve()


def main() -> None:
    _load_repo_dotenv()

    hf_token_default = os.getenv("HF_TOKEN", "").strip() or None

    parser = argparse.ArgumentParser(description="Export a HuggingFace dataset split to JSONL for CODI inference.")
    parser.add_argument("--output", required=True, help="Destination JSONL file path.")
    parser.add_argument("--repo", default="openai/gsm8k", help="HuggingFace dataset repo ID (default: openai/gsm8k).")
    parser.add_argument("--config", dest="dataset_config", default="main", help="Dataset config/subset (default: main).")
    parser.add_argument("--split", default="test", choices=["train", "test", "validation"], help="Dataset split (default: test).")
    parser.add_argument("--question-field", default="question", help="Column name for questions (default: question).")
    parser.add_argument("--answer-field", default="answer", help="Column name for answers (default: answer).")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max samples to export; -1 = all (default: -1).")
    parser.add_argument("--hf-token", default=hf_token_default, help="HuggingFace auth token (defaults to $HF_TOKEN).")
    args = parser.parse_args()

    prepare_data(
        output_path=args.output,
        repo=args.repo,
        dataset_config=args.dataset_config,
        split=args.split,
        question_field=args.question_field,
        answer_field=args.answer_field,
        max_samples=args.max_samples,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
