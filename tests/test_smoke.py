import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from datasets import DatasetDict, load_dataset, load_from_disk  # type: ignore
    from model import HF_TOKEN, MODEL_REPO, get_special_token_ids, load_codi_gpt2  # type: ignore
except ModuleNotFoundError as exc:
    missing_pkg = exc.name or "a required package"
    raise ModuleNotFoundError(
        f"Missing Python dependency '{missing_pkg}' for tests/test_smoke.py.\n"
        "Install the project dependencies in the same environment used by torchrun:\n"
        "  python3 -m pip install -r requirements-katana.txt\n"
        "  python3 -m pip install -e .\n"
        "Then retry:\n"
        "  torchrun --nproc_per_node=1 tests/test_smoke.py"
    ) from exc


# Hard-coded smoke configuration.
HF_HOME = Path("/srv/scratch/z5534565/ssae-codi/hf-home").resolve()
HF_HUB_CACHE = (HF_HOME / "hub").resolve()
HF_DATASETS_CACHE = (HF_HOME / "datasets").resolve()
SMOKE_OUTPUT_ROOT = Path("/srv/scratch/z5534565/ssae-codi/smoke").resolve()
DATASET_REPO = "openai/gsm8k"
DATASET_CONFIG = "main"
TRAIN_SAMPLE_COUNT = 50
SAMPLE_SEED = 42


@dataclass(slots=True)
class SmokeSummary:
    model_repo: str
    model_snapshot_path: str
    model_cache_hit: bool
    weight_files: list[str]
    tokenizer_class: str
    dataset_repo: str
    dataset_config: str
    dataset_cache_path: str
    dataset_cache_hit: bool
    dataset_split_sizes: dict[str, int]
    dataset_sample_path: str
    first_train_question: str
    first_train_answer: str
    hf_home: str
    summary_path: str


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dataset_disk_path(dataset_repo: str, dataset_config: str) -> Path:
    slug = f"{dataset_repo.replace('/', '--')}--{dataset_config}"
    return HF_HOME / "saved-datasets" / slug


def _validate_model_repo(repo_id: str) -> tuple[str, list[str], str, bool]:
    if repo_id != MODEL_REPO:
        raise RuntimeError(f"Unsupported smoke model repository: {repo_id}")

    model, tokenizer = load_codi_gpt2(HF_HOME, HF_HUB_CACHE)
    weight_files = list(model._codi_weight_files)
    tokenizer_class = tokenizer.__class__.__name__
    return tokenizer_class, weight_files, model._codi_snapshot_path, bool(model._codi_cache_hit)


def _load_cached_or_remote_dataset(dataset_repo: str, dataset_config: str) -> tuple[DatasetDict, Path, bool]:
    disk_path = _dataset_disk_path(dataset_repo, dataset_config)
    if disk_path.exists():
        dataset = load_from_disk(str(disk_path))
        return dataset, disk_path, True

    dataset = load_dataset(
        dataset_repo,
        dataset_config,
        cache_dir=str(HF_DATASETS_CACHE),
        token=HF_TOKEN,
    )
    _ensure_dir(disk_path.parent)
    dataset.save_to_disk(str(disk_path))
    return dataset, disk_path, False


def _validate_dataset(dataset: DatasetDict) -> tuple[dict[str, int], str, str]:
    if "train" not in dataset:
        raise RuntimeError("GSM8K dataset does not contain the expected 'train' split.")
    if len(dataset["train"]) == 0:
        raise RuntimeError("GSM8K train split is empty.")

    first_row: dict[str, Any] = dataset["train"][0]
    question = str(first_row.get("question", "")).strip()
    answer = str(first_row.get("answer", "")).strip()
    if not question or not answer:
        raise RuntimeError("GSM8K first row is missing a question or answer.")

    split_sizes = {split_name: len(split_data) for split_name, split_data in dataset.items()}
    return split_sizes, question, answer


def _export_train_samples(
    dataset: DatasetDict,
    output_root: Path,
    sample_count: int = TRAIN_SAMPLE_COUNT,
    seed: int = SAMPLE_SEED,
) -> Path:
    if "train" not in dataset:
        raise RuntimeError("GSM8K dataset does not contain the expected 'train' split.")

    train_split = dataset["train"]
    total = len(train_split)
    if total == 0:
        raise RuntimeError("GSM8K train split is empty.")

    rng = random.Random(seed)
    take_n = min(sample_count, total)
    sample_indices = sorted(rng.sample(range(total), k=take_n))
    samples = [dict(train_split[int(idx)]) for idx in sample_indices]

    sample_path = output_root / "train_samples_50.json"
    sample_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sample_path


def run_smoke() -> SmokeSummary:
    _ensure_dir(HF_HOME)
    _ensure_dir(HF_HUB_CACHE)
    _ensure_dir(HF_DATASETS_CACHE)
    _ensure_dir(SMOKE_OUTPUT_ROOT)

    tokenizer_class, weight_files, model_snapshot_path, model_cache_hit = _validate_model_repo(MODEL_REPO)
    dataset, dataset_cache_path, dataset_cache_hit = _load_cached_or_remote_dataset(DATASET_REPO, DATASET_CONFIG)
    dataset_split_sizes, first_train_question, first_train_answer = _validate_dataset(dataset)
    dataset_sample_path = _export_train_samples(dataset, SMOKE_OUTPUT_ROOT)

    summary_path = SMOKE_OUTPUT_ROOT / "smoke_summary.json"
    summary = SmokeSummary(
        model_repo=MODEL_REPO,
        model_snapshot_path=model_snapshot_path,
        model_cache_hit=model_cache_hit,
        weight_files=weight_files,
        tokenizer_class=tokenizer_class,
        dataset_repo=DATASET_REPO,
        dataset_config=DATASET_CONFIG,
        dataset_cache_path=str(dataset_cache_path.resolve()),
        dataset_cache_hit=dataset_cache_hit,
        dataset_split_sizes=dataset_split_sizes,
        dataset_sample_path=str(dataset_sample_path.resolve()),
        first_train_question=first_train_question,
        first_train_answer=first_train_answer,
        hf_home=str(HF_HOME),
        summary_path=str(summary_path.resolve()),
    )
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank != 0:
        return 0

    summary = run_smoke()
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
