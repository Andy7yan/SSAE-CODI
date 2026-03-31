import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset, load_from_disk  # type: ignore
from huggingface_hub import snapshot_download  # type: ignore
from transformers import AutoConfig, AutoTokenizer  # type: ignore


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


HF_HOME = _env_path("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
HF_HUB_CACHE = _env_path("HF_HUB_CACHE", str(HF_HOME / "hub"))
HF_DATASETS_CACHE = _env_path("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))
SMOKE_OUTPUT_ROOT = _env_path("SMOKE_OUTPUT_ROOT", str(HF_HOME / "smoke"))
MODEL_REPO = os.getenv("MODEL_REPO", "zen-E/CODI-gpt2")
DATASET_REPO = os.getenv("DATASET_REPO", "openai/gsm8k")
DATASET_CONFIG = os.getenv("DATASET_CONFIG", "main")
HF_TOKEN = os.getenv("HF_TOKEN") or None


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
    first_train_question: str
    first_train_answer: str
    hf_home: str
    summary_path: str


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_cache_dir(repo_id: str) -> Path:
    return HF_HUB_CACHE / f"models--{repo_id.replace('/', '--')}"


def _dataset_disk_path(dataset_repo: str, dataset_config: str) -> Path:
    slug = f"{dataset_repo.replace('/', '--')}--{dataset_config}"
    return HF_HOME / "saved-datasets" / slug


def _load_cached_or_remote_model_snapshot(repo_id: str) -> tuple[Path, bool]:
    cache_dir = _model_cache_dir(repo_id)
    cache_hit = cache_dir.exists()

    if cache_hit:
        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                token=HF_TOKEN,
                cache_dir=str(HF_HUB_CACHE),
                local_files_only=True,
            )
            return Path(snapshot_path), True
        except Exception:
            cache_hit = False

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        token=HF_TOKEN,
        cache_dir=str(HF_HUB_CACHE),
    )
    return Path(snapshot_path), cache_hit


def _collect_weight_files(snapshot_path: Path) -> list[str]:
    candidates = [
        *snapshot_path.glob("*.safetensors"),
        *snapshot_path.glob("*.bin"),
        *snapshot_path.glob("*.index.json"),
    ]
    weight_files = sorted({path.name for path in candidates})
    if not weight_files:
        raise RuntimeError(f"No model weight files were found under {snapshot_path}.")
    return weight_files


def _validate_model_repo(snapshot_path: Path) -> tuple[str, list[str]]:
    AutoConfig.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
        token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
        token=HF_TOKEN,
        use_fast=False,
    )
    weight_files = _collect_weight_files(snapshot_path)
    return tokenizer.__class__.__name__, weight_files


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


def run_smoke() -> SmokeSummary:
    _ensure_dir(HF_HOME)
    _ensure_dir(HF_HUB_CACHE)
    _ensure_dir(HF_DATASETS_CACHE)
    _ensure_dir(SMOKE_OUTPUT_ROOT)

    snapshot_path, model_cache_hit = _load_cached_or_remote_model_snapshot(MODEL_REPO)
    tokenizer_class, weight_files = _validate_model_repo(snapshot_path)

    dataset, dataset_cache_path, dataset_cache_hit = _load_cached_or_remote_dataset(DATASET_REPO, DATASET_CONFIG)
    dataset_split_sizes, first_train_question, first_train_answer = _validate_dataset(dataset)

    summary_path = SMOKE_OUTPUT_ROOT / "smoke_summary.json"
    summary = SmokeSummary(
        model_repo=MODEL_REPO,
        model_snapshot_path=str(snapshot_path.resolve()),
        model_cache_hit=model_cache_hit,
        weight_files=weight_files,
        tokenizer_class=tokenizer_class,
        dataset_repo=DATASET_REPO,
        dataset_config=DATASET_CONFIG,
        dataset_cache_path=str(dataset_cache_path.resolve()),
        dataset_cache_hit=dataset_cache_hit,
        dataset_split_sizes=dataset_split_sizes,
        first_train_question=first_train_question,
        first_train_answer=first_train_answer,
        hf_home=str(HF_HOME.resolve()),
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
