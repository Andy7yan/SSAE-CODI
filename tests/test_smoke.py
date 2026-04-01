import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch  # type: ignore
    from datasets import DatasetDict, load_dataset, load_from_disk  # type: ignore
    from huggingface_hub import snapshot_download  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
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
MODEL_REPO = "zen-E/CODI-gpt2"
BASE_MODEL_REPO = "openai-community/gpt2"
DATASET_REPO = "openai/gsm8k"
DATASET_CONFIG = "main"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
NUM_LATENT = 6
TORCH_DTYPE = torch.float32
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


def _model_cache_dir(repo_id: str) -> Path:
    return HF_HUB_CACHE / f"models--{repo_id.replace('/', '--')}"


def _dataset_disk_path(dataset_repo: str, dataset_config: str) -> Path:
    slug = f"{dataset_repo.replace('/', '--')}--{dataset_config}"
    return HF_HOME / "saved-datasets" / slug


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


def _load_state_dict(checkpoint_file: Path) -> dict[str, Any]:
    if checkpoint_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing Python dependency 'safetensors' required to read model.safetensors."
            ) from exc
        return dict(load_file(checkpoint_file))
    return dict(torch.load(checkpoint_file, map_location="cpu"))


def _infer_projection_spec(state_dict: dict[str, Any], hidden_size: int) -> tuple[bool, int, bool]:
    projection_keys = [key for key in state_dict if key.startswith("prj.")]
    if not projection_keys:
        return False, hidden_size, False

    first_linear_weight = state_dict.get("prj.1.weight")
    if first_linear_weight is None:
        return True, hidden_size, any(key.startswith("prj.ln.") for key in projection_keys)

    prj_dim = int(first_linear_weight.shape[0])
    has_layer_norm = any(key.startswith("prj.ln.") for key in projection_keys)
    return True, prj_dim, has_layer_norm


def _resolve_checkpoint_file(snapshot_path: Path) -> Path:
    for file_name in ("model.safetensors", "pytorch_model.bin"):
        candidate = snapshot_path / file_name
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"Unable to find model.safetensors or pytorch_model.bin under {snapshot_path}."
    )


def _load_or_build_model_bundle(snapshot_path: Path) -> tuple[Any, Any]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            token=HF_TOKEN,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            token=HF_TOKEN,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=TORCH_DTYPE,
        )
        model.eval()
        return model, tokenizer
    except Exception:
        checkpoint_file = _resolve_checkpoint_file(snapshot_path)
        state_dict = _load_state_dict(checkpoint_file)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_REPO,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_REPO,
            token=HF_TOKEN,
            trust_remote_code=True,
            use_fast=False,
        )

        original_vocab_size = int(base_model.config.vocab_size)
        hidden_size = int(base_model.config.hidden_size)
        use_prj, prj_dim, has_layer_norm = _infer_projection_spec(state_dict, hidden_size)
        nn = torch.nn

        class MinimalOfficialCodiGpt2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model_name = BASE_MODEL_REPO
                self.codi = base_model
                self.training = False
                self.pad_token_id = original_vocab_size
                self.bot_id = original_vocab_size + 1
                self.eot_id = original_vocab_size + 2
                self.codi.resize_token_embeddings(original_vocab_size + 3)
                self.dim = int(self.codi.config.hidden_size)
                self.num_latent = NUM_LATENT
                self.use_prj = use_prj
                if use_prj:
                    self.prj = nn.Sequential(
                        nn.Dropout(0.0),
                        nn.Linear(self.dim, prj_dim),
                        nn.GELU(),
                        nn.Linear(prj_dim, self.dim),
                    )
                    if has_layer_norm:
                        self.prj.add_module("ln", nn.LayerNorm(self.dim))

            def get_input_embeddings(self) -> Any:
                return self.codi.get_input_embeddings()

        model = MinimalOfficialCodiGpt2()
        model.load_state_dict(state_dict, strict=False)
        model.codi.tie_weights()
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id

        return model, tokenizer


def _validate_model_repo(repo_id: str) -> tuple[str, list[str], str, bool]:
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
        except Exception:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                token=HF_TOKEN,
                cache_dir=str(HF_HUB_CACHE),
            )
    else:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            token=HF_TOKEN,
            cache_dir=str(HF_HUB_CACHE),
        )
    snapshot = Path(snapshot_path).resolve()
    _, tokenizer = _load_or_build_model_bundle(snapshot)
    weight_files = _collect_weight_files(snapshot)
    tokenizer_class = tokenizer.__class__.__name__
    return tokenizer_class, weight_files, str(snapshot), cache_hit


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
