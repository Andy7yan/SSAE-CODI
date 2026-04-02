from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

MODEL_REPO = "zen-E/CODI-gpt2"
BASE_MODEL_REPO = "openai-community/gpt2"
NUM_LATENT = 6
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
TORCH_DTYPE = torch.float32


def _model_cache_dir(repo_id: str, hf_hub_cache: Path) -> Path:
    return hf_hub_cache / f"models--{repo_id.replace('/', '--')}"


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


def _load_or_build_model_bundle(
    snapshot_path: Path,
    dtype: torch.dtype = TORCH_DTYPE,
) -> tuple[nn.Module, PreTrainedTokenizer]:
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
            torch_dtype=dtype,
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
            torch_dtype=dtype,
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


def load_codi_gpt2(
    hf_home: Path,
    hf_hub_cache: Path | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, PreTrainedTokenizer]:
    resolved_hf_home = Path(hf_home).resolve()
    resolved_hf_hub_cache = (
        Path(hf_hub_cache).resolve() if hf_hub_cache is not None else (resolved_hf_home / "hub").resolve()
    )

    cache_dir = _model_cache_dir(MODEL_REPO, resolved_hf_hub_cache)
    cache_hit = cache_dir.exists()

    if cache_hit:
        try:
            snapshot_path = snapshot_download(
                repo_id=MODEL_REPO,
                token=HF_TOKEN,
                cache_dir=str(resolved_hf_hub_cache),
                local_files_only=True,
            )
        except Exception:
            snapshot_path = snapshot_download(
                repo_id=MODEL_REPO,
                token=HF_TOKEN,
                cache_dir=str(resolved_hf_hub_cache),
            )
    else:
        snapshot_path = snapshot_download(
            repo_id=MODEL_REPO,
            token=HF_TOKEN,
            cache_dir=str(resolved_hf_hub_cache),
        )

    snapshot = Path(snapshot_path).resolve()
    weight_files = _collect_weight_files(snapshot)
    model, tokenizer = _load_or_build_model_bundle(snapshot, dtype=dtype)

    # Attach smoke metadata so callers can reuse the same loader without duplicating cache logic.
    model._codi_snapshot_path = str(snapshot)
    model._codi_weight_files = weight_files
    model._codi_cache_hit = cache_hit

    return model, tokenizer


def get_special_token_ids(model: nn.Module) -> dict[str, int]:
    return {
        "pad_id": model.pad_token_id,
        "bot_id": model.bot_id,
        "eot_id": model.eot_id,
    }
