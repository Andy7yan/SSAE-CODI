from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage1.config import Stage1Config


class ModelLoadError(RuntimeError):
    """Raised when Stage 1 cannot construct a usable model bundle."""


@dataclass(slots=True)
class LoadedModelBundle:
    model: Any
    tokenizer: Any
    model_name: str
    backend: str
    base_model_name: str
    model_info: dict[str, Any]

    @property
    def is_official_codi_wrapper(self) -> bool:
        return hasattr(self.model, "codi") and hasattr(self.model, "bot_id") and hasattr(self.model, "eot_id")


def _resolve_auth_token(env_var_name: str) -> str | None:
    token = os.getenv(env_var_name)
    return token.strip() if token else None


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ModelLoadError(
            "PyTorch is required for Stage 1 model loading. Install Stage 1 dependencies first."
        ) from exc
    return torch


def _resolve_dtype(dtype_name: str, torch: Any) -> Any:
    normalized = dtype_name.lower()
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ModelLoadError(f"Unsupported dtype value: {dtype_name}")
    return mapping[normalized]


def _resolve_device(device_name: str, torch: Any, logger: Any | None = None) -> str:
    requested = device_name.lower()
    if requested == "auto":
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        if logger is not None:
            logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"

    return requested


def _load_tokenizer(tokenizer_name: str, token: str | None, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        token=token,
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )


def _load_direct_hf_bundle(config: Stage1Config, token: str | None, logger: Any | None = None) -> LoadedModelBundle:
    torch = _require_torch()
    from transformers import AutoModelForCausalLM  # type: ignore

    resolved_dtype = _resolve_dtype(config.dtype, torch)
    resolved_device = _resolve_device(config.device, torch, logger=logger)

    tokenizer = _load_tokenizer(
        tokenizer_name=config.model_name_or_path,
        token=token,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        token=token,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=resolved_dtype,
    )
    model = model.to(resolved_device)
    model.eval()

    model_info = _collect_model_info(
        model=model,
        tokenizer=tokenizer,
        config=config,
        backend="direct-hf",
        base_model_name=config.model_name_or_path,
        checkpoint_source=config.model_name_or_path,
        device=resolved_device,
    )

    return LoadedModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name_or_path,
        backend="direct-hf",
        base_model_name=config.model_name_or_path,
        model_info=model_info,
    )


def _resolve_checkpoint_file(model_name_or_path: str, token: str | None) -> tuple[Path, str]:
    candidate_root = Path(model_name_or_path)
    if candidate_root.exists():
        local_candidates = (
            candidate_root / "model.safetensors",
            candidate_root / "pytorch_model.bin",
        )
        for candidate in local_candidates:
            if candidate.exists():
                return candidate.resolve(), "local-path"
        raise ModelLoadError(f"No checkpoint file was found under {candidate_root}.")

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:
        raise ModelLoadError(
            "huggingface-hub is required to download the CODI checkpoint from Hugging Face."
        ) from exc

    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            downloaded = hf_hub_download(repo_id=model_name_or_path, filename=filename, token=token)
            return Path(downloaded), "huggingface-hub"
        except Exception:
            continue

    raise ModelLoadError(
        f"Unable to download a supported checkpoint artifact for {model_name_or_path}. "
        "Expected model.safetensors or pytorch_model.bin."
    )


def _load_state_dict(checkpoint_file: Path) -> dict[str, Any]:
    torch = _require_torch()

    if checkpoint_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:
            raise ModelLoadError("safetensors is required to load .safetensors checkpoints.") from exc
        return dict(load_file(checkpoint_file))

    return dict(torch.load(checkpoint_file, map_location="cpu"))


def _infer_base_model_name(model_name_or_path: str) -> str:
    normalized = model_name_or_path.lower()
    if "codi-gpt2" in normalized:
        return "openai-community/gpt2"
    return model_name_or_path


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


def _build_official_codi_gpt2_bundle(
    config: Stage1Config,
    token: str | None,
    logger: Any | None = None,
) -> LoadedModelBundle:
    torch = _require_torch()
    from transformers import AutoModelForCausalLM  # type: ignore

    resolved_dtype = _resolve_dtype(config.dtype, torch)
    resolved_device = _resolve_device(config.device, torch, logger=logger)
    base_model_name = _infer_base_model_name(config.model_name_or_path)

    checkpoint_file, _checkpoint_source = _resolve_checkpoint_file(config.model_name_or_path, token)
    state_dict = _load_state_dict(checkpoint_file)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=token,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=resolved_dtype,
    )
    tokenizer = _load_tokenizer(
        tokenizer_name=base_model_name,
        token=token,
        trust_remote_code=config.trust_remote_code,
    )

    original_vocab_size = int(base_model.config.vocab_size)
    hidden_size = int(base_model.config.hidden_size)
    use_prj, prj_dim, has_layer_norm = _infer_projection_spec(state_dict, hidden_size=hidden_size)
    nn = torch.nn

    class MinimalOfficialCodiGpt2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_name = base_model_name
            self.codi = base_model
            self.training = False
            self.pad_token_id = original_vocab_size
            self.bot_id = original_vocab_size + 1
            self.eot_id = original_vocab_size + 2
            self.codi.resize_token_embeddings(original_vocab_size + 3)
            self.dim = int(self.codi.config.hidden_size)
            self.num_latent = config.num_latent
            self.use_prj = use_prj
            self.prj_no_ln = not has_layer_norm
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
    load_result = model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = model.pad_token_id

    model = model.to(resolved_device)
    model.eval()

    if logger is not None:
        if load_result.missing_keys:
            logger.warning("Missing keys during official CODI wrapper load: %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logger.warning("Unexpected keys during official CODI wrapper load: %s", load_result.unexpected_keys)

    model_info = _collect_model_info(
        model=model,
        tokenizer=tokenizer,
        config=config,
        backend="official-codi-gpt2-wrapper",
        base_model_name=base_model_name,
        checkpoint_source=str(checkpoint_file),
        device=resolved_device,
    )

    return LoadedModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=config.model_name_or_path,
        backend="official-codi-gpt2-wrapper",
        base_model_name=base_model_name,
        model_info=model_info,
    )


def _collect_model_info(
    model: Any,
    tokenizer: Any,
    config: Stage1Config,
    backend: str,
    base_model_name: str,
    checkpoint_source: str,
    device: str,
) -> dict[str, Any]:
    core_model = getattr(model, "codi", model)
    model_config = getattr(core_model, "config", None)

    special_token_info: dict[str, Any] = {}
    if hasattr(tokenizer, "special_tokens_map"):
        special_token_info["tokenizer_special_tokens"] = dict(tokenizer.special_tokens_map)
    if hasattr(tokenizer, "additional_special_tokens"):
        special_token_info["additional_special_tokens"] = list(getattr(tokenizer, "additional_special_tokens"))
    if hasattr(model, "pad_token_id"):
        special_token_info["pad_token_id"] = int(model.pad_token_id)
    if hasattr(model, "bot_id"):
        special_token_info["bot_id"] = int(model.bot_id)
    if hasattr(model, "eot_id"):
        special_token_info["eot_id"] = int(model.eot_id)

    return {
        "model_name": config.model_name_or_path,
        "backend": backend,
        "base_model_name": base_model_name,
        "checkpoint_source": checkpoint_source,
        "device": device,
        "dtype": config.dtype,
        "hidden_size": getattr(model_config, "hidden_size", None),
        "vocab_size": getattr(model_config, "vocab_size", None),
        "number_of_latent_steps_configured": config.inf_latent_iterations,
        "num_latent_training_config": config.num_latent,
        "capture_mode": config.capture_mode,
        "special_token_info": special_token_info,
        "uses_projection_layer": bool(getattr(model, "use_prj", False)),
    }


def validate_model_bundle(bundle: LoadedModelBundle) -> None:
    if bundle.model is None:
        raise ModelLoadError("Model bundle validation failed: model is None.")
    if bundle.tokenizer is None:
        raise ModelLoadError("Model bundle validation failed: tokenizer is None.")
    if not hasattr(bundle.model, "eval"):
        raise ModelLoadError("Model bundle validation failed: model does not expose eval().")
    if bundle.model_info.get("hidden_size") in {None, 0}:
        raise ModelLoadError("Model bundle validation failed: hidden_size is missing.")
    if bundle.model_info.get("vocab_size") in {None, 0}:
        raise ModelLoadError("Model bundle validation failed: vocab_size is missing.")


def _log_model_info(bundle: LoadedModelBundle, logger: Any | None = None) -> None:
    lines = [
        f"Model backend: {bundle.backend}",
        f"Model name: {bundle.model_info['model_name']}",
        f"Base model: {bundle.model_info['base_model_name']}",
        f"Checkpoint source: {bundle.model_info['checkpoint_source']}",
        f"Device: {bundle.model_info['device']}",
        f"Hidden size: {bundle.model_info['hidden_size']}",
        f"Vocab size: {bundle.model_info['vocab_size']}",
        f"Configured latent steps: {bundle.model_info['number_of_latent_steps_configured']}",
        f"Capture mode: {bundle.model_info['capture_mode']}",
        f"Special tokens: {bundle.model_info['special_token_info']}",
    ]
    if logger is None:
        for line in lines:
            print(line)
        return
    for line in lines:
        logger.info(line)


def load_model_bundle(config: Stage1Config, logger: Any | None = None) -> LoadedModelBundle:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    token = _resolve_auth_token(config.hf_token_env_var)
    if logger is not None:
        if token:
            logger.info("Using Hugging Face token from environment variable %s.", config.hf_token_env_var)
        else:
            logger.info(
                "No Hugging Face token was found in %s. Proceeding without authentication.",
                config.hf_token_env_var,
            )

    try:
        bundle = _load_direct_hf_bundle(config=config, token=token, logger=logger)
    except Exception as direct_error:
        if logger is not None:
            logger.warning(
                "Direct Hugging Face loading failed for %s. Falling back to the official CODI GPT-2 wrapper. "
                "Reason: %s",
                config.model_name_or_path,
                direct_error,
            )
        bundle = _build_official_codi_gpt2_bundle(config=config, token=token, logger=logger)

    validate_model_bundle(bundle)
    _log_model_info(bundle, logger=logger)
    return bundle
