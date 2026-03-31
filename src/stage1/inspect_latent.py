import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from stage1.config import Stage1Config
from stage1.io import append_jsonl, ensure_dir, save_pt, write_json
from stage1.load_model import LoadedModelBundle, load_model_bundle
from stage1.logging_utils import setup_logger


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for latent inspection.") from exc
    return torch


def _extract_last_token_hidden(hidden_states: Any, target_layer_index: int) -> Any:
    selected = hidden_states[target_layer_index]
    return selected[:, -1, :].detach()


def _encode_question_official(bundle: LoadedModelBundle, question: str, logger: Any | None = None) -> tuple[Any, Any]:
    torch = _require_torch()
    tokenizer = bundle.tokenizer
    model = bundle.model

    batch = tokenizer(question, return_tensors="pt")
    device = bundle.model_info["device"]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    boundary_ids = [model.bot_id]
    if getattr(tokenizer, "eos_token_id", None) is not None:
        boundary_ids.insert(0, tokenizer.eos_token_id)

    boundary_tensor = torch.tensor(boundary_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = torch.cat((input_ids, boundary_tensor), dim=1)
    attention_mask = torch.cat((attention_mask, torch.ones_like(boundary_tensor)), dim=1)

    if logger is not None:
        logger.debug("Official CODI encoder input shape for sample %s: %s", question, tuple(input_ids.shape))

    outputs = model.codi(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        past_key_values=None,
    )
    return outputs, input_ids


def _capture_seed_only_official(
    bundle: LoadedModelBundle,
    config: Stage1Config,
    sample_id: str,
    question: str,
    hidden_root: Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    outputs, _ = _encode_question_official(bundle=bundle, question=question, logger=logger)
    hidden = _extract_last_token_hidden(outputs.hidden_states, config.target_layer_index).cpu()

    metadata = {
        "sample_id": sample_id,
        "model_name": bundle.model_name,
        "backend": bundle.backend,
        "capture_mode": "seed-only",
        "requested_capture_mode": config.capture_mode,
        "target_layer_index": config.target_layer_index,
        "shape": list(hidden.shape),
        "dtype": str(hidden.dtype),
        "number_of_latent_steps_configured": config.inf_latent_iterations,
        "special_token_info": bundle.model_info.get("special_token_info", {}),
        "capture_status": "ok",
    }

    output_path = hidden_root / f"{sample_id}__seed-only.pt"
    save_pt(output_path, {"hidden": hidden, "metadata": metadata})
    append_jsonl(hidden_root / "capture_index.jsonl", metadata | {"tensor_path": str(output_path.resolve())})
    return metadata | {"tensor_path": str(output_path.resolve())}


def _capture_per_latent_step_official(
    bundle: LoadedModelBundle,
    config: Stage1Config,
    sample_id: str,
    question: str,
    hidden_root: Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    model = bundle.model

    outputs, _ = _encode_question_official(bundle=bundle, question=question, logger=logger)
    past_key_values = outputs.past_key_values
    step_hidden_states = [_extract_last_token_hidden(outputs.hidden_states, config.target_layer_index).squeeze(0)]

    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if getattr(model, "use_prj", False):
        latent_embd = model.prj(latent_embd)

    for latent_step in range(config.inf_latent_iterations):
        outputs = model.codi(
            inputs_embeds=latent_embd,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        step_hidden = _extract_last_token_hidden(outputs.hidden_states, config.target_layer_index).squeeze(0)
        step_hidden_states.append(step_hidden)

        if logger is not None:
            logger.debug(
                "Captured latent step %s for sample %s with shape %s.",
                latent_step,
                sample_id,
                tuple(step_hidden.shape),
            )

        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if getattr(model, "use_prj", False):
            latent_embd = model.prj(latent_embd)

    trajectory = torch.stack(step_hidden_states, dim=0).cpu()
    metadata = {
        "sample_id": sample_id,
        "model_name": bundle.model_name,
        "backend": bundle.backend,
        "capture_mode": "per-latent-step",
        "requested_capture_mode": config.capture_mode,
        "target_layer_index": config.target_layer_index,
        "shape": list(trajectory.shape),
        "dtype": str(trajectory.dtype),
        "number_of_latent_steps_configured": config.inf_latent_iterations,
        "captured_step_count": int(trajectory.shape[0]),
        "special_token_info": bundle.model_info.get("special_token_info", {}),
        "capture_status": "ok",
    }

    output_path = hidden_root / f"{sample_id}__per-latent-step.pt"
    save_pt(output_path, {"hidden": trajectory, "metadata": metadata})
    append_jsonl(hidden_root / "capture_index.jsonl", metadata | {"tensor_path": str(output_path.resolve())})
    return metadata | {"tensor_path": str(output_path.resolve())}


def _capture_seed_only_generic(
    bundle: LoadedModelBundle,
    config: Stage1Config,
    sample_id: str,
    question: str,
    hidden_root: Path,
) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    batch = tokenizer(question, return_tensors="pt")
    device = bundle.model_info["device"]
    batch = {key: value.to(device) for key, value in batch.items()}

    outputs = model(**batch, output_hidden_states=True, use_cache=False)
    hidden = _extract_last_token_hidden(outputs.hidden_states, config.target_layer_index).cpu()

    metadata = {
        "sample_id": sample_id,
        "model_name": bundle.model_name,
        "backend": bundle.backend,
        "capture_mode": "seed-only",
        "requested_capture_mode": config.capture_mode,
        "target_layer_index": config.target_layer_index,
        "shape": list(hidden.shape),
        "dtype": str(hidden.dtype),
        "number_of_latent_steps_configured": config.inf_latent_iterations,
        "special_token_info": bundle.model_info.get("special_token_info", {}),
        "capture_status": "ok",
    }

    output_path = hidden_root / f"{sample_id}__seed-only.pt"
    save_pt(output_path, {"hidden": hidden, "metadata": metadata})
    append_jsonl(hidden_root / "capture_index.jsonl", metadata | {"tensor_path": str(output_path.resolve())})
    return metadata | {"tensor_path": str(output_path.resolve())}


def capture_hidden_for_sample(
    bundle: LoadedModelBundle,
    config: Stage1Config,
    sample_id: str,
    question: str,
    hidden_root: str | Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    hidden_dir = ensure_dir(hidden_root)

    if bundle.is_official_codi_wrapper:
        if config.capture_mode == "per-latent-step":
            return _capture_per_latent_step_official(
                bundle=bundle,
                config=config,
                sample_id=sample_id,
                question=question,
                hidden_root=hidden_dir,
                logger=logger,
            )
        return _capture_seed_only_official(
            bundle=bundle,
            config=config,
            sample_id=sample_id,
            question=question,
            hidden_root=hidden_dir,
            logger=logger,
        )

    if config.capture_mode == "per-latent-step" and logger is not None:
        logger.warning(
            "per-latent-step capture is not stably accessible for backend %s. Falling back to seed-only.",
            bundle.backend,
        )

    fallback_result = _capture_seed_only_generic(
        bundle=bundle,
        config=config,
        sample_id=sample_id,
        question=question,
        hidden_root=hidden_dir,
    )
    if config.capture_mode == "per-latent-step":
        fallback_result["requested_capture_mode"] = "per-latent-step"
        fallback_result["capture_status"] = "fallback_to_seed_only"
        fallback_result["todo"] = (
            "Attach backend-specific latent hooks or expose an upstream latent trajectory API "
            "before enabling full per-latent-step capture for this backend."
        )
    return fallback_result


def run_capture_only(
    config: Stage1Config | None = None,
    max_samples_override: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    from stage1.io import read_jsonl

    config = config or Stage1Config()
    if max_samples_override is not None:
        config = config.with_overrides(max_samples=max_samples_override)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_run_name = run_name or f"capture_{timestamp}"
    output_root = ensure_dir(config.output_root)
    hidden_dir = ensure_dir(output_root / "hidden" / effective_run_name)
    log_file = output_root / "logs" / f"{effective_run_name}.log"
    logger = setup_logger("stage1.inspect_latent", log_file)

    logger.info("Starting capture-only run %s.", effective_run_name)
    bundle = load_model_bundle(config=config, logger=logger)
    samples = read_jsonl(config.data_file)[: config.max_samples]
    capture_records: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = str(sample.get("sample_id", f"sample_{len(capture_records)}"))
        question = str(sample["question"])
        started = time.perf_counter()
        record = capture_hidden_for_sample(
            bundle=bundle,
            config=config,
            sample_id=sample_id,
            question=question,
            hidden_root=hidden_dir,
            logger=logger,
        )
        record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        capture_records.append(record)
        logger.info("Captured hidden states for sample %s.", sample_id)

    summary = {
        "run_name": effective_run_name,
        "capture_mode": config.capture_mode,
        "sample_count": len(capture_records),
        "hidden_dir": str(hidden_dir.resolve()),
        "log_file": str(log_file.resolve()),
    }
    write_json(hidden_dir / "capture_summary.json", summary)
    logger.info("Capture-only run complete. Summary: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Stage 1 CODI hidden states without running full inference.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample override.")
    parser.add_argument("--run-name", default=None, help="Optional run name override.")
    args = parser.parse_args()

    run_capture_only(
        max_samples_override=args.max_samples,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
