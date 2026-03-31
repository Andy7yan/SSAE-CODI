import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import RunConfig
from inspect_latent import capture_hidden_for_sample
from io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from load_model import LoadedModelBundle, load_model_bundle
from logging_utils import setup_logger


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for CODI inference.") from exc
    return torch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        torch = _require_torch()
    except RuntimeError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prompt(bundle: LoadedModelBundle, question: str) -> str:
    if bundle.is_official_codi_wrapper:
        return question

    tokenizer = bundle.tokenizer
    additional = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    if "<|start-latent|>" in additional and "<|start-latent|>" not in question:
        return question + "<|start-latent|>"
    return question


def _sample_next_token(logits: Any, config: RunConfig) -> Any:
    torch = _require_torch()
    if not config.do_sample or config.temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled_logits = logits / config.temperature
    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > config.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        scaled_logits = scaled_logits.clone()
        for batch_index in range(scaled_logits.size(0)):
            scaled_logits[batch_index, sorted_indices[batch_index, sorted_indices_to_remove[batch_index]]] = -float("inf")

    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _run_official_codi_inference(
    bundle: LoadedModelBundle,
    config: RunConfig,
    question: str,
) -> dict[str, Any]:
    torch = _require_torch()
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.model_info["device"]

    batch = tokenizer(question, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    boundary_ids = [model.bot_id]
    if getattr(tokenizer, "eos_token_id", None) is not None:
        boundary_ids.insert(0, tokenizer.eos_token_id)

    boundary_tensor = torch.tensor(boundary_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = torch.cat((input_ids, boundary_tensor), dim=1)
    attention_mask = torch.cat((attention_mask, torch.ones_like(boundary_tensor)), dim=1)

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if getattr(model, "use_prj", False):
            latent_embd = model.prj(latent_embd)

        for _ in range(config.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if getattr(model, "use_prj", False):
                latent_embd = model.prj(latent_embd)

        end_ids = [model.eot_id]
        if getattr(tokenizer, "eos_token_id", None) is not None:
            end_ids.append(tokenizer.eos_token_id)

        end_tensor = torch.tensor(end_ids, dtype=torch.long, device=device)
        output_embeds = model.get_input_embeddings()(end_tensor).unsqueeze(0)

        generated_token_ids: list[int] = []
        for _ in range(config.max_new_tokens):
            outputs = model.codi(
                inputs_embeds=output_embeds,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, : model.codi.config.vocab_size - 1]
            next_token_id = _sample_next_token(logits, config=config)
            next_token_value = int(next_token_id[0].item())

            if getattr(tokenizer, "eos_token_id", None) is not None and next_token_value == tokenizer.eos_token_id:
                break

            generated_token_ids.append(next_token_value)
            output_embeds = model.get_input_embeddings()(next_token_id).unsqueeze(1)

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "input_token_count": int(input_ids.shape[1]),
        "generated_token_count": len(generated_token_ids),
        "latent_iterations": config.inf_latent_iterations,
    }


def _run_generic_inference(
    bundle: LoadedModelBundle,
    config: RunConfig,
    prompt: str,
) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.model_info["device"]

    batch = tokenizer(prompt, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if config.do_sample and config.temperature > 0.0:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    with _require_torch().no_grad():
        output = model.generate(**batch, **generation_kwargs)

    prompt_length = int(batch["input_ids"].shape[1])
    generated_token_ids = output[0][prompt_length:].detach().cpu().tolist()
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "input_token_count": prompt_length,
        "generated_token_count": len(generated_token_ids),
        "latent_iterations": config.inf_latent_iterations,
    }


def run_single_sample(
    bundle: LoadedModelBundle,
    config: RunConfig,
    question: str,
) -> dict[str, Any]:
    prompt = _build_prompt(bundle=bundle, question=question)
    if bundle.is_official_codi_wrapper:
        result = _run_official_codi_inference(bundle=bundle, config=config, question=prompt)
    else:
        result = _run_generic_inference(bundle=bundle, config=config, prompt=prompt)
    result["prompt"] = prompt
    return result


def _create_run_layout(config: RunConfig, run_name: str) -> dict[str, Path]:
    output_root = ensure_dir(config.output_root)
    run_dir = ensure_dir(output_root / run_name)

    return {
        "output_root": output_root,
        "run_dir": run_dir,
        "inference_dir": run_dir,
        "hidden_dir": run_dir,
        "log_file": run_dir / "run.log",
    }


def run_inference_job(
    config: RunConfig | None = None,
    max_samples_override: int | None = None,
    output_dir_override: str | None = None,
    capture_hidden_override: bool | None = None,
    capture_mode_override: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    config = config or RunConfig()
    overrides: dict[str, Any] = {}
    if max_samples_override is not None:
        overrides["max_samples"] = max_samples_override
    if output_dir_override is not None:
        overrides["output_dir"] = output_dir_override
    if capture_hidden_override is not None:
        overrides["capture_hidden"] = capture_hidden_override
    if capture_mode_override is not None:
        overrides["capture_mode"] = capture_mode_override
    if overrides:
        config = config.with_overrides(**overrides)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_run_name = run_name or f"run_{timestamp}"
    run_layout = _create_run_layout(config=config, run_name=effective_run_name)
    logger = setup_logger("run_inference", run_layout["log_file"])

    logger.info("Starting inference run: %s", effective_run_name)
    _set_seed(config.seed)

    bundle = load_model_bundle(config=config, logger=logger)

    write_json(run_layout["run_dir"] / "effective_config.json", config.to_dict())
    write_json(run_layout["run_dir"] / "model_info.json", bundle.model_info)

    samples = read_jsonl(config.data_file)[: config.max_samples]
    logger.info("Loaded %s debug sample(s) from %s.", len(samples), config.data_file.resolve())

    result_rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", f"sample_{len(result_rows)}"))
        question = str(sample["question"])
        answer = sample.get("answer")

        logger.info("Running sample %s.", sample_id)
        started = time.perf_counter()
        generation_result = run_single_sample(bundle=bundle, config=config, question=question)
        elapsed_seconds = round(time.perf_counter() - started, 6)

        hidden_record: dict[str, Any] | None = None
        if config.capture_hidden:
            hidden_record = capture_hidden_for_sample(
                bundle=bundle,
                config=config,
                sample_id=sample_id,
                question=question,
                hidden_root=run_layout["hidden_dir"],
                logger=logger,
            )

        sample_record = {
            "sample_id": sample_id,
            "question": question,
            "reference_answer": answer,
            "prompt": generation_result["prompt"],
            "generated_text": generation_result["generated_text"],
            "generated_token_ids": generation_result["generated_token_ids"],
            "model_name": bundle.model_name,
            "backend": bundle.backend,
            "latent_iterations": generation_result["latent_iterations"],
            "input_token_count": generation_result["input_token_count"],
            "generated_token_count": generation_result["generated_token_count"],
            "elapsed_seconds": elapsed_seconds,
            "capture_mode": config.capture_mode if config.capture_hidden else None,
            "hidden_record": hidden_record,
        }
        result_rows.append(sample_record)

        write_json(run_layout["run_dir"] / f"{sample_id}.json", sample_record)
        logger.info(
            "Completed sample %s in %.3fs. Generated text: %s",
            sample_id,
            elapsed_seconds,
            sample_record["generated_text"],
        )

    write_jsonl(run_layout["run_dir"] / "results.jsonl", result_rows)

    summary = {
        "run_name": effective_run_name,
        "sample_count": len(result_rows),
        "model_name": bundle.model_name,
        "backend": bundle.backend,
        "capture_hidden": config.capture_hidden,
        "capture_mode": config.capture_mode if config.capture_hidden else None,
        "output_root": str(run_layout["output_root"].resolve()),
        "run_dir": str(run_layout["run_dir"].resolve()),
        "inference_dir": str(run_layout["run_dir"].resolve()),
        "hidden_dir": str(run_layout["run_dir"].resolve()),
        "log_file": str(run_layout["log_file"].resolve()),
    }
    write_json(run_layout["run_dir"] / "run_summary.json", summary)
    logger.info("Inference run finished. Summary: %s", summary)
    return summary


run_stage1 = run_inference_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CODI inference and optional hidden capture.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample override.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--capture-hidden", action="store_true", help="Force hidden capture on.")
    parser.add_argument("--no-capture-hidden", action="store_true", help="Force hidden capture off.")
    parser.add_argument(
        "--capture-mode",
        choices=["seed-only", "per-latent-step"],
        default=None,
        help="Optional capture mode override.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run name override.")
    args = parser.parse_args()

    if args.capture_hidden and args.no_capture_hidden:
        raise ValueError("Use only one of --capture-hidden or --no-capture-hidden.")

    capture_hidden_override: bool | None = None
    if args.capture_hidden:
        capture_hidden_override = True
    elif args.no_capture_hidden:
        capture_hidden_override = False

    run_inference_job(
        max_samples_override=args.max_samples,
        output_dir_override=args.output_dir,
        capture_hidden_override=capture_hidden_override,
        capture_mode_override=args.capture_mode,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
