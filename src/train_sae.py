"""Training script for StepwiseSAEBank on CODI latent trajectories.

Usage (Katana):
    python src/train_sae.py \\
        --dataset-dir /srv/scratch/$USER/ssae-codi/datasets/gsm8k-latents-v1 \\
        --output-dir  $RUN_DIR \\
        --n-steps 7 --d-sae 3072 --k 32 \\
        --lr 1e-4 --epochs 10 --batch-size 64

Assumes dataset-dir contains:
    train.safetensors  -- shape (N_train, n_steps, hidden_size)
    test.safetensors   -- shape (N_test,  n_steps, hidden_size)
    meta.json          -- dataset metadata
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from io_utils import ensure_dir, load_safetensors, write_json
from logging_utils import setup_logger
from sae import StepwiseSAEBank, sae_loss


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


def _load_split(dataset_dir: Path, split: str) -> torch.Tensor:
    path = dataset_dir / f"{split}.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    tensors = load_safetensors(path)
    return tensors["hidden"].float()  # (N, n_steps, hidden_size)


def train(
    dataset_dir: str | Path,
    output_dir: str | Path,
    n_steps: int,
    d_sae: int,
    k: int,
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 64,
    log_every: int = 50,
    device_str: str = "auto",
    l1_coeff: float = 0.0,
    seed: int = 0,
) -> None:
    """Train a StepwiseSAEBank and save checkpoints + final weights.

    Args:
        dataset_dir: Path to directory with train.safetensors / test.safetensors / meta.json.
        output_dir: Run output root; checkpoints/ and weights/ are created here.
        n_steps: Number of latent steps (must match dataset).
        d_sae: SAE feature dimension.
        k: TopK sparsity.
        lr: AdamW learning rate.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        log_every: Log training loss every N steps.
        device_str: "auto", "cuda", or "cpu".
        l1_coeff: L1 sparsity coefficient (0 disables it; TopK already enforces sparsity).
        seed: Random seed.
    """
    torch.manual_seed(seed)
    dataset_path = Path(dataset_dir)
    out_dir = ensure_dir(output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    weights_dir = ensure_dir(out_dir / "weights")
    log_file = out_dir / "train_sae.log"
    logger = setup_logger("train_sae", log_file)

    # Device selection.
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # Load data.
    train_hidden = _load_split(dataset_path, "train").to(device)  # (N, n_steps, d_model)
    test_hidden = _load_split(dataset_path, "test").to(device) if (dataset_path / "test.safetensors").exists() else None

    n_samples, actual_steps, d_model = train_hidden.shape
    if actual_steps != n_steps:
        raise ValueError(f"Dataset n_steps={actual_steps} does not match --n-steps={n_steps}.")

    logger.info(
        "Loaded train set: %d samples, n_steps=%d, d_model=%d", n_samples, actual_steps, d_model
    )

    train_loader = DataLoader(TensorDataset(train_hidden), batch_size=batch_size, shuffle=True)

    # Model + optimizer.
    model = StepwiseSAEBank(n_steps=n_steps, d_model=d_model, d_sae=d_sae, k=k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logger.info(
        "Model: StepwiseSAEBank | n_steps=%d d_model=%d d_sae=%d k=%d", n_steps, d_model, d_sae, k
    )

    # Read dataset meta for bookkeeping.
    meta_path = dataset_path / "meta.json"
    dataset_meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        epoch_recon_loss = 0.0

        for (batch,) in train_loader:
            optimizer.zero_grad()
            h_hat_seq, z_seq = model(batch)
            losses = sae_loss(batch, h_hat_seq, z_seq, l1_coeff=l1_coeff)
            losses["total_loss"].backward()
            optimizer.step()
            model.normalize_decoders()

            epoch_recon_loss += losses["reconstruction_loss"].item()
            global_step += 1

            if global_step % log_every == 0:
                logger.info(
                    "step=%d  recon=%.6f  sparsity=%.6f  total=%.6f",
                    global_step,
                    losses["reconstruction_loss"].item(),
                    losses["sparsity_loss"].item(),
                    losses["total_loss"].item(),
                )

        avg_recon = epoch_recon_loss / max(len(train_loader), 1)
        logger.info(
            "Epoch %d/%d done in %.1fs | avg_recon=%.6f",
            epoch,
            epochs,
            time.perf_counter() - epoch_start,
            avg_recon,
        )

        # Save per-epoch checkpoint.
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, ckpt_path)
        logger.info("Checkpoint saved: %s", ckpt_path)

    # Save final weights.
    final_path = weights_dir / "sae_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Final weights saved: %s", final_path)

    # Write run summary.
    summary = {
        "n_steps": n_steps,
        "d_model": d_model,
        "d_sae": d_sae,
        "k": k,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "l1_coeff": l1_coeff,
        "seed": seed,
        "device": str(device),
        "n_train_samples": n_samples,
        "global_steps": global_step,
        "dataset_meta": dataset_meta,
        "final_weights": str(final_path.resolve()),
    }
    write_json(out_dir / "train_summary.json", summary)
    logger.info("Training complete. Summary written to %s", out_dir / "train_summary.json")


def main() -> None:
    _load_repo_dotenv()

    parser = argparse.ArgumentParser(description="Train StepwiseSAEBank on CODI latent trajectories.")
    parser.add_argument("--dataset-dir", required=True, help="Directory with train/test safetensors.")
    parser.add_argument("--output-dir", required=True, help="Run output root (logs, checkpoints, weights).")
    parser.add_argument("--n-steps", type=int, required=True, help="Number of latent steps (e.g. 7 for CODI-GPT2 with 6 latent iters).")
    parser.add_argument("--d-sae", type=int, default=3072, help="SAE feature dimension (default: 3072 = 768*4).")
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64).")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps (default: 50).")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, or cpu (default: auto).")
    parser.add_argument("--l1-coeff", type=float, default=0.0, help="L1 sparsity coefficient (default: 0.0).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        d_sae=args.d_sae,
        k=args.k,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_every=args.log_every,
        device_str=args.device,
        l1_coeff=args.l1_coeff,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
