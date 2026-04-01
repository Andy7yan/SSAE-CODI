"""Aggregate per-sample safetensors latent captures into train/test dataset files.

Usage:
    python src/build_dataset.py \\
        --run-dirs /srv/scratch/.../runs/run-A /srv/scratch/.../runs/run-B \\
        --output-dir /srv/scratch/.../datasets/gsm8k-latents-v1 \\
        --train-ratio 0.9

Output layout:
    <output-dir>/
        train.safetensors   -- shape (N_train, n_steps, hidden_size)
        test.safetensors    -- shape (N_test,  n_steps, hidden_size)
        meta.json           -- dataset metadata
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

from io_utils import ensure_dir, load_safetensors, read_jsonl, save_safetensors, write_json


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


def _collect_records(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Read capture_index.jsonl from each run dir and return all records."""
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        index_path = run_dir / "capture_index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"capture_index.jsonl not found in {run_dir}")
        rows = read_jsonl(index_path)
        for row in rows:
            if "tensor_path" not in row:
                raise ValueError(f"Record missing 'tensor_path' in {index_path}: {row}")
            records.append(row)
    return records


def _load_trajectory(record: dict[str, Any]) -> Any:
    """Load the hidden tensor from a single record. Returns a 2-D tensor (n_steps, d_model)."""
    import torch  # type: ignore

    tensor_path = Path(record["tensor_path"])
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")

    tensors = load_safetensors(tensor_path)
    hidden = tensors["hidden"]  # (n_steps, d_model) or (1, d_model)

    if hidden.ndim == 1:
        hidden = hidden.unsqueeze(0)
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(hidden)}")
    return hidden


def build_dataset(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate latent captures from run dirs into train/test safetensors files.

    Args:
        run_dirs: Directories containing capture_index.jsonl and .safetensors files.
        output_dir: Destination directory for the aggregated dataset.
        train_ratio: Fraction of samples assigned to the training split.
        seed: Random seed for split shuffling.

    Returns:
        Meta dict written to <output_dir>/meta.json.
    """
    import torch  # type: ignore

    run_dir_paths = [Path(d) for d in run_dirs]
    out_dir = ensure_dir(output_dir)

    records = _collect_records(run_dir_paths)
    if not records:
        raise ValueError("No records found across provided run dirs.")

    # Load all trajectories.
    trajectories: list[Any] = []
    for record in records:
        traj = _load_trajectory(record)
        trajectories.append(traj)

    # Validate consistent shape.
    shapes = [tuple(t.shape) for t in trajectories]
    if len(set(shapes)) > 1:
        raise ValueError(
            f"Inconsistent trajectory shapes found: {set(shapes)}. "
            "Ensure all run dirs used the same capture_mode and model."
        )

    n_steps, hidden_size = shapes[0]

    # Shuffle and split.
    rng = random.Random(seed)
    indices = list(range(len(trajectories)))
    rng.shuffle(indices)
    n_train = max(1, int(len(indices) * train_ratio))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def _stack(idx_list: list[int]) -> Any:
        return torch.stack([trajectories[i] for i in idx_list], dim=0)

    train_tensor = _stack(train_idx)  # (N_train, n_steps, hidden_size)
    test_tensor = _stack(test_idx) if test_idx else torch.empty(0, n_steps, hidden_size)

    save_safetensors(out_dir / "train.safetensors", {"hidden": train_tensor})
    save_safetensors(out_dir / "test.safetensors", {"hidden": test_tensor})

    # Infer metadata from first record.
    first = records[0]
    meta: dict[str, Any] = {
        "model_name": first.get("model_name", "unknown"),
        "backend": first.get("backend", "unknown"),
        "capture_mode": first.get("capture_mode", "unknown"),
        "target_layer_index": first.get("target_layer_index", -1),
        "n_steps": n_steps,
        "hidden_size": hidden_size,
        "dtype": str(train_tensor.dtype),
        "n_total": len(trajectories),
        "n_train": int(train_tensor.shape[0]),
        "n_test": int(test_tensor.shape[0]),
        "train_ratio": train_ratio,
        "seed": seed,
        "source_run_dirs": [str(p.resolve()) for p in run_dir_paths],
        "output_dir": str(out_dir.resolve()),
    }
    write_json(out_dir / "meta.json", meta)
    print(
        f"Dataset built: {meta['n_train']} train / {meta['n_test']} test  "
        f"| shape per sample: ({n_steps}, {hidden_size})  "
        f"| output: {out_dir}"
    )
    return meta


def main() -> None:
    _load_repo_dotenv()

    parser = argparse.ArgumentParser(description="Aggregate CODI latent captures into SAE training datasets.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more run directories containing capture_index.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for train.safetensors, test.safetensors, and meta.json.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of samples for training split (default: 0.9).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split (default: 42).")
    args = parser.parse_args()

    build_dataset(
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
