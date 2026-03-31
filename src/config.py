import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def _default_scratch_root() -> Path:
    configured_root = os.getenv("SCRATCH_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()

    username = os.getenv("USER") or os.getenv("USERNAME") or "unknown-user"
    return Path("/srv/scratch") / username


def _default_output_dir() -> str:
    configured_root = os.getenv("SSAE_CODI_OUTPUT_ROOT")
    if configured_root:
        return configured_root
    return (_default_scratch_root() / "ssae-codi" / "runs").as_posix()


@dataclass(slots=True)
class RunConfig:
    model_name_or_path: str = "zen-E/CODI-gpt2"
    hf_token_env_var: str = "HF_TOKEN"
    device: str = "auto"
    dtype: str = "float32"
    num_latent: int = 6
    inf_latent_iterations: int = 6
    output_dir: str = field(default_factory=_default_output_dir)
    capture_hidden: bool = False
    capture_mode: str = "seed-only"
    target_layer_index: int = -1
    max_samples: int = 2
    data_path: str = "data/gsm8k_debug.jsonl"
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must not be empty.")
        if not self.hf_token_env_var:
            raise ValueError("hf_token_env_var must not be empty.")
        if self.num_latent < 0:
            raise ValueError("num_latent must be non-negative.")
        if self.inf_latent_iterations < 0:
            raise ValueError("inf_latent_iterations must be non-negative.")
        if self.max_samples <= 0:
            raise ValueError("max_samples must be greater than zero.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        if self.capture_mode not in {"seed-only", "per-latent-step"}:
            raise ValueError("capture_mode must be 'seed-only' or 'per-latent-step'.")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be within (0, 1].")

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def scratch_root(self) -> Path:
        return _default_scratch_root()

    @property
    def output_root(self) -> Path:
        output_path = Path(self.output_dir)
        return output_path if output_path.is_absolute() else self.repo_root / output_path

    @property
    def data_file(self) -> Path:
        data_path = Path(self.data_path)
        return data_path if data_path.is_absolute() else self.repo_root / data_path

    def to_dict(self) -> dict[str, Any]:
        field_names = tuple(self.__dataclass_fields__.keys())
        return {field_name: getattr(self, field_name) for field_name in field_names}

    def with_overrides(self, **overrides: Any) -> "RunConfig":
        return replace(self, **overrides)
