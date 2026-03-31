from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


def _parse_yaml_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""

    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]

    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if ":" not in raw_line:
            raise ValueError(f"Unsupported YAML syntax on line {line_number}: {raw_line!r}")

        key, value = raw_line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing YAML key on line {line_number}.")

        data[key] = _parse_yaml_scalar(value)

    return data


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return _minimal_yaml_load(text)

    payload = yaml.safe_load(text)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level mapping in {path}, but received {type(payload).__name__}.")
    return dict(payload)


@dataclass(slots=True)
class Stage1Config:
    model_name_or_path: str
    hf_token_env_var: str
    device: str
    dtype: str
    num_latent: int
    inf_latent_iterations: int
    output_dir: str
    capture_hidden: bool
    capture_mode: str
    target_layer_index: int
    max_samples: int
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
        return Path(__file__).resolve().parents[2]

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

    def with_overrides(self, **overrides: Any) -> "Stage1Config":
        return replace(self, **overrides)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage1Config":
        yaml_path = Path(path)
        if not yaml_path.is_absolute():
            yaml_path = Path.cwd() / yaml_path
        yaml_path = yaml_path.resolve()

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {yaml_path}")

        data = _load_yaml_dict(yaml_path)
        valid_fields = set(cls.__dataclass_fields__.keys())
        unknown_fields = sorted(set(data.keys()) - valid_fields)
        if unknown_fields:
            raise ValueError(f"Unknown config fields in {yaml_path}: {unknown_fields}")

        return cls(**data)
