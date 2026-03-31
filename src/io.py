import json
import pickle
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {jsonl_path}, "
                    f"but received {type(payload).__name__}."
                )
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> Path:
    jsonl_path = Path(path)
    ensure_dir(jsonl_path.parent)
    mode = "a" if append else "w"
    with jsonl_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return jsonl_path


def append_jsonl(path: str | Path, row: dict[str, Any]) -> Path:
    return write_jsonl(path, [row], append=True)


def write_json(path: str | Path, payload: Any) -> Path:
    json_path = Path(path)
    ensure_dir(json_path.parent)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return json_path


def write_text(path: str | Path, text: str) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def save_pt(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)

    try:
        import torch  # type: ignore
    except ImportError:
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return output_path

    torch.save(payload, output_path)
    return output_path


def load_pt(path: str | Path) -> Any:
    input_path = Path(path)
    try:
        import torch  # type: ignore
    except ImportError:
        with input_path.open("rb") as handle:
            return pickle.load(handle)

    return torch.load(input_path, map_location="cpu")
