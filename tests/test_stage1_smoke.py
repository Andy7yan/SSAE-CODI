from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage1.config import Stage1Config
from stage1.load_model import LoadedModelBundle, load_model_bundle
from stage1.run_inference import run_stage1


pytest.importorskip("torch")
import torch  # type: ignore


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = "[PAD]"
        self.pad_token_id = 20
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.special_tokens_map = {"pad_token": self.pad_token, "eos_token": self.eos_token}
        self.additional_special_tokens: list[str] = []

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        values = [11, 12, 13]
        input_ids = torch.tensor([values], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        filtered = []
        for token_id in token_ids:
            token_value = int(token_id)
            if skip_special_tokens and token_value in {self.pad_token_id, self.eos_token_id}:
                continue
            filtered.append(token_value)
        if filtered == [7]:
            return "7"
        return " ".join(str(token_id) for token_id in filtered)


class FakeCoreModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, vocab_size=23)
        self.embedding = torch.nn.Embedding(23, 4)
        self.lm_head = torch.nn.Linear(4, 23, bias=False)
        self.decode_step = 0

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def tie_weights(self) -> None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
        past_key_values: object | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> SimpleNamespace:
        if input_ids is not None:
            hidden = self.embedding(input_ids)
        elif inputs_embeds is not None:
            hidden = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        hidden_states = (hidden, hidden + 0.25)
        logits = torch.zeros(hidden.shape[0], hidden.shape[1], self.config.vocab_size, dtype=torch.float32)

        if inputs_embeds is not None and output_hidden_states is False:
            next_token_id = 7 if self.decode_step == 0 else 2
            logits[:, -1, next_token_id] = 100.0
            self.decode_step += 1
        else:
            logits[:, -1, 7] = 10.0

        return SimpleNamespace(
            hidden_states=hidden_states,
            logits=logits,
            past_key_values=[("fake", self.decode_step)],
        )


class FakeOfficialCodiModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codi = FakeCoreModel()
        self.model_name = "fake-codi"
        self.pad_token_id = 20
        self.bot_id = 21
        self.eot_id = 22
        self.num_latent = 2
        self.use_prj = False

    def eval(self) -> "FakeOfficialCodiModel":
        self.codi.decode_step = 0
        super().eval()
        return self

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.codi.get_input_embeddings()


def _build_fake_bundle(config: Stage1Config) -> LoadedModelBundle:
    model = FakeOfficialCodiModel().eval()
    tokenizer = FakeTokenizer()
    model_info = {
        "model_name": "fake-codi",
        "backend": "official-codi-gpt2-wrapper",
        "base_model_name": "fake-gpt2",
        "checkpoint_source": "unit-test",
        "device": "cpu",
        "dtype": config.dtype,
        "hidden_size": 4,
        "vocab_size": 23,
        "number_of_latent_steps_configured": config.inf_latent_iterations,
        "num_latent_training_config": config.num_latent,
        "capture_mode": config.capture_mode,
        "special_token_info": {
            "pad_token_id": 20,
            "bot_id": 21,
            "eot_id": 22,
        },
        "uses_projection_layer": False,
    }
    return LoadedModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name="fake-codi",
        backend="official-codi-gpt2-wrapper",
        base_model_name="fake-gpt2",
        model_info=model_info,
    )


def _build_test_config(tmp_path: Path, capture_hidden: bool, capture_mode: str) -> Stage1Config:
    data_file = tmp_path / "gsm8k_debug.jsonl"
    data_file.write_text(
        '{"sample_id":"sample_001","question":"What is 3 plus 4?","answer":"7"}\n',
        encoding="utf-8",
    )
    return Stage1Config(
        model_name_or_path="zen-E/CODI-gpt2",
        hf_token_env_var="HF_TOKEN",
        device="cpu",
        dtype="float32",
        num_latent=2,
        inf_latent_iterations=2,
        output_dir=(tmp_path / "outputs").as_posix(),
        capture_hidden=capture_hidden,
        capture_mode=capture_mode,
        target_layer_index=-1,
        max_samples=1,
        data_path=data_file.as_posix(),
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        trust_remote_code=True,
        seed=0,
    )


def test_config_can_load() -> None:
    config = Stage1Config()
    assert config.model_name_or_path == "zen-E/CODI-gpt2"
    assert config.capture_mode == "seed-only"
    assert config.inf_latent_iterations == 6


def test_model_can_load(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Stage1Config()
    fake_bundle = _build_fake_bundle(config)
    monkeypatch.setattr("stage1.load_model._load_direct_hf_bundle", lambda config, token, logger=None: fake_bundle)
    bundle = load_model_bundle(config=config)
    assert bundle.backend == "official-codi-gpt2-wrapper"
    assert bundle.model_info["hidden_size"] == 4


def test_one_sample_can_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _build_test_config(tmp_path=tmp_path, capture_hidden=False, capture_mode="seed-only")
    fake_bundle = _build_fake_bundle(config)
    monkeypatch.setattr("stage1.run_inference.load_model_bundle", lambda config, logger=None: fake_bundle)

    summary = run_stage1(config=config, run_name="pytest_smoke")
    results_path = Path(summary["inference_dir"]) / "results.jsonl"

    assert results_path.exists()
    results_text = results_path.read_text(encoding="utf-8")
    assert "sample_001" in results_text
    assert '"generated_text": "7"' in results_text


def test_one_hidden_output_file_can_be_created_in_seed_only_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _build_test_config(tmp_path=tmp_path, capture_hidden=True, capture_mode="seed-only")
    fake_bundle = _build_fake_bundle(config)
    monkeypatch.setattr("stage1.run_inference.load_model_bundle", lambda config, logger=None: fake_bundle)

    summary = run_stage1(config=config, run_name="pytest_capture")
    hidden_dir = Path(summary["hidden_dir"])
    hidden_files = list(hidden_dir.glob("*.pt"))

    assert len(hidden_files) == 1
    assert hidden_files[0].name == "sample_001__seed-only.pt"
