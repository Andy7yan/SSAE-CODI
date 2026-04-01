"""Smoke tests for StepwiseSAE and StepwiseSAEBank.

Run locally (no GPU required):
    python -m pytest tests/test_sae.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sae import StepwiseSAE, StepwiseSAEBank, sae_loss


# ---------------------------------------------------------------------------
# StepwiseSAE
# ---------------------------------------------------------------------------

class TestStepwiseSAE:
    D_MODEL = 32
    D_SAE = 128
    K = 8
    BATCH = 4

    def _make_sae(self) -> StepwiseSAE:
        return StepwiseSAE(self.D_MODEL, self.D_SAE, self.K)

    def test_forward_output_shapes(self) -> None:
        sae = self._make_sae()
        h = torch.randn(self.BATCH, self.D_MODEL)
        h_hat, z = sae(h)
        assert h_hat.shape == (self.BATCH, self.D_MODEL), f"h_hat shape mismatch: {h_hat.shape}"
        assert z.shape == (self.BATCH, self.D_SAE), f"z shape mismatch: {z.shape}"

    def test_topk_sparsity(self) -> None:
        sae = self._make_sae()
        h = torch.randn(self.BATCH, self.D_MODEL)
        _, z = sae(h)
        # Each row should have exactly K non-zero entries.
        nonzero_counts = (z != 0).sum(dim=-1)
        assert (nonzero_counts == self.K).all(), (
            f"Expected exactly k={self.K} non-zeros per row, got: {nonzero_counts.tolist()}"
        )

    def test_encode_decode_consistency(self) -> None:
        sae = self._make_sae()
        h = torch.randn(self.BATCH, self.D_MODEL)
        z = sae.encode(h)
        h_hat = sae.decode(z)
        h_hat2, z2 = sae(h)
        assert torch.allclose(h_hat, h_hat2), "encode+decode vs forward inconsistency"
        assert torch.allclose(z, z2), "encode vs forward z inconsistency"

    def test_decoder_column_norms_after_normalize(self) -> None:
        sae = self._make_sae()
        sae.normalize_decoder()
        col_norms = sae.decoder.weight.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), (
            f"Decoder column norms not unit after normalization: {col_norms[:5]}"
        )

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            StepwiseSAE(self.D_MODEL, self.D_SAE, k=0)
        with pytest.raises(ValueError):
            StepwiseSAE(self.D_MODEL, self.D_SAE, k=self.D_SAE + 1)

    def test_3d_input(self) -> None:
        """SAE should handle (..., d_model) inputs beyond 2-D."""
        sae = self._make_sae()
        h = torch.randn(2, 3, self.D_MODEL)
        h_hat, z = sae(h)
        assert h_hat.shape == (2, 3, self.D_MODEL)
        assert z.shape == (2, 3, self.D_SAE)


# ---------------------------------------------------------------------------
# StepwiseSAEBank
# ---------------------------------------------------------------------------

class TestStepwiseSAEBank:
    N_STEPS = 7
    D_MODEL = 32
    D_SAE = 128
    K = 8
    BATCH = 4

    def _make_bank(self) -> StepwiseSAEBank:
        return StepwiseSAEBank(self.N_STEPS, self.D_MODEL, self.D_SAE, self.K)

    def test_forward_output_shapes(self) -> None:
        bank = self._make_bank()
        h_seq = torch.randn(self.BATCH, self.N_STEPS, self.D_MODEL)
        h_hat_seq, z_seq = bank(h_seq)
        assert h_hat_seq.shape == (self.BATCH, self.N_STEPS, self.D_MODEL)
        assert z_seq.shape == (self.BATCH, self.N_STEPS, self.D_SAE)

    def test_per_step_sparsity(self) -> None:
        bank = self._make_bank()
        h_seq = torch.randn(self.BATCH, self.N_STEPS, self.D_MODEL)
        _, z_seq = bank(h_seq)
        # Every (batch, step) slice should have exactly K non-zeros.
        nonzero_counts = (z_seq != 0).sum(dim=-1)  # (batch, n_steps)
        assert (nonzero_counts == self.K).all(), (
            f"Expected k={self.K} per (batch, step), got: {nonzero_counts}"
        )

    def test_wrong_n_steps_raises(self) -> None:
        bank = self._make_bank()
        h_wrong = torch.randn(self.BATCH, self.N_STEPS + 1, self.D_MODEL)
        with pytest.raises(ValueError):
            bank(h_wrong)

    def test_num_independent_saes(self) -> None:
        bank = self._make_bank()
        assert len(bank.saes) == self.N_STEPS


# ---------------------------------------------------------------------------
# sae_loss
# ---------------------------------------------------------------------------

class TestSaeLoss:
    D_MODEL = 32
    D_SAE = 128
    BATCH = 4

    def test_loss_keys(self) -> None:
        h = torch.randn(self.BATCH, self.D_MODEL)
        h_hat = torch.randn(self.BATCH, self.D_MODEL)
        z = torch.randn(self.BATCH, self.D_SAE)
        losses = sae_loss(h, h_hat, z)
        assert set(losses.keys()) == {"reconstruction_loss", "sparsity_loss", "total_loss"}

    def test_zero_l1_sparsity_loss_is_zero(self) -> None:
        h = torch.randn(self.BATCH, self.D_MODEL)
        h_hat = torch.randn(self.BATCH, self.D_MODEL)
        z = torch.randn(self.BATCH, self.D_SAE)
        losses = sae_loss(h, h_hat, z, l1_coeff=0.0)
        assert losses["sparsity_loss"].item() == 0.0

    def test_reconstruction_loss_is_mse(self) -> None:
        h = torch.zeros(self.BATCH, self.D_MODEL)
        h_hat = torch.ones(self.BATCH, self.D_MODEL)
        z = torch.zeros(self.BATCH, self.D_SAE)
        losses = sae_loss(h, h_hat, z)
        expected = 1.0  # MSE of (0-1)^2 = 1.0
        assert abs(losses["reconstruction_loss"].item() - expected) < 1e-5

    def test_total_equals_recon_plus_sparsity(self) -> None:
        h = torch.randn(self.BATCH, self.D_MODEL)
        h_hat = torch.randn(self.BATCH, self.D_MODEL)
        z = torch.randn(self.BATCH, self.D_SAE)
        losses = sae_loss(h, h_hat, z, l1_coeff=0.01)
        expected_total = losses["reconstruction_loss"] + losses["sparsity_loss"]
        assert torch.allclose(losses["total_loss"], expected_total)
