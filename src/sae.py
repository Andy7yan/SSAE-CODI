"""TopK Sparse Autoencoder for CODI latent step representations.

Reference: SSAE paper (arXiv:2603.03031).

Classes:
    StepwiseSAE       -- SAE for a single latent step position.
    StepwiseSAEBank   -- Collection of n_steps independent SAEs.

Functions:
    sae_loss          -- Reconstruction + optional sparsity loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class StepwiseSAE(nn.Module):
    """TopK Sparse Autoencoder for one latent step position.

    Args:
        d_model: Dimension of the input hidden state (e.g. 768 for GPT-2).
        d_sae: Dimension of the sparse feature space (d_sae >> d_model).
        k: Number of active features per forward pass (TopK sparsity).
    """

    def __init__(self, d_model: int, d_sae: int, k: int) -> None:
        super().__init__()
        if k <= 0 or k > d_sae:
            raise ValueError(f"k must satisfy 0 < k <= d_sae, got k={k}, d_sae={d_sae}.")

        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        # Encoder: projects to sparse feature space.
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        # Decoder: reconstructs from sparse features; columns are kept unit-norm.
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        # Tied init: decoder weight = encoder weight transposed.
        nn.init.kaiming_uniform_(self.encoder.weight)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self._normalize_decoder_columns()

    @torch.no_grad()
    def _normalize_decoder_columns(self) -> None:
        """Normalize decoder weight columns to unit norm (prevents feature shrinkage)."""
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.data.div_(norms)

    def encode(self, h: Tensor) -> Tensor:
        """Encode hidden states to sparse feature activations.

        Args:
            h: Input tensor of shape (..., d_model).

        Returns:
            Sparse activation tensor of shape (..., d_sae).
            Exactly k values per token are non-zero.
        """
        pre_act = self.encoder(h)  # (..., d_sae)
        return self._topk_activation(pre_act)

    def _topk_activation(self, x: Tensor) -> Tensor:
        # x: (..., d_sae)
        topk_vals, topk_idx = x.topk(self.k, dim=-1)
        # ReLU on selected values to ensure non-negative activations.
        topk_vals = torch.relu(topk_vals)
        sparse = torch.zeros_like(x)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    def decode(self, z: Tensor) -> Tensor:
        """Decode sparse features back to hidden state space.

        Args:
            z: Sparse feature tensor of shape (..., d_sae).

        Returns:
            Reconstructed tensor of shape (..., d_model).
        """
        return self.decoder(z)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass: encode then decode.

        Args:
            h: Input tensor of shape (..., d_model).

        Returns:
            Tuple of (h_hat, z):
                h_hat -- reconstruction, shape (..., d_model).
                z     -- sparse features, shape (..., d_sae).
        """
        z = self.encode(h)
        h_hat = self.decode(z)
        return h_hat, z

    def normalize_decoder(self) -> None:
        """Call after each optimizer step to maintain unit-norm decoder columns."""
        with torch.no_grad():
            self._normalize_decoder_columns()


class StepwiseSAEBank(nn.Module):
    """A bank of n_steps independent StepwiseSAEs, one per CODI latent step.

    Args:
        n_steps: Number of latent steps (including seed step). For CODI-GPT2
            with inf_latent_iterations=6 this is 7 (seed + 6 latent).
        d_model: Hidden state dimension.
        d_sae: Sparse feature dimension.
        k: TopK sparsity per SAE.
    """

    def __init__(self, n_steps: int, d_model: int, d_sae: int, k: int) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.saes = nn.ModuleList([StepwiseSAE(d_model, d_sae, k) for _ in range(n_steps)])

    def forward(self, h_seq: Tensor) -> tuple[Tensor, Tensor]:
        """Run each step's hidden state through its dedicated SAE.

        Args:
            h_seq: Latent trajectory tensor of shape (batch, n_steps, d_model).

        Returns:
            Tuple of (h_hat_seq, z_seq):
                h_hat_seq -- reconstructions, shape (batch, n_steps, d_model).
                z_seq     -- sparse features,  shape (batch, n_steps, d_sae).
        """
        if h_seq.shape[1] != self.n_steps:
            raise ValueError(
                f"Expected n_steps={self.n_steps} in dim 1, got shape {tuple(h_seq.shape)}."
            )

        h_hats, zs = [], []
        for step_idx, sae in enumerate(self.saes):
            h_hat_i, z_i = sae(h_seq[:, step_idx, :])  # (batch, d_model), (batch, d_sae)
            h_hats.append(h_hat_i)
            zs.append(z_i)

        h_hat_seq = torch.stack(h_hats, dim=1)  # (batch, n_steps, d_model)
        z_seq = torch.stack(zs, dim=1)          # (batch, n_steps, d_sae)
        return h_hat_seq, z_seq

    def normalize_decoders(self) -> None:
        """Normalize all decoder columns. Call after each optimizer step."""
        for sae in self.saes:
            sae.normalize_decoder()


def sae_loss(
    h: Tensor,
    h_hat: Tensor,
    z: Tensor,
    l1_coeff: float = 0.0,
) -> dict[str, Tensor]:
    """Compute SAE training losses.

    TopK already enforces hard sparsity, so L1 is disabled by default.
    The l1_coeff parameter is provided for ablation experiments.

    Args:
        h: Original hidden states, shape (..., d_model).
        h_hat: Reconstructed hidden states, shape (..., d_model).
        z: Sparse feature activations, shape (..., d_sae).
        l1_coeff: Coefficient for L1 sparsity penalty (default 0.0).

    Returns:
        Dict with keys: 'reconstruction_loss', 'sparsity_loss', 'total_loss'.
        All values are scalar tensors.
    """
    reconstruction_loss = torch.mean((h - h_hat) ** 2)
    sparsity_loss = l1_coeff * z.abs().mean() if l1_coeff > 0.0 else torch.zeros((), device=h.device)
    total_loss = reconstruction_loss + sparsity_loss
    return {
        "reconstruction_loss": reconstruction_loss,
        "sparsity_loss": sparsity_loss,
        "total_loss": total_loss,
    }
