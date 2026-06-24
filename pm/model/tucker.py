"""
Tucker compression layer.

T ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
  G  — core tensor [R1, R2, R3, R4]: importance of each Tucker component
  Uᵢ — factor matrices [dim_i, Rᵢ]: subspace directions per mode

G and Uᵢ are nn.Parameters warm-started by SVD via tensorly, then
trained end-to-end through backprop.
"""
from __future__ import annotations

import tensorly as tl
import torch
import torch.nn as nn
from tensorly.decomposition import tucker

tl.set_backend("pytorch")


class TuckerCompressor(nn.Module):
    """[batch, C, N, M, P] → [batch, R1, R2, R3, R4]."""

    def __init__(self, input_shape: tuple[int, int, int, int], ranks: list[int]) -> None:
        super().__init__()
        assert len(input_shape) == 4 and len(ranks) == 4

        sample = torch.randn(*input_shape)
        core_init, factors_init = tucker(sample, rank=ranks, init="svd", n_iter_max=10)

        self.core    = nn.Parameter(core_init.detach().clone())
        self.factors = nn.ParameterList([
            nn.Parameter(f.detach().clone()) for f in factors_init
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project into Tucker subspaces, then scale by learnable core
        projected = torch.einsum(
            "bcnmp,cR,nS,mT,pU->bRSTU",
            x,
            self.factors[0],  # [C, R1]
            self.factors[1],  # [N, R2]
            self.factors[2],  # [M, R3]
            self.factors[3],  # [P, R4]
        )
        return projected * self.core
