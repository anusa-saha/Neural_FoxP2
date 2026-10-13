"""Stage II: a low-rank language subspace per layer and a contiguous layer window (Section 4.2, Appendix D)."""
import math
from dataclasses import dataclass

import torch

from .config import Hyperparameters
from .stage1 import LayerFeatures


@dataclass
class LayerSubspace:
    layer: int
    support: torch.Tensor
    basis: torch.Tensor
    rank: int
    effective_rank: float
    stability: float
    mu: torch.Tensor
    z_bar: torch.Tensor

    @property
    def projector(self) -> torch.Tensor:
        return self.basis @ self.basis.T


def effective_rank(s: torch.Tensor) -> float:
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def eigengap_index(s: torch.Tensor, max_rank: int, eps: float) -> int:
    n = min(max_rank, len(s) - 1)
    ratios = s[:n] / (s[1:n + 1] + eps)
    return int(ratios.argmax()) + 1


def choose_rank(s: torch.Tensor, max_rank: int, eps: float) -> tuple[int, float]:
    r_eff = effective_rank(s)
    return min(math.ceil(r_eff), eigengap_index(s, max_rank, eps)), r_eff


def top_directions(dz: torch.Tensor, r: int) -> torch.Tensor:
    """First r right singular vectors of the shift matrix: [K, r]."""
    return torch.linalg.svd(dz, full_matrices=False).Vh[:r].T


def principal_sines(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cosines = torch.linalg.svdvals(a.T @ b).clamp(0, 1)
    return (1 - cosines ** 2).sqrt()


def stability(dz: torch.Tensor, basis: torch.Tensor, n_bootstrap: int, seed: int) -> float:
    """1 - mean over bootstrap resamples of the mean principal-angle sine to the full-sample subspace."""
    generator = torch.Generator().manual_seed(seed)
    r = basis.shape[1]
    sines = []
    for _ in range(n_bootstrap):
        rows = torch.randint(len(dz), (len(dz),), generator=generator)
        sines.append(principal_sines(top_directions(dz[rows], r), basis).mean())
    return 1 - float(torch.stack(sines).mean())


def stage2(features: LayerFeatures, hyper: Hyperparameters) -> LayerSubspace:
    dz = features.z_tgt - features.z_en
    s = torch.linalg.svdvals(dz)
    r, r_eff = choose_rank(s, hyper.max_rank, hyper.eps)
    basis = top_directions(dz, r)
    return LayerSubspace(
        layer=features.layer,
        support=features.support,
        basis=basis,
        rank=r,
        effective_rank=r_eff,
        stability=stability(dz, basis, hyper.n_bootstrap, hyper.seed),
        mu=dz.mean(0),
        z_bar=features.z_en.mean(0),
    )


def candidate_windows(subspaces: dict[int, LayerSubspace], n_layers: int,
                      hyper: Hyperparameters) -> list[tuple[int, int]]:
    """Contiguous ranges of stable layers whose width lies between 15% and 35% of model depth."""
    eligible = {layer for layer, s in subspaces.items() if s.stability >= hyper.tau_stab}
    w_min, w_max = math.ceil(hyper.window_min * n_layers), math.floor(hyper.window_max * n_layers)
    windows = []
    for start in sorted(eligible):
        for width in range(w_min, w_max + 1):
            layers = range(start, start + width)
            if all(layer in eligible for layer in layers):
                windows.append((start, start + width - 1))
    return windows
