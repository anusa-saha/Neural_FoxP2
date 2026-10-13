"""Stage I: language features that are both selective and causally effective (Section 4.1)."""
from dataclasses import dataclass

import torch

from .config import Hyperparameters
from .evaluate import Evaluator
from .sae import SAE


@dataclass
class LayerFeatures:
    layer: int
    support: torch.Tensor
    selectivity: torch.Tensor
    lift: torch.Tensor
    z_tgt: torch.Tensor
    z_en: torch.Tensor


def selectivity(z_tgt: torch.Tensor, z_en: torch.Tensor, eps: float) -> torch.Tensor:
    """Standardized mean difference over meaning-matched pairs (Eq. 4)."""
    return (z_tgt - z_en).mean(0) / (z_tgt.std(0) + z_en.std(0) + eps)


def activation_q95(z: torch.Tensor) -> torch.Tensor:
    """95th percentile of each feature's positive activations."""
    q = torch.zeros(z.shape[1])
    for j in range(z.shape[1]):
        active = z[:, j][z[:, j] > 0]
        q[j] = active.quantile(0.95) if len(active) else 0.0
    return q


def causal_lift(evaluator: Evaluator, layer: int, directions: torch.Tensor, q95: torch.Tensor,
                prompts: list[str], hyper: Hyperparameters) -> torch.Tensor:
    """Median over alpha of the rise in Delta M per unit alpha when alpha * d_j is added (Eq. 5)."""
    base = evaluator.opening_delta_m(prompts)
    n_prompts, n_features = len(prompts), len(directions)
    rows = [(j, c) for j in range(n_features) for c in hyper.lift_multipliers]
    gains = torch.zeros(n_features, len(hyper.lift_multipliers))
    per_batch = max(1, hyper.batch_size * 4 // n_prompts)
    for start in range(0, len(rows), per_batch):
        chunk = rows[start:start + per_batch]
        alphas = torch.tensor([c * q95[j].item() for j, c in chunk])
        vectors = torch.stack([directions[j] for j, _ in chunk]) * alphas[:, None]
        vectors = vectors.repeat_interleave(n_prompts, 0).to(evaluator.lm.device)
        edit = {layer: lambda h, v=vectors: h + v[:, None, :].to(h.dtype)}
        steered = evaluator.opening_delta_m([p for _ in chunk for p in prompts], edit)
        per_unit = (steered.view(len(chunk), n_prompts) - base).mean(1) / alphas.clamp(min=1e-6)
        for (j, c), value in zip(chunk, per_unit):
            gains[j, hyper.lift_multipliers.index(c)] = value
    return gains.median(1).values


def stage1(evaluator: Evaluator, sae: SAE, h_tgt: torch.Tensor, h_en: torch.Tensor,
           lift_prompts: list[str], hyper: Hyperparameters) -> LayerFeatures:
    z_tgt, z_en = sae.encode_batched(h_tgt), sae.encode_batched(h_en)
    sel = selectivity(z_tgt, z_en, hyper.eps)
    candidates = sel.topk(hyper.lift_candidates).indices
    candidates = candidates[sel[candidates] > 0]
    q95 = activation_q95(torch.cat([z_tgt, z_en])[:, candidates])
    keep = q95 > 0
    candidates, q95 = candidates[keep], q95[keep]

    directions = sae.decoder(candidates).T.cpu()
    lift = causal_lift(evaluator, sae.layer, directions, q95, lift_prompts, hyper)
    rank = sel[candidates].clamp(min=0) * lift.clamp(min=0)
    top = rank.topk(min(hyper.k_features, int((rank > 0).sum()))).indices
    support = candidates[top]
    return LayerFeatures(sae.layer, support, sel[support], lift[top], z_tgt[:, support], z_en[:, support])
