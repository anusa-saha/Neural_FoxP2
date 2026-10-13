import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .tokens import TokenSets


def class_masses(logits: torch.Tensor, sets: TokenSets) -> tuple[torch.Tensor, torch.Tensor]:
    """M_tgt and M_en (Eq. 1) for every language column: logits [B, T, V] -> two [B, T, L]."""
    probs = logits.float().softmax(-1)
    return probs @ sets.target.to(probs), probs @ sets.english.to(probs)


def horizon_index(horizon) -> list[int]:
    return [t - 1 for t in horizon]


def defaultness(logits: torch.Tensor, sets: TokenSets, horizon=(1, 2, 3)) -> torch.Tensor:
    """Delta M (Eq. 2) for every language column: [B, L], in [-1, 1]."""
    target, english = class_masses(logits, sets)
    return (target - english)[:, horizon_index(horizon)].mean(1)


def target_share(logits: torch.Tensor, sets: TokenSets, horizon=(1, 2, 3), eps: float = 1e-6) -> torch.Tensor:
    target, english = class_masses(logits, sets)
    return (target / (target + english + eps))[:, horizon_index(horizon)].mean(1)


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per position: [B, T, V] x 2 -> [B, T]."""
    log_p = p_logits.float().log_softmax(-1)
    log_q = q_logits.float().log_softmax(-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def auc(scores, labels) -> float:
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores)))


def select_horizon(scores_by_k: dict[int, np.ndarray], labels, tolerance: float = 0.01) -> int:
    """Rule D2: the smallest horizon whose AUC is within `tolerance` of the best."""
    aucs = {k: auc(s, labels) for k, s in scores_by_k.items()}
    best = max(aucs.values())
    return min(k for k, value in aucs.items() if value >= best - tolerance)
