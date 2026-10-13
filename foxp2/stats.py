"""Bootstrap intervals, paired sign-flip tests and Holm correction (Appendix M)."""
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 17, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def sign_flip_test(a, b, n_perm: int = 10000, seed: int = 17) -> tuple[float, float]:
    """Paired permutation test of mean(a - b) = 0. Returns (standardized statistic, two-sided p)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    observed = diff.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = (signs * diff).mean(1)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    statistic = observed / (null.std() + 1e-12)
    return float(statistic), float(p)


def wilcoxon_test(a, b) -> float:
    return float(wilcoxon(a, b).pvalue)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values within one family."""
    order = sorted(p_values, key=p_values.get)
    m, running, adjusted = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        adjusted[name] = running
    return adjusted
