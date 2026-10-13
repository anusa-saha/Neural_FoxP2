"""Stage III: the signed edit inside the window (Section 4.3, Eqs. 6-8)."""
from dataclasses import dataclass

import torch

from .evaluate import Method
from .model import Edit
from .prompts import PLAIN, PromptStyle
from .sae import SAEBank
from .stage2 import LayerSubspace


@dataclass
class LayerEdit:
    layer: int
    support: torch.Tensor
    z_bar: torch.Tensor
    promote: torch.Tensor
    suppress: torch.Tensor
    subspace: torch.Tensor

    @classmethod
    def build(cls, s: LayerSubspace, decoder: torch.Tensor) -> "LayerEdit":
        """decoder: the SAE decoder columns of the support, [d_model, K]."""
        decoder = decoder.float().cpu()
        return cls(
            layer=s.layer,
            support=s.support,
            z_bar=s.z_bar,
            promote=decoder @ (s.projector @ s.mu),
            suppress=decoder @ s.z_bar,
            subspace=torch.linalg.qr(decoder @ s.basis).Q,
        )


class NeuralFoxp2:
    """h <- h + gamma * W_dec Pi_N (lambda P mu - beta a(x) z_bar), with a(x) the alignment with z_bar."""

    def __init__(self, layers: dict[int, LayerEdit], saes: SAEBank, gamma: float,
                 lam: float = 1.0, beta: float = 1.0, eps: float = 1e-6):
        self.layers = layers
        self.saes = saes
        self.gamma, self.lam, self.beta, self.eps = gamma, lam, beta, eps

    @property
    def window(self) -> tuple[int, int]:
        return min(self.layers), max(self.layers)

    def _edit(self, e: LayerEdit):
        sae = self.saes[e.layer]
        device = sae.sae.W_enc.device
        support, z_bar = e.support.to(device), e.z_bar.to(device)
        promote, suppress = e.promote.to(device), e.suppress.to(device)
        norm = z_bar @ z_bar + self.eps

        def edit(h: torch.Tensor) -> torch.Tensor:
            delta = self.lam * promote
            if self.beta:
                a = (sae.encode(h)[..., support] @ z_bar) / norm
                delta = delta - self.beta * a[..., None] * suppress
            return h + (self.gamma * delta).to(h.device, h.dtype)

        return edit

    def edits(self) -> dict[int, Edit]:
        if self.gamma == 0:
            return {}
        return {layer: self._edit(e) for layer, e in self.layers.items()}

    def method(self, name: str = "neural_foxp2", style: PromptStyle = PLAIN) -> Method:
        return Method(name, style=style, edits=self.edits())

    def with_settings(self, **changes) -> "NeuralFoxp2":
        settings = {"gamma": self.gamma, "lam": self.lam, "beta": self.beta, "eps": self.eps, **changes}
        return NeuralFoxp2(self.layers, self.saes, **settings)

    def save(self, path: str, **metadata):
        torch.save({"layers": self.layers, "gamma": self.gamma, "lam": self.lam, "beta": self.beta,
                    "eps": self.eps, **metadata}, path)

    @classmethod
    def load(cls, path: str, saes: SAEBank) -> "NeuralFoxp2":
        data = torch.load(path, weights_only=False)
        return cls(data["layers"], saes, data["gamma"], data["lam"], data["beta"], data["eps"])


def subspace_ablation(layers: dict[int, LayerEdit], name: str = "subspace_ablation",
                      style: PromptStyle = PLAIN) -> Method:
    """Necessity test (Appendix I.3): project the steering subspace out of the unedited model."""
    def make(q: torch.Tensor):
        def edit(h):
            basis = q.to(h.device, torch.float32)
            return h - ((h.float() @ basis) @ basis.T).to(h.dtype)
        return edit

    return Method(name, style=style, edits={layer: make(e.subspace) for layer, e in layers.items()})
