import torch

from .config import ModelSpec


class SAE:
    """A frozen public residual-stream SAE (Llama Scope, Gemma Scope or Qwen-Scope)."""

    def __init__(self, spec: ModelSpec, layer: int, device: str = "cuda"):
        from sae_lens import SAE as LensSAE

        loaded = LensSAE.from_pretrained(spec.sae_release, spec.sae_id.format(layer=layer), device=device)
        self.sae = (loaded[0] if isinstance(loaded, tuple) else loaded).eval()
        self.layer = layer
        self.width = self.sae.W_dec.shape[0]

    @torch.no_grad()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(h.to(self.sae.W_enc.device)).float()

    @torch.no_grad()
    def encode_batched(self, h: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        return torch.cat([self.encode(h[i:i + batch_size]).cpu() for i in range(0, len(h), batch_size)])

    def decoder(self, features: torch.Tensor) -> torch.Tensor:
        """Decoder columns d_j for the given features: [d_model, K]."""
        return self.sae.W_dec[features.to(self.sae.W_dec.device)].T.float()


class SAEBank:
    """Loads each layer's SAE once and keeps it on the device until released."""

    def __init__(self, spec: ModelSpec, device: str = "cuda"):
        self.spec = spec
        self.device = device
        self.loaded: dict[int, SAE] = {}

    def __getitem__(self, layer: int) -> SAE:
        if layer not in self.loaded:
            self.loaded[layer] = SAE(self.spec, layer, self.device)
        return self.loaded[layer]

    def release(self, keep: set[int] = frozenset()):
        for layer in list(self.loaded):
            if layer not in keep:
                del self.loaded[layer]
        torch.cuda.empty_cache()
