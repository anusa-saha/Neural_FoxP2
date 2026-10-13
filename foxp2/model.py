import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor

Edit = Callable[[torch.Tensor], torch.Tensor]

BIAS_STEPS = 3


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _with_hidden(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


class OpeningLogits(LogitsProcessor):
    """Adds an optional bias over the first BIAS_STEPS steps and records the first n_steps logits."""

    def __init__(self, prompt_len: int, n_steps: int, bias: torch.Tensor | None = None):
        self.prompt_len = prompt_len
        self.n_steps = n_steps
        self.bias = bias
        self.steps = []

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_len
        if self.bias is not None and step < BIAS_STEPS:
            scores = scores + self.bias.to(scores)
        if step < self.n_steps:
            self.steps.append(scores.float())
        return scores

    def stacked(self) -> torch.Tensor:
        steps = self.steps + [self.steps[-1]] * (self.n_steps - len(self.steps))
        return torch.stack(steps, dim=1)


@dataclass
class Generation:
    tokens: list[list[int]]
    texts: list[str]
    opening_logits: torch.Tensor


class LM:
    def __init__(self, name: str, device: str = "cuda", dtype=torch.bfloat16):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.vocab_size = self.model.get_output_embeddings().weight.shape[0]
        self.device = self.model.device
        thinking = "enable_thinking" in (self.tokenizer.chat_template or "")
        self.template_kwargs = {"enable_thinking": False} if thinking else {}
        self.supports_system = self._supports_system()
        base = self.model.generation_config
        self.generation_config = GenerationConfig(
            do_sample=False, bos_token_id=base.bos_token_id, eos_token_id=base.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def _supports_system(self) -> bool:
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tokenize=False
            )
            return True
        except Exception:
            return False

    def chat(self, user: str, system: str | None = None, history: list[tuple[str, str]] = ()) -> str:
        turns = [*history, (user, None)]
        if system and not self.supports_system:
            first_user, first_reply = turns[0]
            turns[0] = (f"{system}\n\n{first_user}", first_reply)
            system = None
        messages = [{"role": "system", "content": system}] if system else []
        for question, answer in turns:
            messages.append({"role": "user", "content": question})
            if answer is not None:
                messages.append({"role": "assistant", "content": answer})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )

    def encode(self, texts: list[str], suffixes: list[list[int]] | None = None):
        ids = [self.tokenizer(t, add_special_tokens=False).input_ids for t in texts]
        if suffixes is not None:
            ids = [a + list(b) for a, b in zip(ids, suffixes)]
        width = max(map(len, ids))
        pad = self.tokenizer.pad_token_id
        input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in ids], device=self.device)
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device=self.device)
        return input_ids, mask

    def forward(self, input_ids, mask) -> torch.Tensor:
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        return self.model(input_ids=input_ids, attention_mask=mask, position_ids=positions).logits

    @contextmanager
    def steer(self, edits: dict[int, Edit], neuron_edits: dict[int, Edit] | None = None):
        """Edit the residual stream after each layer, and optionally the FFN neurons feeding down_proj."""
        def residual_hook(edit):
            return lambda module, inputs, output: _with_hidden(output, edit(_hidden(output)))

        def neuron_hook(edit):
            return lambda module, inputs: (edit(inputs[0]), *inputs[1:])

        handles = [self.layers[layer].register_forward_hook(residual_hook(edit)) for layer, edit in edits.items()]
        handles += [self.layers[layer].mlp.down_proj.register_forward_pre_hook(neuron_hook(edit))
                    for layer, edit in (neuron_edits or {}).items()]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def adapter(self, path: str | None):
        if path is None:
            yield
            return
        from peft import PeftModel

        base = self.model
        self.model = PeftModel.from_pretrained(base, path)
        try:
            yield
        finally:
            self.model = self.model.unload()

    @torch.no_grad()
    def residuals(self, texts: list[str], layers: list[int], batch_size: int = 16) -> dict[int, torch.Tensor]:
        """Residual stream after each layer at the final prompt token, as float32 on CPU."""
        captured = {layer: [] for layer in layers}

        def make_hook(layer):
            return lambda module, inputs, output: captured[layer].append(_hidden(output)[:, -1].float().cpu())

        handles = [self.layers[layer].register_forward_hook(make_hook(layer)) for layer in layers]
        try:
            for start in range(0, len(texts), batch_size):
                self.forward(*self.encode(texts[start:start + batch_size]))
        finally:
            for handle in handles:
                handle.remove()
        return {layer: torch.cat(chunks) for layer, chunks in captured.items()}

    @torch.no_grad()
    def generate(self, texts: list[str], max_new_tokens: int, opening_steps: int = 3,
                 logit_bias: torch.Tensor | None = None) -> Generation:
        input_ids, mask = self.encode(texts)
        recorder = OpeningLogits(input_ids.shape[1], opening_steps, logit_bias)
        config = copy.copy(self.generation_config)
        config.max_new_tokens = max_new_tokens
        output = self.model.generate(input_ids=input_ids, attention_mask=mask, generation_config=config,
                                     logits_processor=[recorder])
        new = output[:, input_ids.shape[1]:].tolist()
        tokens = [self._strip(seq) for seq in new]
        texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return Generation(tokens, texts, recorder.stacked())

    def _strip(self, seq: list[int]) -> list[int]:
        eos = self.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for i, token in enumerate(seq):
            if token in eos:
                return seq[:i]
        return seq

    @torch.no_grad()
    def prefix_logits(self, texts: list[str], prefixes: list[list[int]], n_steps: int,
                      logit_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Next-token logits at decoding steps 1..n_steps along given prefixes (teacher forcing)."""
        pad = self.tokenizer.pad_token_id
        prefixes = [(list(p) + [pad] * n_steps)[:n_steps - 1] for p in prefixes]
        logits = self.forward(*self.encode(texts, prefixes))[:, -n_steps:].float()
        if logit_bias is not None:
            logits[:, :BIAS_STEPS] += logit_bias.to(logits)
        return logits
