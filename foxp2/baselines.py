"""The nine comparison methods of Section 5 and Appendix E."""
import random
from pathlib import Path

import torch

from .config import Hyperparameters
from .evaluate import NO_EDIT, Evaluator, Method, Scores
from .langid import in_language, sentence_languages
from .model import LM
from .prompts import SHORT_INSTRUCTION, STRATEGIES, SYSTEM_PROMPT
from .sae import SAEBank
from .stage1 import selectivity
from .tokens import TokenSets


# Prompting

def short_instruction() -> Method:
    return Method("short_instruction", style=SHORT_INSTRUCTION)


def system_prompt() -> Method:
    return Method("system_prompt", style=SYSTEM_PROMPT)


def best_prompt(evaluator: Evaluator, dev_prompts: list[str]) -> tuple[Method, dict[str, float]]:
    """The strategy among P1-P10 with the highest DEFAULT on D_dev (an optimistic upper bound)."""
    rates = {}
    for style in STRATEGIES:
        out = evaluator.run(Method(style.name, style=style), dev_prompts, max_new_tokens=evaluator.hyper.lid_tokens)
        rates[style.name] = float(evaluator.score(out, full_text=False).default[evaluator.lang].mean())
    best = max(rates, key=rates.get)
    return Method("best_prompt", style=next(s for s in STRATEGIES if s.name == best)), rates


# Dense activation steering

def add_vector(v: torch.Tensor):
    return lambda h: h + v.to(h.device, h.dtype)


def remove_and_add(v: torch.Tensor, english: torch.Tensor, lam: float, beta: float, eps: float):
    """h + lambda v - beta a(h) h_en, the dense counterpart of Eq. 7."""
    norm = english @ english + eps

    def edit(h):
        e = english.to(h.device, torch.float32)
        a = (h.float() @ e) / norm
        delta = lam * v.to(h.device, torch.float32) - beta * a[..., None] * e
        return h + delta.to(h.dtype)

    return edit


def diff_in_means(h_tgt: dict[int, torch.Tensor], h_en: dict[int, torch.Tensor], window, gamma: float) -> Method:
    edits = {l: add_vector(gamma * (h_tgt[l] - h_en[l]).mean(0)) for l in range(window[0], window[1] + 1)}
    return Method("diff_in_means", edits=edits)


def diff_in_means_projection(h_tgt, h_en, window, gamma: float, hyper: Hyperparameters) -> Method:
    edits = {l: remove_and_add((h_tgt[l] - h_en[l]).mean(0), h_en[l].mean(0), gamma * hyper.lam,
                               gamma * hyper.beta, hyper.eps) for l in range(window[0], window[1] + 1)}
    return Method("diff_in_means_projection", edits=edits)


# LAPE neurons (Tang et al., 2024)

@torch.no_grad()
def neuron_activation_rates(lm: LM, texts: list[str], layers: list[int],
                            batch_size: int = 16) -> dict[int, torch.Tensor]:
    """Share of tokens on which each FFN neuron's activation is positive."""
    totals = {l: 0.0 for l in layers}
    state = {"mask": None, "tokens": 0}

    def make_hook(layer):
        def hook(module, inputs, output):
            totals[layer] = totals[layer] + (output > 0)[state["mask"]].float().sum(0).cpu()
        return hook

    handles = [lm.layers[l].mlp.act_fn.register_forward_hook(make_hook(l)) for l in layers]
    try:
        for start in range(0, len(texts), batch_size):
            ids, mask = lm.encode(texts[start:start + batch_size])
            state["mask"] = mask.bool()
            state["tokens"] += int(mask.sum())
            lm.forward(ids, mask)
    finally:
        for handle in handles:
            handle.remove()
    return {l: totals[l] / state["tokens"] for l in layers}


@torch.no_grad()
def neuron_inputs(lm: LM, texts: list[str], layers: list[int], batch_size: int = 16) -> dict[int, torch.Tensor]:
    """down_proj inputs at the final prompt token."""
    captured = {l: [] for l in layers}

    def make_hook(layer):
        return lambda module, inputs: captured[layer].append(inputs[0][:, -1].float().cpu())

    handles = [lm.layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(l)) for l in layers]
    try:
        for start in range(0, len(texts), batch_size):
            lm.forward(*lm.encode(texts[start:start + batch_size]))
    finally:
        for handle in handles:
            handle.remove()
    return {l: torch.cat(v) for l, v in captured.items()}


def lape_neurons(rates: dict[str, dict[int, torch.Tensor]], lang: str, k: int) -> dict[int, torch.Tensor]:
    """Per layer, the k lowest-entropy neurons whose activation probability peaks on `lang`."""
    langs = list(rates)
    selected = {}
    for layer in rates[lang]:
        probs = torch.stack([rates[l][layer] for l in langs])
        dist = probs / probs.sum(0).clamp(min=1e-12)
        entropy = -(dist * dist.clamp(min=1e-12).log()).sum(0)
        entropy[probs.argmax(0) != langs.index(lang)] = float("inf")
        selected[layer] = entropy.topk(k, largest=False).indices
    return selected


def lape(neurons: dict[int, torch.Tensor], shift_tgt: dict[int, torch.Tensor], shift_en: dict[int, torch.Tensor],
         gamma: float) -> Method:
    edits = {}
    for layer, idx in neurons.items():
        shift = torch.zeros(shift_tgt[layer].shape[1])
        shift[idx] = (shift_tgt[layer][:, idx] - shift_en[layer][:, idx]).mean(0)
        edits[layer] = add_vector(gamma * shift)
    return Method("lape_neurons", neuron_edits=edits)


# SAE features ranked by selectivity alone (Chou et al., 2025)

def sae_feature_vectors(saes: SAEBank, h_tgt, h_en, window, n: int, eps: float) -> dict[int, torch.Tensor]:
    vectors = {}
    for layer in range(window[0], window[1] + 1):
        sae = saes[layer]
        z_tgt, z_en = sae.encode_batched(h_tgt[layer]), sae.encode_batched(h_en[layer])
        top = selectivity(z_tgt, z_en, eps).topk(n).indices
        vectors[layer] = (sae.decoder(top).cpu() @ (z_tgt - z_en)[:, top].mean(0))
    return vectors


def sae_features(vectors: dict[int, torch.Tensor], gamma: float) -> Method:
    return Method("sae_features", edits={l: add_vector(gamma * v) for l, v in vectors.items()})


# Output and weight interventions

def opening_logit_bias(sets: TokenSets, lang: str, scale: float) -> Method:
    return Method("opening_logit_bias", logit_bias=scale * sets.target[:, sets.column(lang)])


def lora_data(evaluator: Evaluator, prompts: list[str]) -> list[tuple[str, str]]:
    """The model's own answers under an explicit instruction, kept if fully in the target language."""
    out = evaluator.run(Method("lora_data", style=SHORT_INSTRUCTION), prompts)
    keep = []
    for prompt, response in zip(prompts, out.responses):
        labels = sentence_languages(response, evaluator.lid, evaluator.hyper.min_sentence_chars)
        if in_language(labels, evaluator.lang):
            keep.append((prompt, response))
    return keep


def train_lora(lm: LM, examples: list[tuple[str, str]], out_dir: str, epochs: int = 3, rank: int = 16,
               alpha: int = 32, lr: float = 2e-4, batch_size: int = 4, seed: int = 17) -> list[str]:
    """Fine-tune LoRA adapters on (plain prompt -> target-language answer); one checkpoint per epoch."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(lm.model, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    eos = lm.tokenizer.eos_token
    data = []
    for prompt, answer in examples:
        context = lm.tokenizer(lm.chat(prompt), add_special_tokens=False).input_ids
        target = lm.tokenizer(answer + eos, add_special_tokens=False).input_ids
        data.append((context, target))

    rng = random.Random(seed)
    checkpoints = []
    model.train()
    for epoch in range(1, epochs + 1):
        rng.shuffle(data)
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            width = max(len(c) + len(t) for c, t in batch)
            ids, labels, mask = [], [], []
            for context, target in batch:
                pad = width - len(context) - len(target)
                ids.append([lm.tokenizer.pad_token_id] * pad + context + target)
                labels.append([-100] * (pad + len(context)) + target)
                mask.append([0] * pad + [1] * (len(context) + len(target)))
            ids, labels, mask = (torch.tensor(x, device=lm.device) for x in (ids, labels, mask))
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            loss = model(input_ids=ids, attention_mask=mask, position_ids=positions, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        path = str(Path(out_dir) / f"epoch{epoch}")
        model.save_pretrained(path)
        checkpoints.append(path)
    model.eval()
    lm.model = model.unload()
    return checkpoints


def lora(path: str) -> Method:
    return Method("lora", adapter=path)


def full_rate(evaluator: Evaluator, method: Method, prompts: list[str]) -> float:
    return float(evaluator.score(evaluator.run(method, prompts)).full.mean())


def baseline_scores(evaluator: Evaluator, prompts: list[str]) -> Scores:
    return evaluator.score(evaluator.run(NO_EDIT, prompts, max_new_tokens=evaluator.hyper.lid_tokens), full_text=False)

