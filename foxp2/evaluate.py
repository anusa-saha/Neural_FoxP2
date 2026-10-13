from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import Hyperparameters
from .langid import LanguageIdentifier, in_language, line_pass, sentence_languages, switches_away
from .model import LM, Edit
from .prompts import FOLLOW_UPS, PLAIN, PromptStyle
from .score import defaultness, horizon_index, kl_divergence
from .tokens import TokenSets


@dataclass
class Method:
    """Anything that changes the response language: a prompt style, residual edits, a logit bias or an adapter."""
    name: str
    style: PromptStyle = PLAIN
    edits: dict[int, Edit] = field(default_factory=dict)
    neuron_edits: dict[int, Edit] = field(default_factory=dict)
    logit_bias: torch.Tensor | None = None
    adapter: str | None = None

    @property
    def is_identity(self) -> bool:
        return (self.style == PLAIN and not self.edits and not self.neuron_edits
                and self.logit_bias is None and self.adapter is None)

    @contextmanager
    def active(self, lm: LM):
        with lm.steer(self.edits, self.neuron_edits), lm.adapter(self.adapter):
            yield


NO_EDIT = Method("no_edit")


@dataclass
class Outputs:
    prompts: list[str]
    responses: list[str]
    tokens: list[list[int]]
    delta_m: torch.Tensor
    opening_lang: list[str | None]
    kl3: torch.Tensor
    kl_ref: torch.Tensor | None


@dataclass
class Scores:
    """Per-prompt outcomes for one method on one prompt set."""
    delta_m: np.ndarray
    default: dict[str, np.ndarray]
    full: np.ndarray
    full_english: np.ndarray
    switch_away: np.ndarray
    line_pass: np.ndarray
    kl3: np.ndarray
    kl_ref: np.ndarray | None


class Evaluator:
    def __init__(self, lm: LM, sets: TokenSets, lid: LanguageIdentifier, lang: str, hyper: Hyperparameters):
        self.lm = lm
        self.sets = sets.to(lm.device)
        self.lid = lid
        self.lang = lang
        self.hyper = hyper
        self.column = sets.column(lang)

    def format(self, method: Method, prompts: list[str]) -> list[str]:
        return [self.lm.chat(user, system) for system, user in (method.style.render(p, self.lang) for p in prompts)]

    def batches(self, items: list, size: int | None = None):
        size = size or self.hyper.batch_size
        for start in range(0, len(items), size):
            yield items[start:start + size]

    @torch.no_grad()
    def opening_delta_m(self, texts: list[str], edits: dict[int, Edit] | None = None) -> torch.Tensor:
        """Target-language Delta M from the first decoding steps only (used by the Stage I lift)."""
        with self.lm.steer(edits or {}):
            gen = self.lm.generate(texts, max_new_tokens=max(self.hyper.horizon), opening_steps=max(self.hyper.horizon))
        return defaultness(gen.opening_logits, self.sets, self.hyper.horizon)[:, self.column].cpu()

    @torch.no_grad()
    def run(self, method: Method, prompts: list[str], max_new_tokens: int | None = None,
            reference: Outputs | None = None) -> Outputs:
        """Generate greedily under `method`, recording Delta M, the opening language and KL drift."""
        hyper, lm = self.hyper, self.lm
        max_new_tokens = max_new_tokens or hyper.max_new_tokens
        steps = max(hyper.horizon)
        responses, tokens, delta_m, opening, kl3, kl_ref = [], [], [], [], [], []
        for start in range(0, len(prompts), hyper.batch_size):
            batch = prompts[start:start + hyper.batch_size]
            texts, plain = self.format(method, batch), self.format(NO_EDIT, batch)
            with method.active(lm):
                gen = lm.generate(texts, max(max_new_tokens, hyper.lid_tokens), steps, method.logit_bias)
            responses += gen.texts
            tokens += gen.tokens
            delta_m.append(defaultness(gen.opening_logits, self.sets, hyper.horizon).cpu())
            openings = lm.tokenizer.batch_decode([t[:hyper.lid_tokens] for t in gen.tokens], skip_special_tokens=True)
            opening += [self.lid(text) for text in openings]

            if method.is_identity:
                kl3.append(torch.zeros(len(batch)))
            else:
                unedited = lm.prefix_logits(plain, gen.tokens, steps)
                kl = kl_divergence(gen.opening_logits, unedited)
                kl3.append(kl[:, horizon_index(hyper.horizon)].mean(1).cpu())

            if reference is not None:
                y0 = reference.tokens[start:start + len(batch)]
                with method.active(lm):
                    edited = lm.prefix_logits(texts, y0, hyper.kl_ref_steps, method.logit_bias)
                unedited = lm.prefix_logits(plain, y0, hyper.kl_ref_steps)
                kl_ref.append(kl_divergence(edited, unedited).mean(1).cpu())

        return Outputs(prompts, responses, tokens, torch.cat(delta_m), opening, torch.cat(kl3),
                       torch.cat(kl_ref) if kl_ref else None)

    def score(self, out: Outputs, full_text: bool = True) -> Scores:
        dm = out.delta_m.numpy()
        opening = np.array(out.opening_lang, dtype=object)
        default = {lang: (dm[:, i] > 0) & (opening == lang) for i, lang in enumerate(self.sets.langs)}
        n = len(out.responses)
        full = english = switch = lines = np.zeros(n, dtype=bool)
        if full_text:
            labels = [sentence_languages(r, self.lid, self.hyper.min_sentence_chars) for r in out.responses]
            full = np.array([in_language(l, self.lang) for l in labels])
            english = np.array([in_language(l, "en") for l in labels])
            switch = np.array([switches_away(l, self.lang) for l in labels])
            lines = np.array([line_pass(r, self.lang, self.lid) for r in out.responses])
        return Scores(dm[:, self.column], default, full, english, switch, lines, out.kl3.numpy(),
                      None if out.kl_ref is None else out.kl_ref.numpy())

    def leakage(self, scores: Scores, baseline: Scores) -> float:
        others = [lang for lang in self.sets.langs if lang != self.lang]
        return float(np.mean([max(0.0, scores.default[l].mean() - baseline.default[l].mean()) for l in others]))

    def summary(self, scores: Scores, baseline: Scores | None = None) -> dict:
        result = {
            "delta_m": float(scores.delta_m.mean()),
            "default": float(scores.default[self.lang].mean()),
            "full": float(scores.full.mean()),
            "full_english": float(scores.full_english.mean()),
            "switch_away": float(scores.switch_away.mean()),
            "line_pass": float(scores.line_pass.mean()),
            "kl3": float(scores.kl3.mean()),
        }
        if scores.kl_ref is not None:
            result["kl_ref"] = float(scores.kl_ref.mean())
        if baseline is not None:
            result["leakage"] = self.leakage(scores, baseline)
        return result

    @torch.no_grad()
    def multi_turn(self, method: Method, prompts: list[str], max_new_tokens: int | None = None) -> list[float]:
        """Full-response rate on turns 1-3 with fixed follow-ups and the method active throughout."""
        max_new_tokens = max_new_tokens or self.hyper.max_new_tokens
        histories = [[] for _ in prompts]
        rates = []
        for turn in range(len(FOLLOW_UPS) + 1):
            responses = []
            for batch_ids in self.batches(list(range(len(prompts)))):
                texts = []
                for i in batch_ids:
                    system, first = method.style.render(prompts[i], self.lang)
                    users = [first, *FOLLOW_UPS[:turn]]
                    texts.append(self.lm.chat(users[-1], system, list(zip(users[:-1], histories[i]))))
                with method.active(self.lm):
                    responses += self.lm.generate(texts, max_new_tokens, logit_bias=method.logit_bias).texts
            for history, response in zip(histories, responses):
                history.append(response)
            labels = [sentence_languages(r, self.lid, self.hyper.min_sentence_chars) for r in responses]
            rates.append(float(np.mean([in_language(l, self.lang) for l in labels])))
        return rates


def matched_shift_rate(kls: list[float], rates: list[float], target_kl: float) -> float:
    """Full-response rate of a strength sweep interpolated at a given KL_ref (Section 6.1)."""
    order = np.argsort(kls)
    return float(np.interp(target_kl, np.asarray(kls)[order], np.asarray(rates)[order]))
