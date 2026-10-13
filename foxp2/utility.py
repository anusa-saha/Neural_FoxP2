"""Utility change Delta S over six benchmarks (Appendix F, regimes A and B)."""
import random
import re

import numpy as np
import torch
from datasets import load_dataset

from .config import LANGUAGES
from .data import Pair
from .evaluate import NO_EDIT, Evaluator, Method
from .prompts import (FLORES_TEMPLATE, GSM8K_TEMPLATE, MULTIPLE_CHOICE_OPTIONS, MULTIPLE_CHOICE_TEMPLATE,
                      SUMMARY_TEMPLATE, XNLI_OPTIONS, XNLI_TEMPLATE)

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def rouge_l(prediction: list[int], reference: list[int]) -> float:
    if not prediction or not reference:
        return 0.0
    previous = [0] * (len(reference) + 1)
    for p in prediction:
        current = [0]
        for j, r in enumerate(reference):
            current.append(previous[j] + 1 if p == r else max(previous[j + 1], current[j]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(prediction), lcs / len(reference)
    return 2 * precision * recall / (precision + recall)


def last_number(text: str) -> float | None:
    match = re.search(r"answer is\s*\$?(-?\d[\d,]*\.?\d*)", text, re.IGNORECASE)
    found = match.group(1) if match else (NUMBER.findall(text) or [None])[-1]
    try:
        return float(found.replace(",", "").rstrip("."))
    except (AttributeError, ValueError):
        return None


class Utility:
    def __init__(self, evaluator: Evaluator, flores_pairs: list[Pair], n: int = 200, part: str = "eval",
                 seed: int = 17, comet_model: str | None = "Unbabel/wmt22-comet-da"):
        self.evaluator = evaluator
        self.lm = evaluator.lm
        self.lang = evaluator.lang
        self.n = n
        self.part = part
        self.seed = seed
        self.flores = self.take(flores_pairs)
        self.comet = None
        if comet_model:
            from comet import download_model, load_from_checkpoint

            self.comet = load_from_checkpoint(download_model(comet_model))
        self.benchmarks = {"xnli": self.xnli, "belebele": self.belebele, "xlsum": self.xlsum,
                           "flores": self.flores_translation, "mmlu": self.mmlu, "gsm8k": self.gsm8k}
        self._reference = None

    def take(self, items: list) -> list:
        """Evaluation items come from the front and development items from the back, so they never overlap."""
        return items[:self.n] if self.part == "eval" else items[-self.n:]

    def sample(self, dataset) -> list:
        order = list(range(len(dataset)))
        random.Random(self.seed).shuffle(order)
        return [dataset[i] for i in self.take(order)]

    def render(self, method: Method, question: str) -> str:
        system, user = method.style.render(question, self.lang)
        return self.lm.chat(user, system)

    @torch.no_grad()
    def choose(self, method: Method, questions: list[str], options: list[str]) -> np.ndarray:
        """Index of the option with the highest log-likelihood as the assistant's answer."""
        tokenizer, scores = self.lm.tokenizer, []
        for option in options:
            ids = tokenizer(option, add_special_tokens=False).input_ids
            per_option = []
            for batch in self.evaluator.batches(questions):
                texts = [self.render(method, q) for q in batch]
                with method.active(self.lm):
                    logits = self.lm.prefix_logits(texts, [ids] * len(batch), len(ids), method.logit_bias)
                logprobs = logits.log_softmax(-1)[:, torch.arange(len(ids)), ids].sum(-1)
                per_option.append(logprobs.cpu())
            scores.append(torch.cat(per_option))
        return torch.stack(scores, 1).argmax(1).numpy()

    def generate(self, method: Method, questions: list[str], max_new_tokens: int) -> list[str]:
        responses = []
        for batch in self.evaluator.batches(questions):
            with method.active(self.lm):
                responses += self.lm.generate([self.render(method, q) for q in batch], max_new_tokens,
                                              logit_bias=method.logit_bias).texts
        return responses

    def xnli(self, method: Method) -> float:
        rows = self.sample(load_dataset("facebook/xnli", self.lang, split="test"))
        questions = [XNLI_TEMPLATE.format(**row) for row in rows]
        predicted = self.choose(method, questions, XNLI_OPTIONS)
        return 100 * float(np.mean(predicted == np.array([row["label"] for row in rows])))

    def belebele(self, method: Method) -> float:
        rows = self.sample(load_dataset("facebook/belebele", LANGUAGES[self.lang].belebele, split="test"))
        questions = [MULTIPLE_CHOICE_TEMPLATE.format(
            context=row["flores_passage"] + "\n\n", question=row["question"], a=row["mc_answer1"],
            b=row["mc_answer2"], c=row["mc_answer3"], d=row["mc_answer4"]) for row in rows]
        predicted = self.choose(method, questions, MULTIPLE_CHOICE_OPTIONS)
        gold = np.array([int(row["correct_answer_num"]) - 1 for row in rows])
        return 100 * float(np.mean(predicted == gold))

    def mmlu(self, method: Method) -> float:
        rows = self.sample(load_dataset("cais/mmlu", "all", split="test"))
        questions = [MULTIPLE_CHOICE_TEMPLATE.format(context="", question=row["question"], a=row["choices"][0],
                                                     b=row["choices"][1], c=row["choices"][2], d=row["choices"][3])
                     for row in rows]
        predicted = self.choose(method, questions, MULTIPLE_CHOICE_OPTIONS)
        return 100 * float(np.mean(predicted == np.array([row["answer"] for row in rows])))

    def gsm8k(self, method: Method) -> float:
        rows = self.sample(load_dataset("openai/gsm8k", "main", split="test"))
        responses = self.generate(method, [GSM8K_TEMPLATE.format(question=row["question"]) for row in rows], 384)
        gold = [float(row["answer"].split("####")[-1].strip().replace(",", "")) for row in rows]
        return 100 * float(np.mean([last_number(r) == g for r, g in zip(responses, gold)]))

    def xlsum(self, method: Method) -> float:
        rows = self.sample(load_dataset("csebuetnlp/xlsum", LANGUAGES[self.lang].xlsum, split="test"))
        responses = self.generate(method, [SUMMARY_TEMPLATE.format(text=row["text"][:3000]) for row in rows], 96)
        tokens = lambda text: self.lm.tokenizer(text, add_special_tokens=False).input_ids  # noqa: E731
        return 100 * float(np.mean([rouge_l(tokens(r), tokens(row["summary"])) for r, row in zip(responses, rows)]))

    def flores_translation(self, method: Method) -> float:
        from sacrebleu.metrics import CHRF

        name = LANGUAGES[self.lang].name
        sources = [pair.en for pair in self.flores]
        references = [pair.tgt for pair in self.flores]
        hypotheses = self.generate(method, [FLORES_TEMPLATE.format(lang=name, text=s) for s in sources], 256)
        chrf = CHRF(word_order=2).corpus_score(hypotheses, [references]).score / 100
        if self.comet is None:
            return 100 * chrf
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
        comet = self.comet.predict(data, batch_size=32, progress_bar=False).system_score
        return 100 * (chrf + comet) / 2

    def scores(self, method: Method) -> dict[str, float]:
        return {name: run(method) for name, run in self.benchmarks.items()}

    def reference(self) -> dict[str, float]:
        if self._reference is None:
            self._reference = self.scores(NO_EDIT)
        return self._reference

    def change(self, method: Method) -> dict[str, float]:
        """Per-benchmark change against the unedited model, in points."""
        reference = self.reference()
        return {name: value - reference[name] for name, value in self.scores(method).items()}

    def __call__(self, method: Method) -> float:
        """Delta S: the predeclared mean of the six benchmark changes."""
        return float(np.mean(list(self.change(method).values())))
