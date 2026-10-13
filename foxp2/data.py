import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import LANGUAGES, Hyperparameters

FLORES = "openlanguagedata/flores_plus"


@dataclass(frozen=True)
class Pair:
    en: str
    tgt: str


@dataclass
class Splits:
    disc: list[Pair]
    dev: list[Pair]
    eval: list[Pair]
    flores_utility: list[Pair]

    def save(self, path: str):
        Path(path).write_text(json.dumps({k: [asdict(p) for p in v] for k, v in vars(self).items()},
                                         ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "Splits":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: [Pair(**p) for p in v] for k, v in data.items()})


def load_flores(lang: str, split: str) -> dict[int, str]:
    rows = load_dataset(FLORES, LANGUAGES[lang].flores, split=split)
    return {row["id"]: row["text"] for row in rows}


def flores_pairs(lang: str, split: str) -> list[Pair]:
    english, target = load_flores("en", split), load_flores(lang, split)
    return [Pair(english[i], target[i]) for i in sorted(english) if i in target]


class EntailmentFilter:
    """Keeps a pair only if each side entails the other with probability >= tau (Appendix B)."""

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.entail = self.model.config.label2id.get("entailment", 0)
        self.batch_size = batch_size

    @torch.no_grad()
    def entailment(self, premises: list[str], hypotheses: list[str]) -> torch.Tensor:
        probs = []
        for i in range(0, len(premises), self.batch_size):
            batch = self.tokenizer(premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size],
                                   padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            probs.append(self.model(**batch).logits.softmax(-1)[:, self.entail].cpu())
        return torch.cat(probs)

    def keep(self, pairs: list[Pair], tau: float) -> list[Pair]:
        en, tgt = [p.en for p in pairs], [p.tgt for p in pairs]
        both = torch.minimum(self.entailment(en, tgt), self.entailment(tgt, en))
        return [pair for pair, prob in zip(pairs, both) if prob >= tau]


def build_splits(lang: str, hyper: Hyperparameters, device: str = "cuda") -> Splits:
    """D_disc and D_dev from FLORES dev, D_eval from devtest, minus the FLORES utility sentences."""
    rng = random.Random(hyper.seed)
    dev_pool, test_pool = flores_pairs(lang, "dev"), flores_pairs(lang, "devtest")
    rng.shuffle(dev_pool)
    rng.shuffle(test_pool)
    utility, test_pool = test_pool[:hyper.n_flores_utility], test_pool[hyper.n_flores_utility:]

    nli = EntailmentFilter(hyper.nli_model, device)
    dev_kept = nli.keep(dev_pool, hyper.tau_sem)
    test_kept = nli.keep(test_pool, hyper.tau_sem)
    return Splits(
        disc=dev_kept[:hyper.n_disc],
        dev=dev_kept[hyper.n_disc:hyper.n_disc + hyper.n_dev],
        eval=test_kept[:hyper.n_eval],
        flores_utility=utility,
    )
