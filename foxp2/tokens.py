import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import torch

NO_LETTERS = re.compile(r"^[\W\d_]*$")
URL_PIECE = re.compile(r"^\s*(https?|www\.?|\.?(com|org|net|html?)|:?//?)\s*$", re.IGNORECASE)


@dataclass
class TokenSets:
    """Weighted diagnostic token sets: column i scores langs[i] against English (Appendix F)."""
    langs: list[str]
    target: torch.Tensor
    english: torch.Tensor

    def column(self, lang: str) -> int:
        return self.langs.index(lang)

    def to(self, device) -> "TokenSets":
        return TokenSets(self.langs, self.target.to(device), self.english.to(device))

    def save(self, path: str):
        torch.save({"langs": self.langs, "target": self.target.cpu(), "english": self.english.cpu()}, path)

    @classmethod
    def load(cls, path: str) -> "TokenSets":
        return cls(**torch.load(path))


def token_frequencies(tokenizer, texts: list[str], vocab_size: int) -> torch.Tensor:
    counts = Counter()
    for text in texts:
        counts.update(tokenizer(unicodedata.normalize("NFKC", text), add_special_tokens=False).input_ids)
    freq = torch.zeros(vocab_size)
    for token, count in counts.items():
        freq[token] = count
    return freq / freq.sum()


def token_weights(tokenizer, vocab_size: int) -> torch.Tensor:
    """w(u) = 0 for the shared pool (digits, punctuation, whitespace, URL fragments), else 1."""
    pieces = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
    weights = torch.ones(vocab_size)
    for i, piece in enumerate(pieces):
        if NO_LETTERS.match(piece) or URL_PIECE.match(piece):
            weights[i] = 0.0
    weights[len(pieces):] = 0.0
    return weights


def build_token_sets(tokenizer, vocab_size: int, texts: dict[str, list[str]], langs: list[str],
                     rho: float, min_frequency: float) -> TokenSets:
    weights = token_weights(tokenizer, vocab_size)
    english = token_frequencies(tokenizer, texts["en"], vocab_size)
    target_cols, english_cols = [], []
    for lang in langs:
        freq = token_frequencies(tokenizer, texts[lang], vocab_size)
        is_target = (freq >= rho * english) & (freq >= min_frequency)
        is_english = (english >= rho * freq) & (english >= min_frequency)
        target_cols.append(weights * is_target)
        english_cols.append(weights * is_english)
    return TokenSets(list(langs), torch.stack(target_cols, 1), torch.stack(english_cols, 1))
