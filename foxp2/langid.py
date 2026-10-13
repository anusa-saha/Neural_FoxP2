import re
import urllib.request
from pathlib import Path

from .config import LANGUAGES

LID176_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
CODE_BLOCK = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？؟।])\s*|\n+")


def _fasttext(path: str):
    import fasttext

    return fasttext.load_model(path)


class LanguageIdentifier:
    """GlotLID as the primary identifier and fastText lid.176 as the second (Appendix F)."""

    def __init__(self, cache_dir: str = "~/.cache/foxp2"):
        from huggingface_hub import hf_hub_download

        cache = Path(cache_dir).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        lid176 = cache / "lid.176.bin"
        if not lid176.exists():
            urllib.request.urlretrieve(LID176_URL, lid176)
        self.glotlid = _fasttext(hf_hub_download("cis-lmu/glotlid", "model.bin"))
        self.lid176 = _fasttext(str(lid176))
        self.labels = {label: code for code, lang in LANGUAGES.items() for label in lang.glotlid}

    def __call__(self, text: str) -> str | None:
        """Language code of `text` (e.g. "es"), the raw GlotLID label if outside the suite, or None."""
        text = " ".join(text.split())
        if not text:
            return None
        label = self.glotlid.predict(text)[0][0].removeprefix("__label__")
        return self.labels.get(label, label)

    def secondary(self, text: str) -> str | None:
        text = " ".join(text.split())
        if not text:
            return None
        return self.lid176.predict(text)[0][0].removeprefix("__label__")


def strip_code(text: str) -> str:
    return CODE_BLOCK.sub(" ", text)


def sentences(text: str, min_chars: int = 20) -> list[str]:
    parts = (part.strip() for part in SENTENCE_END.split(strip_code(text)))
    return [part for part in parts if len(part) >= min_chars]


def sentence_languages(text: str, lid: LanguageIdentifier, min_chars: int = 20) -> list[str | None]:
    found = sentences(text, min_chars)
    if not found and text.strip():
        found = [strip_code(text)]
    return [lid(sentence) for sentence in found]


def in_language(labels: list[str | None], lang: str) -> bool:
    """Full-response criterion: every sentence of at least 20 characters is in `lang`."""
    return bool(labels) and all(label == lang for label in labels)


def switches_away(labels: list[str | None], lang: str) -> bool:
    return bool(labels) and labels[0] == lang and any(label != lang for label in labels[1:])


def line_pass(text: str, lang: str, lid: LanguageIdentifier) -> bool:
    """Line-level language pass in the style of the Language Confusion Benchmark."""
    lines = [line for line in strip_code(text).splitlines() if re.search(r"\w{2,}", line)]
    return bool(lines) and all(lid.secondary(line) == lang for line in lines)
