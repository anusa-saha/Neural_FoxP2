"""LLM judge for answer quality (regime C), refusals (Appendix K.3) and language labels."""
import json
import re

from .config import LANGUAGES
from .model import LM
from .prompts import FULL_RESPONSE_JUDGE, QUALITY_JUDGE, REFUSAL_JUDGE


class Judge:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 8):
        self.lm = LM(model_name, device)
        self.batch_size = batch_size

    def ask(self, questions: list[str], max_new_tokens: int = 64) -> list[str]:
        answers = []
        for start in range(0, len(questions), self.batch_size):
            texts = [self.lm.chat(q) for q in questions[start:start + self.batch_size]]
            answers += self.lm.generate(texts, max_new_tokens).texts
        return answers

    def quality(self, prompts: list[str], responses: list[str], lang: str) -> list[dict]:
        name = LANGUAGES[lang].name
        answers = self.ask([QUALITY_JUDGE.format(prompt=p, response=r, lang=name) for p, r in zip(prompts, responses)])
        return [parse_json(a) for a in answers]

    def refusals(self, prompts: list[str], responses: list[str]) -> list[bool]:
        answers = self.ask([REFUSAL_JUDGE.format(prompt=p, response=r) for p, r in zip(prompts, responses)], 8)
        return ["refusal" in a.lower() for a in answers]

    def in_language(self, responses: list[str], lang: str) -> list[bool]:
        name = LANGUAGES[lang].name
        answers = self.ask([FULL_RESPONSE_JUDGE.format(response=r, lang=name) for r in responses], 4)
        return [a.strip().lower().startswith("yes") for a in answers]


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}
