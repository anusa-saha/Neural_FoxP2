"""Refusal on harmful prompts and over-refusal on benign ones (Appendix K.3)."""
import csv
from pathlib import Path

import numpy as np
from datasets import load_dataset

from .evaluate import Evaluator, Method
from .judge import Judge


def load_xstest_safe() -> list[str]:
    """Benign but superficially alarming prompts; refusing them is over-refusal."""
    rows = load_dataset("walledai/XSTest", split="test")
    return [row["prompt"] for row in rows if row["label"] == "safe"]


def load_english_harmful() -> list[str]:
    return [row["prompt"] for row in load_dataset("walledai/AdvBench", split="train")]


def load_xsafety(path: str) -> list[str]:
    """Target-language XSafety prompts from a CSV file with one prompt per row (first column)."""
    with Path(path).open(encoding="utf-8") as f:
        return [row[0] for row in csv.reader(f) if row and row[0].strip()]


def refusal_rate(evaluator: Evaluator, judge: Judge, method: Method, prompts: list[str]) -> float:
    responses = evaluator.run(method, prompts).responses
    return float(np.mean(judge.refusals(prompts, responses)))


def safety_report(evaluator: Evaluator, judge: Judge, method: Method, xsafety: list[str] | None = None,
                  n: int = 200) -> dict[str, float]:
    report = {
        "english_refusal": refusal_rate(evaluator, judge, method, load_english_harmful()[:n]),
        "over_refusal": refusal_rate(evaluator, judge, method, load_xstest_safe()[:n]),
    }
    if xsafety:
        report["xsafety_refusal"] = refusal_rate(evaluator, judge, method, xsafety[:n])
    return report
