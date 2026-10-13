from dataclasses import dataclass
from typing import Any, Callable

from .evaluate import Evaluator, Method, Scores

UtilityFn = Callable[[Method], float]


@dataclass
class Trial:
    setting: Any
    gain: float
    leak: float
    kl: float
    util: float
    objective: float
    feasible: bool


def assess(evaluator: Evaluator, method: Method, prompts: list[str], baseline: Scores,
           setting: Any = None, utility: UtilityFn | None = None) -> Trial:
    """Gain in DEFAULT, leakage, KL_3 and utility change on D_dev, with the guardrails of Table 3."""
    hyper = evaluator.hyper
    scores = evaluator.score(evaluator.run(method, prompts, max_new_tokens=hyper.lid_tokens), full_text=False)
    gain = float(scores.default[evaluator.lang].mean() - baseline.default[evaluator.lang].mean())
    leak = evaluator.leakage(scores, baseline)
    kl = float(scores.kl3.mean())
    util = utility(method) if utility else 0.0
    objective = gain - hyper.alpha_leak * leak - hyper.alpha_kl * kl - hyper.alpha_util * abs(util)
    feasible = leak <= hyper.eps_leak and kl <= hyper.eps_kl and abs(util) <= hyper.eps_util
    return Trial(setting, gain, leak, kl, util, objective, feasible)


def select(trials: list[Trial]) -> Trial:
    """Best objective among settings that meet every guardrail, else best overall (a constrained pair)."""
    feasible = [t for t in trials if t.feasible]
    return max(feasible or trials, key=lambda t: t.objective)


def tune(evaluator: Evaluator, build: Callable[[Any], Method], settings, prompts: list[str],
         baseline: Scores, utility: UtilityFn | None = None, log: Callable = print) -> tuple[Trial, list[Trial]]:
    trials = []
    for setting in settings:
        trial = assess(evaluator, build(setting), prompts, baseline, setting, utility)
        log(f"  {setting}: gain={trial.gain:+.3f} leak={trial.leak:.3f} kl={trial.kl:.3f} "
            f"util={trial.util:+.2f} objective={trial.objective:+.3f}{'' if trial.feasible else ' (constrained)'}")
        trials.append(trial)
    return select(trials), trials
