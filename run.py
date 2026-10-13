"""Neural FOXP2 command line.

    python run.py build    --model llama-3.1-8b --lang es
    python run.py evaluate --model llama-3.1-8b --lang es --set no_cue
    python run.py sweep    --model llama-3.1-8b --lang es
    python run.py ablate   --model llama-3.1-8b --lang es
    python run.py validate --model llama-3.1-8b --lang es
"""
import argparse
import json

from foxp2.config import MODELS, PAPER_ARTIFACTS, TARGETS, Hyperparameters, Paths
from foxp2.pipeline import Experiment
from foxp2.utility import Utility


def parse_layers(text: str | None) -> list[int] | None:
    if not text:
        return None
    start, _, end = text.partition("-")
    return list(range(int(start), int(end or start) + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "evaluate", "sweep", "ablate", "validate"])
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--lang", required=True, choices=list(TARGETS))
    parser.add_argument("--layers", help="restrict Stages I-II to a layer range, e.g. 8-24")
    parser.add_argument("--paper-window", action="store_true", help="use the window of Table 4 instead of searching")
    parser.add_argument("--utility", action="store_true", help="include the utility guardrail while tuning")
    parser.add_argument("--set", default="no_cue", choices=["no_cue", "neutral", "intent"])
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    hyper = Hyperparameters()
    exp = Experiment(args.model, args.lang, hyper, Paths(root=args.output), args.device)
    utility = Utility(exp.evaluator, exp.splits.flores_utility, n=50, part="dev") if args.utility else None

    if args.command == "build":
        window = PAPER_ARTIFACTS[args.model][args.lang][0] if args.paper_window else None
        exp.build(parse_layers(args.layers), window, utility)
        return

    foxp2 = exp.load_edit()
    if args.command == "evaluate":
        methods = exp.baselines(foxp2, utility, with_lora=not args.no_lora)
        results = exp.evaluate(methods, args.set)
        if args.utility:
            evaluation = Utility(exp.evaluator, exp.splits.flores_utility, n=150, part="eval")
            results["utility"] = {name: evaluation.change(m) for name, m in methods.items() if name != "no_edit"}
    elif args.command == "sweep":
        results = exp.matched_shift(foxp2, args.set)
    elif args.command == "ablate":
        results = exp.ablations(foxp2, args.set)
    else:
        results = exp.validate_horizon()
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
