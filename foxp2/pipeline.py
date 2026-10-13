"""End-to-end experiment for one model-language pair: build the edit, tune baselines, evaluate."""
import json
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from . import baselines as B
from .config import MODELS, TARGETS, Hyperparameters, Paths
from .data import Splits, build_splits, load_flores
from .evaluate import NO_EDIT, Evaluator, Method, Scores, matched_shift_rate
from .langid import LanguageIdentifier, in_language, sentence_languages
from .model import LM
from .prompts import CUE_LEVELS, SHORT_INSTRUCTION, STRATEGIES, SYSTEM_PROMPT, TRANSLATE, load_prompt_set
from .sae import SAEBank
from .score import auc, defaultness, select_horizon
from .stage1 import stage1
from .stage2 import LayerSubspace, candidate_windows, stage2
from .stage3 import LayerEdit, NeuralFoxp2, subspace_ablation
from .stats import bootstrap_ci, holm, sign_flip_test
from .tokens import TokenSets, build_token_sets
from .tuning import UtilityFn, tune


def layer_range(window) -> range:
    return range(window[0], window[1] + 1)


class Experiment:
    def __init__(self, model: str, lang: str, hyper: Hyperparameters | None = None,
                 paths: Paths | None = None, device: str = "cuda", log=print):
        self.model_key, self.lang = model, lang
        self.spec = MODELS[model]
        self.hyper = hyper or Hyperparameters()
        self.paths = paths or Paths()
        self.device = device
        self.log = log
        self.dir = Path(self.paths.run_dir(model, lang))
        self.dir.mkdir(parents=True, exist_ok=True)

        self.lm = LM(self.spec.name, device)
        self.splits = self._splits()
        self.sets = self._token_sets()
        self.lid = LanguageIdentifier()
        self.evaluator = Evaluator(self.lm, self.sets, self.lid, lang, self.hyper)
        self.saes = SAEBank(self.spec, device)

    def _splits(self) -> Splits:
        path = self.dir / "splits.json"
        if not path.exists():
            build_splits(self.lang, self.hyper, self.device).save(str(path))
        return Splits.load(str(path))

    def _token_sets(self) -> TokenSets:
        path = self.dir.parent / "token_sets.pt"
        if not path.exists():
            texts = {lang: list(load_flores(lang, "dev").values()) for lang in ("en", *TARGETS)}
            build_token_sets(self.lm.tokenizer, self.lm.vocab_size, texts, list(TARGETS),
                             self.hyper.rho, self.hyper.min_frequency).save(str(path))
        return TokenSets.load(str(path))

    @property
    def dev_prompts(self) -> list[str]:
        return [pair.en for pair in self.splits.dev]

    @cached_property
    def residuals(self) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Final-token residuals of the discovery pairs at every layer: (target, English)."""
        layers = list(range(self.lm.n_layers))
        target = self.lm.residuals([self.lm.chat(p.tgt) for p in self.splits.disc], layers, self.hyper.batch_size)
        english = self.lm.residuals([self.lm.chat(p.en) for p in self.splits.disc], layers, self.hyper.batch_size)
        return target, english

    @cached_property
    def dev_baseline(self) -> Scores:
        return B.baseline_scores(self.evaluator, self.dev_prompts)

    # Neural FOXP2

    def build(self, layers: list[int] | None = None, window: tuple[int, int] | None = None,
              utility: UtilityFn | None = None) -> NeuralFoxp2:
        hyper = self.hyper
        layers = list(layer_range(window)) if window else (layers or list(range(self.lm.n_layers)))
        h_tgt, h_en = self.residuals
        lift_prompts = self.evaluator.format(NO_EDIT, self.dev_prompts[:hyper.n_lift_prompts])

        edits, subspaces, profile = {}, {}, []
        for layer in layers:
            sae = self.saes[layer]
            features = stage1(self.evaluator, sae, h_tgt[layer], h_en[layer], lift_prompts, hyper)
            subspace = stage2(features, hyper)
            edits[layer] = LayerEdit.build(subspace, sae.decoder(subspace.support))
            subspaces[layer] = subspace
            self.saes.release()
            profile.append({"layer": layer, "support": len(subspace.support), "rank": subspace.rank,
                            "effective_rank": subspace.effective_rank, "stability": subspace.stability,
                            "selectivity": float(features.selectivity.mean()), "lift": float(features.lift.mean())})
            self.log(f"layer {layer}: rank={subspace.rank} r_eff={subspace.effective_rank:.1f} "
                     f"stability={subspace.stability:.3f}")
        (self.dir / "layer_profile.json").write_text(json.dumps(profile, indent=1))

        if window is None:
            window = self.search_window(edits, subspaces, utility)
        self.saes.release(keep=set(layer_range(window)))

        foxp2 = NeuralFoxp2({l: edits[l] for l in layer_range(window)}, self.saes, 1.0,
                            hyper.lam, hyper.beta, hyper.eps)
        self.log(f"window {window}: tuning gamma")
        best, _ = tune(self.evaluator, lambda g: foxp2.with_settings(gamma=g).method(), hyper.gamma_grid,
                       self.dev_prompts, self.dev_baseline, utility, self.log)
        foxp2 = foxp2.with_settings(gamma=best.setting)
        foxp2.save(str(self.dir / "neural_foxp2.pt"), window=window, constrained=not best.feasible,
                   model=self.model_key, lang=self.lang)
        self.log(f"saved Neural FOXP2: window={window} gamma={best.setting} constrained={not best.feasible}")
        return foxp2

    def search_window(self, edits: dict[int, LayerEdit], subspaces: dict[int, LayerSubspace],
                      utility: UtilityFn | None = None) -> tuple[int, int]:
        hyper = self.hyper
        windows = candidate_windows(subspaces, self.lm.n_layers, hyper)
        if not windows:
            raise RuntimeError("No contiguous window of stable layers; lower tau_stab or include more layers.")
        prompts = self.dev_prompts[:hyper.n_search_prompts]
        baseline = B.baseline_scores(self.evaluator, prompts)

        def build(window):
            layers = {l: edits[l] for l in layer_range(window)}
            return NeuralFoxp2(layers, self.saes, hyper.probe_gamma, hyper.lam, hyper.beta, hyper.eps).method()

        self.log(f"searching {len(windows)} windows")
        best, _ = tune(self.evaluator, build, windows, prompts, baseline, utility, self.log)
        return tuple(best.setting)

    def load_edit(self) -> NeuralFoxp2:
        return NeuralFoxp2.load(str(self.dir / "neural_foxp2.pt"), self.saes)

    # Baselines

    @property
    def settings_path(self) -> Path:
        return self.dir / "baselines.json"

    def load_settings(self) -> dict:
        return json.loads(self.settings_path.read_text()) if self.settings_path.exists() else {}

    def baselines(self, foxp2: NeuralFoxp2, utility: UtilityFn | None = None,
                  with_lora: bool = True) -> dict[str, Method]:
        """Every comparison method at its operating point, tuned on D_dev under the same guardrails."""
        settings = self.load_settings()
        window = foxp2.window

        if "best_prompt" not in settings:
            method, rates = B.best_prompt(self.evaluator, self.dev_prompts)
            settings["best_prompt"] = method.style.name
            self.log(f"best prompt: {method.style.name} ({rates})")
        strategy = next(s for s in STRATEGIES if s.name == settings["best_prompt"])

        methods = {
            "no_edit": NO_EDIT,
            "short_instruction": B.short_instruction(),
            "system_prompt": B.system_prompt(),
            "best_prompt": Method("best_prompt", style=strategy),
        }
        for name, (grid, build) in self._tunable(window).items():
            if name not in settings:
                self.log(f"tuning {name}")
                best, _ = tune(self.evaluator, build, grid, self.dev_prompts, self.dev_baseline, utility, self.log)
                settings[name] = best.setting
                self.settings_path.write_text(json.dumps(settings, indent=1))
            methods[name] = build(settings[name])

        if with_lora:
            if "lora" not in settings:
                settings["lora"] = self._train_lora()
            methods["lora"] = B.lora(settings["lora"])

        methods["neural_foxp2"] = foxp2.method()
        methods["neural_foxp2_system"] = foxp2.method("neural_foxp2_system", SYSTEM_PROMPT)
        self.settings_path.write_text(json.dumps(settings, indent=1))
        return methods

    def _tunable(self, window) -> dict:
        hyper = self.hyper
        h_tgt, h_en = self.residuals
        layers = list(layer_range(window))
        gammas = hyper.gamma_grid

        def lape_parts():
            texts = {lang: list(load_flores(lang, "dev").values())[:200] for lang in ("en", *TARGETS)}
            rates = {lang: B.neuron_activation_rates(self.lm, t, layers) for lang, t in texts.items()}
            neurons = B.lape_neurons(rates, self.lang, hyper.k_features)
            shift_tgt = B.neuron_inputs(self.lm, [self.lm.chat(p.tgt) for p in self.splits.disc], layers)
            shift_en = B.neuron_inputs(self.lm, [self.lm.chat(p.en) for p in self.splits.disc], layers)
            return neurons, shift_tgt, shift_en

        cache = {}

        def lape(g):
            if "lape" not in cache:
                cache["lape"] = lape_parts()
            return B.lape(*cache["lape"], g)

        def sae_features(setting):
            n, g = setting
            if n not in cache:
                cache[n] = B.sae_feature_vectors(self.saes, h_tgt, h_en, window, n, hyper.eps)
            return B.sae_features(cache[n], g)

        return {
            "diff_in_means": (gammas, lambda g: B.diff_in_means(h_tgt, h_en, window, g)),
            "diff_in_means_projection": (gammas, lambda g: B.diff_in_means_projection(h_tgt, h_en, window, g, hyper)),
            "lape_neurons": (gammas, lape),
            "sae_features": ([(n, g) for n in (1, 4, 16) for g in gammas], sae_features),
            "opening_logit_bias": ((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
                                   lambda s: B.opening_logit_bias(self.sets, self.lang, s)),
        }

    def _train_lora(self) -> str:
        prompts = [pair.en for pair in self.splits.disc]
        examples = B.lora_data(self.evaluator, prompts)
        self.log(f"LoRA: {len(examples)} of {len(prompts)} discovery answers kept")
        checkpoints = B.train_lora(self.lm, examples, str(self.dir / "lora"), seed=self.hyper.seed)
        rates = {path: B.full_rate(self.evaluator, B.lora(path), self.dev_prompts) for path in checkpoints}
        self.log(f"LoRA dev full-response rates: {rates}")
        return max(rates, key=rates.get)

    # Evaluation

    def evaluate(self, methods: dict[str, Method], prompt_set: str = "no_cue") -> dict:
        prompts = load_prompt_set(prompt_set)
        reference = self.evaluator.run(NO_EDIT, prompts)
        base = self.evaluator.score(reference)
        scores, results = {}, {}
        out_dir = self.dir / prompt_set
        out_dir.mkdir(exist_ok=True)
        for name, method in methods.items():
            self.log(f"evaluating {name} on {prompt_set}")
            out = reference if method.is_identity else self.evaluator.run(method, prompts, reference=reference)
            s = scores[name] = self.evaluator.score(out)
            results[name] = {**self.evaluator.summary(s, base), "full_ci": bootstrap_ci(s.full, self.hyper.n_boot)}
            with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for prompt, response in zip(prompts, out.responses):
                    f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")

        if "neural_foxp2" in scores:
            tests = {name: sign_flip_test(scores["neural_foxp2"].full, s.full, self.hyper.n_perm)
                     for name, s in scores.items() if name != "neural_foxp2"}
            adjusted = holm({name: p for name, (_, p) in tests.items()})
            results["tests_vs_neural_foxp2"] = {
                name: {"statistic": stat, "p": p, "holm_p": adjusted[name]} for name, (stat, p) in tests.items()}
        (self.dir / f"results_{prompt_set}.json").write_text(json.dumps(results, indent=1))
        return results

    def matched_shift(self, foxp2: NeuralFoxp2, prompt_set: str = "no_cue") -> dict:
        """Dense steering swept in strength and read off at the KL_ref of Neural FOXP2 (Section 6.1)."""
        prompts = load_prompt_set(prompt_set)
        reference = self.evaluator.run(NO_EDIT, prompts)

        def point(method: Method) -> tuple[float, float]:
            scores = self.evaluator.score(self.evaluator.run(method, prompts, reference=reference))
            return float(scores.kl_ref.mean()), float(scores.full.mean())

        target_kl, rate = point(foxp2.method())
        results = {"neural_foxp2": {"kl_ref": target_kl, "full": rate}}
        tunable = self._tunable(foxp2.window)
        n_features = self.load_settings().get("sae_features", [4])[0]
        sweeps = {name: (self.hyper.gamma_grid, tunable[name][1])
                  for name in ("diff_in_means", "diff_in_means_projection", "lape_neurons")}
        sweeps["sae_features"] = (self.hyper.gamma_grid, lambda g: tunable["sae_features"][1]((n_features, g)))
        for name, (grid, build) in sweeps.items():
            self.log(f"sweeping {name}")
            points = [point(build(g)) for g in grid]
            kls, rates = zip(*points)
            results[name] = {"sweep": points, "full_at_matched_kl": matched_shift_rate(kls, rates, target_kl)}
        (self.dir / f"matched_shift_{prompt_set}.json").write_text(json.dumps(results, indent=1))
        return results

    def ablations(self, foxp2: NeuralFoxp2, prompt_set: str = "no_cue") -> dict:
        """Promotion-only and suppression-only edits, and the necessity projection (Appendix I)."""
        variants = {
            "neural_foxp2": foxp2.method(),
            "promotion_only": foxp2.with_settings(beta=0.0).method("promotion_only"),
            "suppression_only": foxp2.with_settings(lam=0.0).method("suppression_only"),
        }
        necessity = {
            "answer_in_x": Method("answer_in_x", style=SHORT_INSTRUCTION),
            "answer_in_x_without_subspace": subspace_ablation(foxp2.layers, "answer_in_x_without_subspace",
                                                              SHORT_INSTRUCTION),
        }
        return {"ablations": self.evaluate(variants, prompt_set), "necessity": self.evaluate(necessity, prompt_set)}

    # Validation of the defaultness score

    @torch.no_grad()
    def validate_horizon(self, n_items: int = 100, max_k: int = 10) -> dict:
        """Test D2 on cue-level prompts: AUC of Delta M at horizons 1..max_k for the full-response label."""
        items = load_prompt_set("no_cue")[:n_items]
        translated = self._translate(items)
        texts = [t for level in CUE_LEVELS for t in self.evaluator.format(Method(level.name, style=level), items)]
        texts += self.evaluator.format(NO_EDIT, translated)

        by_k, labels = {k: [] for k in range(1, max_k + 1)}, []
        for batch in self.evaluator.batches(texts):
            gen = self.lm.generate(batch, self.hyper.max_new_tokens, opening_steps=max_k)
            for k in by_k:
                dm = defaultness(gen.opening_logits, self.evaluator.sets, range(1, k + 1))
                by_k[k].append(dm[:, self.evaluator.column].cpu())
            labels += [in_language(sentence_languages(r, self.lid, self.hyper.min_sentence_chars), self.lang)
                       for r in gen.texts]
        scores = {k: torch.cat(v).numpy() for k, v in by_k.items()}
        result = {"auc": {k: auc(s, labels) for k, s in scores.items()},
                  "selected_horizon": select_horizon(scores, labels), "positive_rate": float(np.mean(labels))}
        (self.dir / "validate_horizon.json").write_text(json.dumps(result, indent=1))
        return result

    def _translate(self, items: list[str]) -> list[str]:
        texts = self.evaluator.format(Method("translate", style=TRANSLATE), items)
        return [t for batch in self.evaluator.batches(texts) for t in self.lm.generate(batch, 128).texts]
