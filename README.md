# Neural FOXP2

Code for **Neural FOXP2: Measuring and Steering the Default Response Language of Multilingual Language Models**.

When a prompt gives no cue about the answer language, multilingual LMs usually reply in English. This repository

1. **measures** that default with an early-decoding *defaultness score* ΔM read from the first three decoding steps, and
2. **steers** it with Neural FOXP2, a gradient-free edit built from frozen public sparse autoencoders (SAEs):
   - **Stage I** selects SAE features that are both language-selective and causally effective,
   - **Stage II** compresses them into a low-rank subspace per layer and picks a contiguous layer window,
   - **Stage III** adds a signed update inside that window: promote the target language, remove the English component.

The model's weights are never changed, and at γ = 0 the hook reproduces the unedited model exactly.

## Layout

```
run.py                 command line
prompts/               evaluation prompt sets (Appendix B)
  no_cue.txt           500 everyday English requests with no language cue
  neutral.txt          200 prompts made only of arithmetic and symbols
  intent.txt           240 requests that explicitly ask for an English answer
foxp2/
  config.py            languages, models + SAE releases, hyperparameters (Table 3), paper artifacts (Table 4)
  prompts.py           prompting baselines, strategies P1–P10, cue levels L0–L5, judge rubrics, task templates
  data.py              FLORES-200 meaning-matched pairs with the bidirectional entailment filter
  model.py             chat formatting, residual hooks, greedy generation, teacher-forced logits
  sae.py               Llama Scope / Gemma Scope / Qwen-Scope loading
  tokens.py            diagnostic token sets V_tgt, V_en and weights w(u)
  score.py             class mass (Eq. 1), ΔM (Eq. 2), KL, AUC, horizon rule D2
  langid.py            GlotLID + fastText, full-response and switch-away criteria
  stage1.py            selectivity (Eq. 4) and causal lift (Eq. 5)
  stage2.py            effective rank, eigengap, bootstrap stability, candidate windows
  stage3.py            the signed edit (Eqs. 6–8) and the necessity projection
  tuning.py            window / γ objective with the leakage, KL and utility guardrails
  evaluate.py          DEFAULT, full-response rate, leakage, KL₃, KL¹⁰_ref, multi-turn
  baselines.py         prompting, difference-in-means (± projection), LAPE, SAE features, logit bias, LoRA
  utility.py           ΔS over XNLI, Belebele, XL-Sum, FLORES, MMLU, GSM8K
  safety.py, judge.py  refusal / over-refusal and LLM-judge ratings
  stats.py             bootstrap intervals, sign-flip tests, Holm correction
  pipeline.py          one model–language experiment end to end
```

## Setup

```bash
pip install -r requirements.txt
hf auth login   # Llama 3.1, Gemma 2 and FLORES+ are gated
```

A single 80 GB GPU is comfortable for the 8–9B models. SAEs are loaded one layer at a time during Stages I–II and only
the window's SAEs stay resident afterwards.

## Usage

Models: `llama-3.1-8b`, `gemma-2-9b`, `gemma-2-2b`, `qwen3-8b`. Languages: `es fr zh ru hi ar`.

```bash
# Build the edit: Stages I–II on every layer, window search, then γ on D_dev
python run.py build --model gemma-2-2b --lang es

# Faster: reuse the paper's window (Table 4) and only run Stages I–II inside it
python run.py build --model gemma-2-2b --lang es --paper-window

# Tune every baseline under the same guardrails and evaluate all methods
python run.py evaluate --model gemma-2-2b --lang es --set no_cue
python run.py evaluate --model gemma-2-2b --lang es --set neutral
python run.py evaluate --model gemma-2-2b --lang es --set intent      # see full_english

# Matched-shift comparison, ablations / necessity, and the horizon test D2
python run.py sweep    --model gemma-2-2b --lang es
python run.py ablate   --model gemma-2-2b --lang es
python run.py validate --model gemma-2-2b --lang es
```

Add `--utility` to include the utility guardrail while tuning and to report ΔS per method (slow: six benchmarks per
setting). Add `--no-lora` to skip LoRA training.

Using a built edit from Python:

```python
from foxp2.pipeline import Experiment

exp = Experiment("gemma-2-2b", "es")
foxp2 = exp.load_edit()
with foxp2.method().active(exp.lm):
    print(exp.lm.generate([exp.lm.chat("What are the health benefits of green tea?")], 128).texts[0])
```

## Outputs

Everything is written to `outputs/<model>/<lang>/`:

| File | Contents |
|------|----------|
| `splits.json` | D_disc (300), D_dev (200), D_eval (300) and the FLORES utility sentences |
| `layer_profile.json` | per-layer rank, effective rank, stability, selectivity and lift |
| `neural_foxp2.pt` | the edit: window, per-layer vectors, support, γ, λ, β |
| `baselines.json` | each baseline's tuned setting |
| `results_<set>.json` | DEFAULT, full-response rate (95% CI), switch-away, line pass, leakage, KL₃, KL¹⁰_ref, tests |
| `<set>/<method>.jsonl` | every prompt and response |

## Prompts from the paper

All prompts are in `foxp2/prompts.py` and `prompts/`:

- **Short instruction** – `"{prompt} Answer in X."`; **system prompt** – `"Always respond in X."`
- **Best prompt** – the best of ten strategies on D_dev (Table 6): user-turn cue, bilingual cue, system prompt,
  system + instruction, native-language system message, language first, role framing, task cue, stated preference, and
  the P3 + P4 + P8 combination.
- **Cue levels L0–L5** for the score validation: no cue, a region, one target-language word, an English instruction,
  the instruction in the target language, and the whole prompt translated.
- **Multi-turn follow-ups** – "Can you explain that in a little more detail?" and
  "What is the most important point I should remember?"
- **LLM-judge rubrics** for fluency, adequacy, naturalness and language (Appendix L) and for refusals (Appendix K.3).

