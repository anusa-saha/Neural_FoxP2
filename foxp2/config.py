from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    flores: str
    glotlid: tuple
    belebele: str
    xlsum: str


LANGUAGES = {
    "en": Language("en", "English", "eng_Latn", ("eng_Latn",), "eng_Latn", "english"),
    "es": Language("es", "Spanish", "spa_Latn", ("spa_Latn",), "spa_Latn", "spanish"),
    "fr": Language("fr", "French", "fra_Latn", ("fra_Latn",), "fra_Latn", "french"),
    "zh": Language("zh", "Chinese", "cmn_Hans", ("cmn_Hani", "cmn_Hans", "zho_Hani", "zho_Hans"),
                   "zho_Hans", "chinese_simplified"),
    "ru": Language("ru", "Russian", "rus_Cyrl", ("rus_Cyrl",), "rus_Cyrl", "russian"),
    "hi": Language("hi", "Hindi", "hin_Deva", ("hin_Deva",), "hin_Deva", "hindi"),
    "ar": Language("ar", "Arabic", "arb_Arab", ("arb_Arab", "ara_Arab"), "arb_Arab", "arabic"),
}

TARGETS = ("es", "fr", "zh", "ru", "hi", "ar")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    sae_release: str
    sae_id: str
    n_layers: int


MODELS = {
    "llama-3.1-8b": ModelSpec(
        "meta-llama/Llama-3.1-8B-Instruct", "llama_scope_lxr_8x", "l{layer}r_8x", 32
    ),
    "gemma-2-9b": ModelSpec(
        "google/gemma-2-9b-it", "gemma-scope-9b-pt-res-canonical", "layer_{layer}/width_16k/canonical", 42
    ),
    "gemma-2-2b": ModelSpec(
        "google/gemma-2-2b-it", "gemma-scope-2b-pt-res-canonical", "layer_{layer}/width_16k/canonical", 26
    ),
    "qwen3-8b": ModelSpec(
        "Qwen/Qwen3-8B", "qwen-scope-3-8b-base-w64k-l50", "layer{layer}", 36
    ),
}


@dataclass
class Hyperparameters:
    # Data (Appendix B)
    n_disc: int = 300
    n_dev: int = 200
    n_eval: int = 300
    n_flores_utility: int = 200
    tau_sem: float = 0.85
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    # Defaultness score (Section 3.2, Appendix F)
    horizon: tuple = (1, 2, 3)
    rho: float = 5.0
    min_frequency: float = 2e-6
    lid_tokens: int = 20
    min_sentence_chars: int = 20
    kl_ref_steps: int = 10

    # Stage I
    k_features: int = 64
    lift_multipliers: tuple = (0.5, 1.0, 2.0)
    lift_candidates: int = 256
    n_lift_prompts: int = 32

    # Stage II
    max_rank: int = 16
    n_bootstrap: int = 200
    tau_stab: float = 0.70
    window_min: float = 0.15
    window_max: float = 0.35
    n_search_prompts: int = 64
    alpha_leak: float = 1.0
    alpha_kl: float = 0.5
    alpha_util: float = 0.05

    # Stage III
    lam: float = 1.0
    beta_over_lambda: float = 1.0
    gamma_grid: tuple = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    probe_gamma: float = 1.0

    # Guardrails (Table 3)
    eps_leak: float = 0.060
    eps_kl: float = 0.250
    eps_util: float = 1.0

    # Generation and statistics
    max_new_tokens: int = 256
    batch_size: int = 16
    n_boot: int = 2000
    n_perm: int = 10000
    seeds: tuple = (17, 23, 41)
    eps: float = 1e-6

    @property
    def beta(self) -> float:
        return self.lam * self.beta_over_lambda

    @property
    def seed(self) -> int:
        return self.seeds[0]


# Table 4 of the paper: selected window (inclusive), rank r and scale gamma per pair.
PAPER_ARTIFACTS = {
    "llama-3.1-8b": {
        "es": ((10, 19), 7, 1.0), "fr": ((9, 18), 6, 0.75), "zh": ((11, 21), 8, 1.5),
        "ru": ((10, 20), 7, 1.0), "hi": ((12, 21), 6, 0.5), "ar": ((8, 17), 6, 0.75),
    },
    "gemma-2-9b": {
        "es": ((13, 25), 6, 0.75), "fr": ((12, 24), 6, 0.75), "zh": ((14, 27), 7, 1.5),
        "ru": ((12, 25), 6, 1.0), "hi": ((15, 28), 5, 0.5), "ar": ((11, 23), 5, 0.75),
    },
    "gemma-2-2b": {
        "es": ((7, 15), 6, 0.75), "fr": ((7, 14), 5, 0.75), "zh": ((8, 16), 7, 1.5),
        "ru": ((7, 15), 6, 1.0), "hi": ((9, 16), 5, 0.25), "ar": ((6, 14), 5, 0.5),
    },
    "qwen3-8b": {
        "es": ((11, 21), 5, 0.75), "fr": ((10, 20), 5, 0.75), "zh": ((13, 24), 6, 1.5),
        "ru": ((11, 21), 5, 1.0), "hi": ((14, 23), 4, 0.25), "ar": ((10, 20), 4, 0.5),
    },
}


@dataclass
class Paths:
    root: str = "outputs"
    prompts: str = "prompts"

    def run_dir(self, model: str, lang: str) -> str:
        return f"{self.root}/{model}/{lang}"
