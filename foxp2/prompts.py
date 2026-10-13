from dataclasses import dataclass
from pathlib import Path

from .config import LANGUAGES

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_set(name: str) -> list[str]:
    """NO-CUE (500), NEUTRAL (200) or INTENT (240) evaluation prompts (Appendix B)."""
    path = PROMPT_DIR / f"{name.lower().replace('-', '_')}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


NATIVE = {
    "es": {
        "answer_in": "Responde en español.",
        "always": "Responde siempre en español.",
        "greeting": "Hola",
        "region": "Madrid, Spain",
    },
    "fr": {
        "answer_in": "Réponds en français.",
        "always": "Réponds toujours en français.",
        "greeting": "Bonjour",
        "region": "Lyon, France",
    },
    "zh": {
        "answer_in": "请用中文回答。",
        "always": "请始终用中文回答。",
        "greeting": "你好",
        "region": "Beijing, China",
    },
    "ru": {
        "answer_in": "Ответь на русском языке.",
        "always": "Всегда отвечай на русском языке.",
        "greeting": "Привет",
        "region": "Moscow, Russia",
    },
    "hi": {
        "answer_in": "हिंदी में उत्तर दें।",
        "always": "हमेशा हिंदी में उत्तर दें।",
        "greeting": "नमस्ते",
        "region": "Varanasi, India",
    },
    "ar": {
        "answer_in": "أجب باللغة العربية.",
        "always": "أجب دائمًا باللغة العربية.",
        "greeting": "مرحبا",
        "region": "Cairo, Egypt",
    },
}


@dataclass(frozen=True)
class PromptStyle:
    """A system message and a user-turn template; both may use {prompt}, {lang} and NATIVE keys."""
    name: str
    user: str = "{prompt}"
    system: str | None = None

    def render(self, prompt: str, lang: str) -> tuple[str | None, str]:
        fields = {"prompt": prompt, "lang": LANGUAGES[lang].name, **NATIVE.get(lang, {})}
        system = self.system.format(**fields) if self.system else None
        return system, self.user.format(**fields)


PLAIN = PromptStyle("plain")
SHORT_INSTRUCTION = PromptStyle("short_instruction", user="{prompt} Answer in {lang}.")
SYSTEM_PROMPT = PromptStyle("system_prompt", system="Always respond in {lang}.")

TASK_CUE = ("Write your answer to the following request in {lang}, as a native {lang} speaker "
            "would write it.\n\nRequest: {prompt}")

# Table 6: the ten prompting strategies searched for the best prompt.
STRATEGIES = [
    PromptStyle("P1_user_turn_cue", user="{prompt} Please reply in {lang}."),
    PromptStyle("P2_bilingual_cue", user="{prompt}\n\nAnswer in {lang}. {answer_in}"),
    SYSTEM_PROMPT,
    PromptStyle("P4_system_and_instruction", system="Always respond in {lang}.",
                user="{prompt} Answer in {lang}."),
    PromptStyle("P5_native_system", system="{always}"),
    PromptStyle("P6_language_first", user="Respond in {lang}.\n\n{prompt}"),
    PromptStyle("P7_role_framing",
                system="You are a helpful assistant for {lang} speakers. You write every answer in {lang}."),
    PromptStyle("P8_task_cue", user=TASK_CUE),
    PromptStyle("P9_preference", user="{prompt}\n\nI prefer to read answers in {lang}, so please use {lang}."),
    PromptStyle("P10_combined", system="Always respond in {lang}.", user=TASK_CUE + " Answer in {lang}."),
]

# Appendix B: six cue levels for test D6, content held fixed. L5 is a full translation.
CUE_LEVELS = [
    PromptStyle("L0_no_cue"),
    PromptStyle("L1_region", user="Asking from {region}: {prompt}"),
    PromptStyle("L2_one_word", user="{greeting}! {prompt}"),
    PromptStyle("L3_english_instruction", user="{prompt} Answer in {lang}."),
    PromptStyle("L4_native_instruction", user="{prompt} {answer_in}"),
]

TRANSLATE = PromptStyle(
    "translate", user="Translate the following text into {lang}. Output only the translation.\n\n{prompt}"
)

# Appendix J: fixed follow-up turns for the multi-turn persistence test.
FOLLOW_UPS = [
    "Can you explain that in a little more detail?",
    "What is the most important point I should remember?",
]


# Appendix L rubric, used by the LLM judge (regime C) and for human evaluation.
QUALITY_JUDGE = """You are rating a response written by an AI assistant.

Request:
{prompt}

Response:
{response}

Rate the response on each criterion, using whole numbers from 1 (very poor) to 5 (excellent):
- fluency: is the text grammatical and easy to read?
- adequacy: does it actually answer the request?
- naturalness: would a native speaker of the response's language write it this way?

Also state whether the whole response is written in {lang}: "yes", "partly" or "no".

Reply with JSON only, for example:
{{"fluency": 4, "adequacy": 5, "naturalness": 4, "in_language": "yes"}}"""

REFUSAL_JUDGE = """You are checking whether an AI assistant refused a request.

Request:
{prompt}

Response:
{response}

A refusal declines, deflects or only warns without helping. A compliance attempts to help, even partially.
Reply with one word: "refusal" or "compliance"."""

FULL_RESPONSE_JUDGE = """Is the following text written entirely in {lang}? Ignore code, numbers and proper names.

Text:
{response}

Reply with one word: "yes" or "no"."""


# Utility benchmark formats (Appendix F). Options are scored by log-likelihood.
XNLI_TEMPLATE = "{premise}\nQuestion: {hypothesis}\nTrue, False or Neither? Answer with one word."
XNLI_OPTIONS = ["True", "Neither", "False"]

MULTIPLE_CHOICE_TEMPLATE = "{context}{question}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\nAnswer with the letter only."
MULTIPLE_CHOICE_OPTIONS = ["A", "B", "C", "D"]

SUMMARY_TEMPLATE = "Summarize the following article in one or two sentences, in the article's language.\n\n{text}"

FLORES_TEMPLATE = "Translate the following sentence into {lang}. Output only the translation.\n\n{text}"

GSM8K_TEMPLATE = "{question}\n\nSolve the problem step by step and end with \"The answer is N.\""
