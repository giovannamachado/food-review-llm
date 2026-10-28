"""Prompt e parser para o baseline zero-shot com LLM via API (usado por scripts/eval_zeroshot_llm.py)."""
import json
import re

from .data import ASPECTS, SENTIMENTS

PROMPT = (
    "Classifique a review de um app de delivery.\n"
    f"aspect: um de {ASPECTS}\nsentiment: um de {SENTIMENTS}\n"
    'Responda SOMENTE com JSON no formato {{"aspect": "...", "sentiment": "..."}}.\n\nReview: {text}'
)


def parse_labels(raw: str) -> tuple[str | None, str | None]:
    """Extrai (aspect, sentiment) da resposta do LLM; devolve None no campo ausente ou fora do vocabulário."""
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    a, s = str(d.get("aspect", "")).strip().lower(), str(d.get("sentiment", "")).strip().lower()
    return (a if a in ASPECTS else None), (s if s in SENTIMENTS else None)
