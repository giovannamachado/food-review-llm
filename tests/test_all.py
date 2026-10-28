import csv
import json
from pathlib import Path

import numpy as np

from foodreview.baselines import make
from foodreview.data import ASPECTS, SENTIMENTS, build
from foodreview.llm_eval import parse_labels
from foodreview.metrics import bootstrap_ci, ece, macro_f1

ROOT = Path(__file__).resolve().parents[1]


def test_build_is_deterministic_and_labels_valid():
    a, b = build(), build()
    assert a == b
    for rows in a.values():
        assert all(r["aspect"] in ASPECTS and r["sentiment"] in SENTIMENTS for r in rows)


def test_no_text_leakage_between_train_and_eval_splits():
    s = build()
    train = {r["text"] for r in s["train"]}
    for name in ("val", "test_id", "test_ood"):
        assert not train & {r["text"] for r in s[name]}, name
    assert not {r["text"] for r in s["val"]} & {r["text"] for r in s["test_id"]}


def test_ood_uses_vocabulary_unseen_in_training():
    s = build()
    train_words = {w.strip(".,!?").lower() for r in s["train"] for w in r["text"].split()}
    for w in ("motoboy", "marmita", "açaí", "saquinho", "sac", "lesma", "intragável"):
        assert w not in train_words
    assert any("negation" in r["flags"] for r in s["test_ood"]) and any("typo" in r["flags"] for r in s["test_ood"])


def test_committed_csvs_match_generator():
    s = build()
    with open(ROOT / "data" / "test_ood.csv", encoding="utf-8") as f:
        assert [r["text"] for r in csv.DictReader(f)] == [r["text"] for r in s["test_ood"]]


def test_metrics():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert abs(ece(np.array(["a", "b", "b"]), proba, ["a", "b"]) - 1 / 3) < 1e-9  # confiança 1.0, acurácia 2/3
    y = np.array(["a", "b"] * 30)
    p = y.copy(); p[:6] = "b"
    lo, hi = bootstrap_ci(y, p, n=200)
    assert lo <= macro_f1(y, p) <= hi


def test_pipeline_in_dist_is_easy_and_ood_is_hard():
    s = build()
    xtr, ytr = [r["text"] for r in s["train"]], [r["sentiment"] for r in s["train"]]
    m = make("tfidf_word_lr").fit(xtr, ytr)
    f1 = lambda sp: macro_f1([r["sentiment"] for r in s[sp]], m.predict([r["text"] for r in s[sp]]))  # noqa: E731
    assert f1("test_id") > 0.95 and f1("test_ood") < f1("test_id") - 0.3  # guarda do desenho do experimento: ID não discrimina, OOD sim
    assert m.predict_proba(["a entrega foi péssima"]).shape == (1, 3)


def test_saved_metrics_are_consistent_with_results_doc():
    m = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    md = (ROOT / "docs" / "RESULTS.md").read_text()
    v = m["sentiment/tfidf_word+char_lr"]["test_ood"]["macro_f1"]
    assert f"{v:.3f}" in md


def test_llm_parser():
    assert parse_labels('{"aspect": "Comida", "sentiment": "positivo"}') == ("comida", "positivo")
    assert parse_labels('```json\n{"aspect": "entrega", "sentiment": "negativo"}\n```') == ("entrega", "negativo")
    assert parse_labels('{"aspect": "preço", "sentiment": "positivo"}') == (None, "positivo")
    assert parse_labels("não sei") == (None, None)
    assert parse_labels("{quebrado") == (None, None)
