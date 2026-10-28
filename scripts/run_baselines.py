"""Treina e avalia os baselines para as duas tarefas (aspecto, sentimento) em val / test_id / test_ood.

Saída: artifacts/metrics.json, artifacts/errors_ood.csv e docs/RESULTS.md. Modelo final -> artifacts/model_*.joblib.
Métricas: macro-F1 com IC95% (bootstrap 1000x), ECE, F1 por fatia (negação, typo) e latência de inferência em CPU.
"""
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodreview.baselines import NAMES, make, timed_predict  # noqa: E402
from foodreview.metrics import bootstrap_ci, ece, macro_f1  # noqa: E402

TASKS = ["aspect", "sentiment"]
SPLITS = ["val", "test_id", "test_ood"]


def load(name):
    with open(ROOT / "data" / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    data = {s: load(s) for s in ["train"] + SPLITS}
    out, errors = {}, []
    for task in TASKS:
        ytr = [r[task] for r in data["train"]]
        xtr = [r["text"] for r in data["train"]]
        for name in NAMES:
            m = make(name).fit(xtr, ytr)
            res = {}
            for sp in SPLITS:
                x, y = [r["text"] for r in data[sp]], np.array([r[task] for r in data[sp]])
                pred, ms1k = timed_predict(m, x)
                lo, hi = bootstrap_ci(y, pred)
                res[sp] = {"macro_f1": round(macro_f1(y, pred), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(y), "cpu_ms_per_1k": round(ms1k, 1)}
                if hasattr(m, "predict_proba"):
                    res[sp]["ece"] = round(ece(y, m.predict_proba(x), m.classes_), 4)
                if sp == "test_ood":
                    flags = [r["flags"] for r in data[sp]]
                    for sl, cond in {"negation": lambda f: "negation" in f, "typo": lambda f: "typo" in f, "clean": lambda f: not any(k in f for k in ("typo", "negation"))}.items():
                        idx = [i for i, f in enumerate(flags) if cond(f)]
                        res[sp][f"slice_{sl}"] = {"macro_f1": round(macro_f1(y[idx], pred[idx]), 4), "n": len(idx)}
                    if name == "tfidf_word+char_lr":
                        errors += [{"task": task, "text": data[sp][i]["text"], "gold": y[i], "pred": pred[i], "flags": flags[i]} for i in range(len(y)) if pred[i] != y[i]]
            out[f"{task}/{name}"] = res
            if name == "tfidf_word+char_lr":
                joblib.dump(m, ROOT / "artifacts" / f"model_{task}.joblib")
    (ROOT / "artifacts" / "metrics.json").write_text(json.dumps(out, indent=2))
    with open(ROOT / "artifacts" / "errors_ood.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "text", "gold", "pred", "flags"]); w.writeheader(); w.writerows(errors)
    write_md(out)


def write_md(out):
    L = ["# Resultados dos baselines", "", "_Gerado por `scripts/run_baselines.py`. Dados **sintéticos** (ver `docs/DATA_CARD.md`). macro-F1 [IC95% bootstrap]._", ""]
    for task in TASKS:
        L += [f"## Tarefa: {task}", "", "| modelo | val | test_id | test_ood | ECE ood | ood: negação | ood: typo | ood: limpo | CPU ms/1k textos |", "|---|---|---|---|---|---|---|---|---|"]
        for name in NAMES:
            r = out[f"{task}/{name}"]
            f = lambda sp: f"{r[sp]['macro_f1']:.3f} [{r[sp]['ci95'][0]:.2f}, {r[sp]['ci95'][1]:.2f}]"  # noqa: E731
            o = r["test_ood"]
            L.append(f"| {name} | {f('val')} | {f('test_id')} | {f('test_ood')} | {o.get('ece', '-')} | {o['slice_negation']['macro_f1']:.2f} (n={o['slice_negation']['n']}) | "
                     f"{o['slice_typo']['macro_f1']:.2f} (n={o['slice_typo']['n']}) | {o['slice_clean']['macro_f1']:.2f} (n={o['slice_clean']['n']}) | {o['cpu_ms_per_1k']} |")
        L.append("")
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
