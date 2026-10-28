"""Baseline zero-shot com LLM via API no split test_ood — NÃO executado nesta entrega (requer ANTHROPIC_API_KEY e custo).

Reporta macro-F1 e custo REAL a partir de `usage`. Os preços são argumentos obrigatórios (USD por 1M tokens): consulte a tabela
vigente do provedor, não há valor padrão para evitar números desatualizados.
Uso: python scripts/eval_zeroshot_llm.py --model <id> --price-in <x> --price-out <y> [--n 300]
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodreview.llm_eval import PROMPT, parse_labels  # noqa: E402
from foodreview.metrics import bootstrap_ci, macro_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--price-in", type=float, required=True)
    ap.add_argument("--price-out", type=float, required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "data" / "test_ood.csv", encoding="utf-8")))[: a.n]
    key = os.environ["ANTHROPIC_API_KEY"]
    preds, tin, tout, lat = {"aspect": [], "sentiment": []}, 0, 0, []
    with httpx.Client(timeout=60) as c:
        for r in rows:
            t = time.perf_counter()
            resp = c.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                          json={"model": a.model, "max_tokens": 60, "messages": [{"role": "user", "content": PROMPT.format(text=r["text"])}]})
            resp.raise_for_status()
            lat.append((time.perf_counter() - t) * 1000)
            d = resp.json()
            tin += d["usage"]["input_tokens"]; tout += d["usage"]["output_tokens"]
            asp, sent = parse_labels("".join(b.get("text", "") for b in d["content"] if b.get("type") == "text"))
            preds["aspect"].append(asp or "invalid"); preds["sentiment"].append(sent or "invalid")
    out = {"model": a.model, "n": len(rows), "latency_ms_p50": sorted(lat)[len(lat) // 2],
           "cost_usd_per_1k": (tin * a.price_in + tout * a.price_out) / 1e6 / len(rows) * 1000}
    for t in ("aspect", "sentiment"):
        y = [r[t] for r in rows]
        lo, hi = bootstrap_ci(y, preds[t])
        out[t] = {"macro_f1": round(macro_f1(y, preds[t]), 4), "ci95": [round(lo, 4), round(hi, 4)], "invalid": preds[t].count("invalid")}
    (ROOT / "artifacts" / "zeroshot_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
