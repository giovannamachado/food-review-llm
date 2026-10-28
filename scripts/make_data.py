import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodreview.data import build  # noqa: E402

splits = build()
for name, rows in splits.items():
    with open(ROOT / "data" / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "aspect", "sentiment", "flags"])
        w.writeheader(); w.writerows(rows)
    print(name, len(rows), dict(Counter(r["sentiment"] for r in rows)))
