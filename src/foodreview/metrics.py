import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def bootstrap_ci(y, p, fn=macro_f1, n=1000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece(y, proba, classes, bins=10) -> float:
    """Expected Calibration Error (confiança máxima vs. acurácia, bins uniformes)."""
    proba, y = np.asarray(proba), np.asarray(y)
    conf, pred = proba.max(1), np.asarray(classes)[proba.argmax(1)]
    correct = (pred == y).astype(float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
