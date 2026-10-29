"""Publica modelo (baseline sklearn + cards) e dataset no Hugging Face Hub. NÃO executado nesta entrega (requer HF_TOKEN).

Repositórios são criados PRIVADOS por padrão; use --public só depois de revisar os cards.
Uso: HF_TOKEN=... python scripts/push_to_hub.py --user <usuario> [--public]
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def stage(files: dict[str, Path]) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, src in files.items():
        shutil.copy(src, d / name)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    model_id, data_id = f"{a.user}/food-review-baselines", f"{a.user}/food-review-synthetic-ptbr"
    api.create_repo(model_id, repo_type="model", private=not a.public, exist_ok=True)
    api.create_repo(data_id, repo_type="dataset", private=not a.public, exist_ok=True)
    # create_repo(exist_ok=True) não altera a visibilidade de repositórios já existentes
    api.update_repo_settings(repo_id=model_id, repo_type="model", private=not a.public)
    api.update_repo_settings(repo_id=data_id, repo_type="dataset", private=not a.public)
    api.upload_folder(repo_id=model_id, repo_type="model", folder_path=stage({
        "README.md": ROOT / "docs" / "MODEL_CARD.md", "model_aspect.joblib": ROOT / "artifacts" / "model_aspect.joblib",
        "model_sentiment.joblib": ROOT / "artifacts" / "model_sentiment.joblib", "metrics.json": ROOT / "artifacts" / "metrics.json"}))
    api.upload_folder(repo_id=data_id, repo_type="dataset", folder_path=stage({
        "README.md": ROOT / "docs" / "DATA_CARD.md", **{f"{s}.csv": ROOT / "data" / f"{s}.csv" for s in ("train", "val", "test_id", "test_ood")}}))
    print("ok:", model_id, data_id)


if __name__ == "__main__":
    main()
