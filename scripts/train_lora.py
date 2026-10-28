"""Fine-tuning LoRA de um encoder PT-BR para classificação (aspecto OU sentimento).

STATUS: NÃO EXECUTADO neste repositório — o ambiente de desenvolvimento não tem GPU nem torch. `--dry-run` valida dados/config sem torch.
Esperado: rodar em Colab/GPU (~minutos), gerar artifacts/lora_<task>/{adapter,metrics.json} com o mesmo esquema dos baselines
e então comparar com docs/RESULTS.md (o ponto do experimento é o split test_ood).
Instalação: pip install -e ".[lora]"
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"aspect": ["entrega", "comida", "embalagem", "atendimento"], "sentiment": ["negativo", "neutro", "positivo"]}


def read(split):
    with open(ROOT / "data" / f"{split}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(LABELS), required=True)
    ap.add_argument("--base-model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    label2id = {lab: i for i, lab in enumerate(LABELS[a.task])}
    data = {s: read(s) for s in ("train", "val", "test_id", "test_ood")}
    if a.dry_run:
        print(json.dumps({"args": vars(a), "labels": label2id, "sizes": {k: len(v) for k, v in data.items()}}, indent=2))
        return

    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(a.base_model, num_labels=len(label2id), id2label={i: lab for lab, i in label2id.items()}, label2id=label2id)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=["query", "value"]))
    model.print_trainable_parameters()

    def enc(rows):
        ds = Dataset.from_dict({"text": [r["text"] for r in rows], "label": [label2id[r[a.task]] for r in rows]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len, padding="max_length"), batched=True)

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"), "accuracy": float((pred == p.label_ids).mean())}

    out = ROOT / "artifacts" / f"lora_{a.task}"
    args = TrainingArguments(output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=a.batch,
                             per_device_eval_batch_size=64, eval_strategy="epoch", save_strategy="no", seed=a.seed, fp16=torch.cuda.is_available(),
                             report_to="none", logging_steps=20)
    tr = Trainer(model=model, args=args, train_dataset=enc(data["train"]), eval_dataset=enc(data["val"]), compute_metrics=metrics)
    tr.train()
    res = {}
    for sp in ("val", "test_id", "test_ood"):
        m = tr.evaluate(enc(data[sp]), metric_key_prefix=sp)
        res[sp] = {"macro_f1": round(float(m[f"{sp}_macro_f1"]), 4), "n": len(data[sp])}
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    (out / "metrics.json").write_text(json.dumps({"base_model": a.base_model, "rank": a.rank, "seed": a.seed, **res}, indent=2))
    print(json.dumps(res, indent=2))
    _ = np  # noqa: F841


if __name__ == "__main__":
    main()
