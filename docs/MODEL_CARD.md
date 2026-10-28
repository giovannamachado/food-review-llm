---
language: [pt]
license: mit
library_name: scikit-learn
pipeline_tag: text-classification
tags: [baseline, sentiment, aspect, pt-br]
---
# Model card — food-review-baselines

Dois classificadores scikit-learn (aspecto e sentimento): TF-IDF palavra(1–2) + char_wb(2–5) → regressão logística. **São baselines**: o objetivo é fixar o piso que o fine-tuning LoRA (`scripts/train_lora.py`, ainda não executado) precisa superar.

**Dados:** [food-review-synthetic-ptbr](DATA_CARD.md) — sintético. **Uso pretendido:** demonstração/benchmark de pipeline. **Fora de escopo:** qualquer decisão sobre reviews reais.

## Desempenho (macro-F1, IC95% bootstrap em `docs/RESULTS.md`)
| tarefa | test_id | test_ood |
|---|---|---|
| aspecto | 1.000 | **0.662** |
| sentimento | 1.000 | **0.413** (ECE 0.29: mal calibrado sob mudança de distribuição) |

O score in-distribution é 1.0 porque a gramática é compartilhada com o treino: **não discrimina modelos**. O split OOD é a medida relevante, e mostra que os modelos de bag-of-words memorizam vocabulário (colapso em substantivos/construções novos, pior ainda em negação: 0.25 F1 em sentimento).
**Latência:** ~56 ms por 1000 textos em 1 CPU (inferência, sem batching especial); artefatos de ~90 KB.
**Reprodução:** `python scripts/make_data.py && python scripts/run_baselines.py`.
