---
language: [pt]
license: mit
task_categories: [text-classification]
pretty_name: Food review aspect & sentiment (synthetic, PT-BR)
size_categories: [1K<n<10K]
---
# Dataset card — food-review-synthetic-ptbr

**100% sintético.** Gerado por gramática (`src/foodreview/data.py`, seed 13): aspecto (entrega, comida, embalagem, atendimento) × sentimento (negativo, neutro, positivo). Não contém reviews reais nem dados pessoais.

| split | n | papel |
|---|---|---|
| train | 1478 | treino |
| val | 317 | seleção/diagnóstico |
| test_id | 317 | mesma gramática do treino, textos distintos |
| test_ood | 900 (balanceado) | vocabulário e construções **ausentes do treino**: gírias, negação ("não demorou nada"), erros de digitação, sem acentos |

Colunas: `text`, `aspect`, `sentiment`, `flags` (`lower|noaccent|typo|negation`, só relevante no OOD).

**Garantias testadas** (`tests/test_all.py`): nenhum texto repetido entre splits; palavras-chave do OOD ausentes do treino; geração determinística.
**Limitações.** Frases de um único aspecto, curtas e de gramática simples; não captura ironia, múltiplos aspectos por review ou ruído real. Os scores neste dataset **não** estimam desempenho em reviews reais. O pipeline aceita CSV real com as mesmas colunas (substituir `data/*.csv`).
**Viés.** O vocabulário reflete a escolha do autor da gramática (português brasileiro urbano informal).
