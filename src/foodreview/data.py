"""Gerador de dataset SINTÉTICO de reviews (aspecto × sentimento) com split de distribuição.

- Splits in-distribution (train/val/test_id): vocabulário "ID", textos distintos entre si (split por texto, sem vazamento).
- Split test_ood: vocabulário e construções NÃO vistos no treino (gírias, negação, erros de digitação, sem acento).
O dataset serve para testar o pipeline e a robustez a mudança de distribuição; NÃO representa reviews reais.
O pipeline aceita CSV real com as colunas `text,aspect,sentiment` no lugar deste gerador.
"""
import itertools
import random
import re
import unicodedata

ASPECTS = ["entrega", "comida", "embalagem", "atendimento"]
SENTIMENTS = ["negativo", "neutro", "positivo"]

# (substantivo, gênero)
ID_NOUNS = {
    "entrega": [("a entrega", "f"), ("o entregador", "m"), ("o pedido", "m"), ("a chegada do pedido", "f")],
    "comida": [("a comida", "f"), ("o prato", "m"), ("o hambúrguer", "m"), ("a pizza", "f")],
    "embalagem": [("a embalagem", "f"), ("o pote", "m"), ("a sacola", "f"), ("a caixa", "f")],
    "atendimento": [("o atendimento", "m"), ("o suporte", "m"), ("a atendente", "f"), ("o chat", "m")],
}
ID_SPECIFIC = {
    "entrega": {"positivo": ["chegou rápido", "chegou antes do prazo", "veio bem rápido", "foi pontual"],
                "negativo": ["chegou atrasado", "demorou demais", "atrasou bastante", "chegou tarde"],
                "neutro": ["demorou o tempo médio", "chegou em tempo mediano"]},
    "comida": {"positivo": ["tinha ótimo sabor", "estava no ponto certo", "chegou bem quente", "tinha tempero perfeito"],
               "negativo": ["tinha sabor ruim", "estava passando do ponto", "chegou frio", "veio sem tempero"],
               "neutro": ["tinha sabor comum", "estava na média"]},
    "embalagem": {"positivo": ["veio bem lacrada", "protegeu bem o pedido", "chegou sem nenhum dano", "veio muito bem fechada"],
                  "negativo": ["veio aberta", "vazou tudo", "chegou com danos", "veio mal fechada"],
                  "neutro": ["veio normal", "estava na média"]},
    "atendimento": {"positivo": ["resolveu tudo rápido", "respondeu na hora", "tratou muito bem", "resolveu meu problema"],
                    "negativo": ["ignorou meu pedido", "demorou a responder", "tratou mal", "não resolveu nada"],
                    "neutro": ["respondeu no tempo médio", "atendeu de forma comum"]},
}
ID_ADJ = {"positivo": [("ótimo", "ótima"), ("excelente", "excelente"), ("maravilhoso", "maravilhosa"), ("perfeito", "perfeita")],
          "negativo": [("péssimo", "péssima"), ("horrível", "horrível"), ("ruim", "ruim"), ("terrível", "terrível")],
          "neutro": [("ok", "ok"), ("normal", "normal"), ("razoável", "razoável"), ("mediano", "mediana")]}
ID_OPENERS = ["", "Sinceramente, ", "No geral, "]
ID_END = [".", "!"]

OOD_NOUNS = {
    "entrega": [("a entrega", "f"), ("o motoboy", "m"), ("o cara da entrega", "m")],
    "comida": [("a marmita", "f"), ("o açaí", "m"), ("o lanche", "m"), ("a janta", "f")],
    "embalagem": [("a caixinha", "f"), ("a tampa", "f"), ("o saquinho", "m"), ("o lacre", "m")],
    "atendimento": [("o SAC", "m"), ("o pessoal do chat", "m"), ("a moça do suporte", "f"), ("o povo do restaurante", "m")],
}
OOD_SPECIFIC = {
    "entrega": {"positivo": ["voou de rápido", "não demorou nada", "nem vi o tempo passar", "chegou antes de eu terminar o filme"],
                "negativo": ["sumiu no caminho", "levou uma eternidade", "não chegou no prazo", "foi uma lesma"],
                "neutro": ["demorou o de sempre", "chegou mais ou menos na hora"]},
    "comida": {"positivo": ["estava de outro mundo", "não tinha nada de errado", "estava de babar", "estava divina"],
               "negativo": ["estava intragável", "não estava boa", "estava salgada demais", "era pura gordura"],
               "neutro": ["estava mais ou menos", "não é nada demais"]},
    "embalagem": {"positivo": ["não vazou nada", "não teve nenhum dano", "veio tudo certinho", "veio super seguro"],
                  "negativo": ["veio um desastre", "não fechou direito", "derramou tudo", "veio todo bagunçado"],
                  "neutro": ["veio no padrão", "é nada demais"]},
    "atendimento": {"positivo": ["não deixou a desejar", "me ajudou rapidinho", "foi gente boa", "salvou meu pedido"],
                    "negativo": ["me deixou no vácuo", "não ajudou em nada", "me tratou com descaso", "não deu a mínima"],
                    "neutro": ["foi tranquilo, nada demais", "respondeu e pronto"]},
}
OOD_GENERIC = {"positivo": ["foi show", "mandou bem", "foi top", "me conquistou"],
               "negativo": ["foi uma porcaria", "deixou a desejar", "foi decepcionante", "foi nota zero"],
               "neutro": ["foi mais ou menos", "foi normalzinho", "foi só ok", "ficou no meio termo"]}
OOD_OPENERS = ["", "Mano, ", "Olha, ", "Pô, ", "Sério, "]
OOD_END = ["", "!", "!!", "..."]
NEGATION = re.compile(r"\b(n[aã]o|nem|nada)\b", re.I)


def _adj(pair, gender):
    return pair[1] if gender == "f" else pair[0]


def _cap(s):
    return s[:1].upper() + s[1:]


def enumerate_id():
    rows = []
    for asp in ASPECTS:
        for sent in SENTIMENTS:
            preds = [("s", p) for p in ID_SPECIFIC[asp][sent]] + [("g", a) for a in ID_ADJ[sent]]
            for (noun, g), (kind, p), op, end in itertools.product(ID_NOUNS[asp], preds, ID_OPENERS, ID_END):
                body = f"{noun} {p}" if kind == "s" else f"{noun} foi {_adj(p, g)}"
                text = (op + body) if op else _cap(body)
                rows.append({"text": text + end, "aspect": asp, "sentiment": sent, "flags": ""})
    return rows


def _typo(word, rng):
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    return rng.choice([word[:i] + word[i + 1:], word[:i] + word[i + 1] + word[i] + word[i + 2:]]) if i + 2 <= len(word) else word


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def sample_ood(n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        asp, sent = ASPECTS[i % 4], SENTIMENTS[(i // 4) % 3]
        noun, g = rng.choice(OOD_NOUNS[asp])
        pool = [("s", p) for p in OOD_SPECIFIC[asp][sent]] * 2 + [("g", p) for p in OOD_GENERIC[sent]]
        kind, p = rng.choice(pool)
        text = rng.choice(OOD_OPENERS) + f"{noun} {p}" + rng.choice(OOD_END)
        flags = []
        if rng.random() < 0.5:
            text = text.lower(); flags.append("lower")
        if rng.random() < 0.4:
            text = _strip_accents(text); flags.append("noaccent")
        if rng.random() < 0.35:
            ws = text.split(" ")
            j = rng.randrange(len(ws)); ws[j] = _typo(ws[j], rng); text = " ".join(ws); flags.append("typo")
        if NEGATION.search(text):
            flags.append("negation")
        rows.append({"text": text, "aspect": asp, "sentiment": sent, "flags": "|".join(flags)})
    return rows


def build(seed=13, n_ood=900):
    ids = enumerate_id()
    seen, uniq = set(), []
    for r in ids:
        if r["text"] not in seen:
            seen.add(r["text"]); uniq.append(r)
    random.Random(seed).shuffle(uniq)
    n = len(uniq)
    train, val, test_id = uniq[: int(.7 * n)], uniq[int(.7 * n): int(.85 * n)], uniq[int(.85 * n):]
    ood = sample_ood(n_ood, seed + 1)
    train_texts = {r["text"] for r in train}
    ood = [r for r in ood if r["text"] not in train_texts]
    return {"train": train, "val": val, "test_id": test_id, "test_ood": ood}
