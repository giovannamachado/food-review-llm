import time

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def make(name: str, seed: int = 0):
    word = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents=None, lowercase=True)  # noqa: E731
    char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True)  # noqa: E731
    lr = lambda: LogisticRegression(max_iter=2000, C=10.0, random_state=seed)  # noqa: E731
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "tfidf_word_lr":
        return Pipeline([("v", word()), ("clf", lr())])
    if name == "tfidf_char_lr":
        return Pipeline([("v", char()), ("clf", lr())])
    if name == "tfidf_word+char_lr":
        return Pipeline([("v", FeatureUnion([("w", word()), ("c", char())])), ("clf", lr())])
    raise ValueError(name)


NAMES = ["majority", "tfidf_word_lr", "tfidf_char_lr", "tfidf_word+char_lr"]


def timed_predict(model, texts):
    t = time.perf_counter()
    p = model.predict(texts)
    return p, (time.perf_counter() - t) / max(1, len(texts)) * 1000 * 1000  # ms por 1k textos
