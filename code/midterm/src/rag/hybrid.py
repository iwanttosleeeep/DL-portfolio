from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Document
from .fields import DocFields, extract_query_signals, field_match_score
from .tfidf_baseline import TFIDFRetriever

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/'][a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    tokens = []
    for match in TOKEN_RE.finditer(text.casefold()):
        token = match.group(0)
        if token.isalpha() and token in ENGLISH_STOP_WORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


class BM25Retriever:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document.text) for document in documents]
        self.doc_lengths = np.array([len(tokens) for tokens in self.doc_tokens], dtype=float)
        self.avg_length = float(self.doc_lengths.mean()) if len(documents) else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        document_frequency = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))
        n_docs = len(documents)
        self.idf = {
            term: math.log(1.0 + (n_docs - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score_all(self, query: str) -> np.ndarray:
        query_terms = Counter(tokenize(query))
        scores = np.zeros(len(self.documents), dtype=float)
        for index, frequencies in enumerate(self.term_frequencies):
            length_norm = self.k1 * (
                1.0 - self.b + self.b * self.doc_lengths[index] / (self.avg_length or 1.0)
            )
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                score += (
                    self.idf.get(term, 0.0)
                    * (term_frequency * (self.k1 + 1.0))
                    / (term_frequency + length_norm)
                    * query_frequency
                )
            scores[index] = score
        return scores


def _normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max()) if values.size else 0.0
    if maximum <= 0:
        return np.zeros_like(values)
    return values / maximum


class HybridRetriever:
    def __init__(
        self,
        documents: list[Document],
        fields: list[DocFields],
        baseline: TFIDFRetriever,
        weights: dict[str, float] | None = None,
    ):
        self.documents = documents
        self.fields = fields
        self.baseline = baseline
        self.weights = weights or {"word": 0.45, "char": 0.40, "bm25": 0.15}
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
        )
        self.char_matrix = self.char_vectorizer.fit_transform(document.text for document in documents)
        self.bm25 = BM25Retriever(documents)

    def score_all(self, question: str) -> np.ndarray:
        word_scores = cosine_similarity(
            self.baseline.vectorizer.transform([question]), self.baseline.matrix
        ).ravel()
        char_scores = cosine_similarity(
            self.char_vectorizer.transform([question]), self.char_matrix
        ).ravel()
        bm25_scores = self.bm25.score_all(question)

        word_norm = _normalize(word_scores)
        char_norm = _normalize(char_scores)
        bm25_norm = _normalize(bm25_scores)
        fused = (
            self.weights["word"] * word_norm
            + self.weights["char"] * char_norm
            + self.weights["bm25"] * bm25_norm
        )

        signals = extract_query_signals(question, self.fields)
        field_scores = np.array(
            [field_match_score(question, fields, signals) for fields in self.fields],
            dtype=float,
        )
        return fused + field_scores

    def rank_all(self, question: str) -> list[dict]:
        scores = self.score_all(question)
        order = np.argsort(-scores, kind="stable")
        results = []
        for rank, index in enumerate(order, start=1):
            results.append(
                {
                    "filename": self.documents[index].filename,
                    "doc_type": self.documents[index].doc_type,
                    "score": float(scores[index]),
                    "rank": rank,
                }
            )
        return results
