import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Document


class TFIDFRetriever:
    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        stop_words: str | None = "english",
    ):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words,
            min_df=1,
        )
        self.documents: list[Document] = []
        self.matrix = None

    def fit(self, documents: list[Document]) -> "TFIDFRetriever":
        self.documents = list(documents)
        self.matrix = self.vectorizer.fit_transform(d.text for d in documents)
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.matrix is None:
            raise RuntimeError("retriever is not fitted")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(self.documents))]
        return [
            {
                "filename": self.documents[i].filename,
                "doc_type": self.documents[i].doc_type,
                "score": float(scores[i]),
                "rank": rank,
            }
            for rank, i in enumerate(order, start=1)
        ]

    def rank_all(self, query: str) -> list[dict]:
        return self.search(query, top_k=len(self.documents))
