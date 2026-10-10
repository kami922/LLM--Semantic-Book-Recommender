from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {"title", "description", "category", "emotion"}


@dataclass
class Recommendation:
    title: str
    description: str
    category: str
    emotion: str
    score: float


class SemanticBookRecommender:
    def __init__(self, books: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(books.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self._books = books.fillna("").copy()
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._book_vectors = self._vectorizer.fit_transform(self._books["description"])

    @classmethod
    def from_csv(cls, csv_path: str) -> "SemanticBookRecommender":
        return cls(pd.read_csv(csv_path))

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        emotion: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Recommendation]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._book_vectors).ravel()

        ranked = self._books.copy()
        ranked["score"] = scores

        if category:
            ranked = ranked[ranked["category"].str.lower() == category.lower()]
        if emotion:
            ranked = ranked[ranked["emotion"].str.lower() == emotion.lower()]

        ranked = ranked[ranked["score"] >= min_score]
        ranked = ranked.sort_values("score", ascending=False).head(top_k)

        return [
            Recommendation(
                title=row["title"],
                description=row["description"],
                category=row["category"],
                emotion=row["emotion"],
                score=float(row["score"]),
            )
            for _, row in ranked.iterrows()
        ]
