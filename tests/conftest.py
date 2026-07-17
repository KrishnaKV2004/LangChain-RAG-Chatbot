"""Shared test fixtures.

``FakeEmbeddings`` produces deterministic keyword-count vectors so vector
search behaves meaningfully (a "cargo" query really does match a cargo
document) without any network calls or model downloads.
"""

import math
from typing import List

import pytest
from langchain_core.embeddings import Embeddings

# The fake vocabulary spans the test corpus topics.
_VOCABULARY = ["cargo", "weather", "customs", "battery", "warehouse", "invoice"]


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-keywords embedding for tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> List[float]:
        lowered = text.lower()
        vector = [float(lowered.count(word)) for word in _VOCABULARY]
        vector.append(1.0)  # constant component avoids all-zero vectors
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]


@pytest.fixture()
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
