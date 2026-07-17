"""Local embeddings via sentence-transformers (no API, no data egress).

Useful for air-gapped deployments or cost-free development. Implemented
directly against the ``sentence_transformers`` package rather than the
deprecated ``langchain_community.embeddings.HuggingFaceEmbeddings`` wrapper.
"""

from typing import List

from langchain_core.embeddings import Embeddings

from app.utils.exceptions import ConfigurationError


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain ``Embeddings`` adapter around a SentenceTransformer model."""

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "EMBEDDING__PROVIDER=sentence_transformers requires the "
                "'sentence-transformers' package: pip install sentence-transformers"
            ) from exc

        self._batch_size = batch_size
        # Model download happens once and is cached under ~/.cache/torch.
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            # Normalized vectors make cosine similarity a plain dot product.
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
