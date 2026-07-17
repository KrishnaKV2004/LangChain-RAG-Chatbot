"""Embedding-provider factory (OpenAI, Voyage, Cohere, Sentence Transformers)."""

from app.embeddings.factory import create_embeddings
from app.embeddings.local import SentenceTransformerEmbeddings

__all__ = ["create_embeddings", "SentenceTransformerEmbeddings"]
