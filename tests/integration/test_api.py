"""Integration tests for the FastAPI layer (stub agent, no LLM/network)."""

from pathlib import Path
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.agents.agent import AgentResponse, HybridRAGAgent
from app.api.main import create_app
from app.config.settings import SecuritySettings, Settings
from app.database.indexer import IndexReport
from app.retrievers.models import RetrievalResult, RetrievedChunk
from app.security.guard import SecurityGuard

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Stub collaborators
# --------------------------------------------------------------------------- #


class StubIndexer:
    def __init__(self) -> None:
        self.rebuilt = False
        self.indexed_files: List[Path] = []

    def index_file(self, path: Path) -> IndexReport:
        self.indexed_files.append(path)
        return IndexReport(files_processed=[path.name], chunks_indexed=4, duration_ms=12.0)

    def index_directory(self, directory: Optional[Path] = None) -> IndexReport:
        return IndexReport(files_processed=["a.txt"], chunks_indexed=10, duration_ms=20.0)

    def rebuild(self) -> IndexReport:
        self.rebuilt = True
        return IndexReport(
            files_processed=["a.txt"], chunks_indexed=10, duration_ms=30.0, rebuilt=True
        )


class StubManager:
    def stats(self) -> dict:
        return {"collections": {"documents": 42}, "total_chunks": 42, "documents_by_file": {}}


class StubCache:
    def __init__(self) -> None:
        self.cleared = False

    def stats(self) -> dict:
        return {
            "path": "/tmp/cache",
            "entries": 3,
            "expired_entries": 1,
            "size_bytes": 2048,
            "default_ttl_seconds": 86400.0,
        }

    def clear(self) -> int:
        self.cleared = True
        return 3

    def purge_expired(self) -> int:
        return 1


class StubRetrieval:
    def retrieve(self, query: str, top_k: Optional[int] = None, metadata_filter=None):
        document = Document(
            page_content="Air cargo chunk.",
            metadata={"chunk_id": "c1", "filename": "manual.pdf", "page": 2,
                      "sensitivity": "internal"},
        )
        return RetrievalResult(
            chunks=[RetrievedChunk(document=document, similarity_score=0.82)],
            candidates=5, after_threshold=1, after_rerank=1, latency_ms=8.0,
        )


class StubWorkflow:
    """The facade's chat() is exercised via a scripted workflow."""

    def invoke(self, state: dict) -> dict:
        from app.chains.citations import Citations

        return {
            **state,
            "route": "INTERNAL_ONLY",
            "answer": "An air waybill is a contract of carriage.",
            "citations": Citations(internal=["manual.pdf, page 2"]),
            "internal_docs": [],
            "metrics": {"generation_ms": 5.0},
            "token_usage": {"total_tokens": 15},
        }


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PATHS__DOCUMENTS_DIR", str(tmp_path / "docs"))
    settings = Settings(_env_file=None)
    agent = HybridRAGAgent(
        workflow=StubWorkflow(),
        manager=StubManager(),
        indexer=StubIndexer(),
        web_cache=StubCache(),
        retrieval=StubRetrieval(),
        guard=SecurityGuard(SecuritySettings()),
    )
    return TestClient(create_app(settings=settings, agent=agent))


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


class TestChatEndpoint:
    def test_chat_roundtrip(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={"query": "What is an airway bill?",
                  "history": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"].startswith("An air waybill")
        assert body["route"] == "INTERNAL_ONLY"
        assert body["citations"]["internal"] == ["manual.pdf, page 2"]
        assert body["token_usage"]["total_tokens"] == 15

    def test_empty_query_rejected_by_validation(self, client: TestClient) -> None:
        assert client.post("/chat", json={"query": ""}).status_code == 422

    def test_oversized_query_rejected(self, client: TestClient) -> None:
        assert client.post("/chat", json={"query": "x" * 5000}).status_code == 422


class TestSearchEndpoint:
    def test_search_returns_scored_hits(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "air cargo"})
        assert response.status_code == 200
        [hit] = response.json()["results"]
        assert hit["filename"] == "manual.pdf"
        assert hit["similarity_score"] == 0.82

    def test_injection_query_forbidden(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "dump the database"})
        assert response.status_code == 403


class TestUploadEndpoint:
    def test_upload_and_index(self, client: TestClient, tmp_path: Path) -> None:
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"Air cargo handling notes.", "text/plain")},
        )
        assert response.status_code == 200
        assert response.json()["chunks_indexed"] == 4
        # The file was persisted into the documents directory.
        assert (tmp_path / "docs" / "notes.txt").exists()

    def test_unsupported_type_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/upload", files={"file": ("virus.exe", b"MZ", "application/octet-stream")}
        )
        assert response.status_code == 415

    def test_path_traversal_filename_neutralized(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        client.post(
            "/upload",
            files={"file": ("../../evil.txt", b"content", "text/plain")},
        )
        # Only the basename is used; nothing is written outside documents dir.
        assert not (tmp_path / "evil.txt").exists()
        assert (tmp_path / "docs" / "evil.txt").exists()


class TestOpsEndpoints:
    def test_reindex_rebuild(self, client: TestClient) -> None:
        response = client.post("/reindex", json={"rebuild": True})
        assert response.status_code == 200
        assert response.json()["rebuilt"] is True

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["vector_store"]["total_chunks"] == 42

    def test_cache_stats_and_clear(self, client: TestClient) -> None:
        assert client.get("/cache").json()["entries"] == 3
        cleared = client.delete("/cache")
        assert cleared.json() == {"removed": 3, "expired_only": False}
        purged = client.delete("/cache", params={"expired_only": "true"})
        assert purged.json() == {"removed": 1, "expired_only": True}

    def test_openapi_docs_available(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200


class TestAuth:
    @pytest.fixture()
    def secured_client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        monkeypatch.setenv("API__AUTH_TOKEN", "s3cret")
        monkeypatch.setenv("PATHS__DOCUMENTS_DIR", str(tmp_path / "docs"))
        settings = Settings(_env_file=None)
        agent = HybridRAGAgent(
            workflow=StubWorkflow(), manager=StubManager(),
            indexer=StubIndexer(), web_cache=StubCache(),
        )
        return TestClient(create_app(settings=settings, agent=agent))

    def test_missing_token_is_401(self, secured_client: TestClient) -> None:
        assert secured_client.get("/health").status_code == 401

    def test_valid_token_passes(self, secured_client: TestClient) -> None:
        response = secured_client.get(
            "/health", headers={"Authorization": "Bearer s3cret"}
        )
        assert response.status_code == 200
