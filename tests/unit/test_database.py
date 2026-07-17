"""Unit tests for ChromaManager and IndexingService (real ChromaDB, fake
embeddings, tmp_path persistence — no network)."""

from pathlib import Path

import pytest

from app.config.settings import Settings
from app.database.chroma import ChromaManager, sanitize_metadata
from app.database.indexer import IndexingService
from app.loaders.ingestion import IngestionService
from tests.conftest import FakeEmbeddings


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("VECTOR_STORE__PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setenv("PATHS__DOCUMENTS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("CHUNKING__CHUNK_SIZE", "300")
    monkeypatch.setenv("CHUNKING__CHUNK_OVERLAP", "50")
    monkeypatch.setenv("CHUNKING__MIN_CHUNK_CHARS", "10")
    return Settings(_env_file=None)


@pytest.fixture()
def manager(settings: Settings) -> ChromaManager:
    return ChromaManager(settings, FakeEmbeddings())


@pytest.fixture()
def docs_dir(settings: Settings) -> Path:
    directory = settings.paths.documents_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cargo.txt").write_text(
        "Air cargo is transported in unit load devices. Cargo capacity varies by aircraft."
    )
    (directory / "customs.txt").write_text(
        "Customs clearance requires a commercial invoice and a packing list at customs."
    )
    return directory


@pytest.fixture()
def indexer(settings: Settings, manager: ChromaManager, docs_dir: Path) -> IndexingService:
    ingestion = IngestionService(settings.chunking)
    return IndexingService(ingestion, manager, docs_dir)


class TestSanitizeMetadata:
    def test_drops_none_and_stringifies_nonscalars(self) -> None:
        clean = sanitize_metadata({"a": 1, "b": None, "c": ["x"], "d": True})
        assert clean == {"a": 1, "c": "['x']", "d": True}


class TestChromaManager:
    def test_add_and_search(self, manager: ChromaManager, indexer: IndexingService) -> None:
        indexer.index_directory()
        store = manager.vector_store()
        results = store.similarity_search("cargo aircraft", k=1)
        assert "cargo" in results[0].page_content.lower()

    def test_upsert_is_idempotent(self, manager: ChromaManager, indexer: IndexingService) -> None:
        first = indexer.index_directory()
        count_after_first = manager.stats()["total_chunks"]
        indexer.index_directory()  # same content, same chunk IDs
        assert manager.stats()["total_chunks"] == count_after_first
        assert first.chunks_indexed == count_after_first

    def test_delete_by_source(self, manager: ChromaManager, indexer: IndexingService) -> None:
        indexer.index_directory()
        deleted = manager.delete_by_source("cargo.txt")
        assert deleted > 0
        assert "cargo.txt" not in manager.stats()["documents_by_file"]

    def test_stats_shape(self, manager: ChromaManager, indexer: IndexingService) -> None:
        indexer.index_directory()
        stats = manager.stats()
        assert stats["total_chunks"] > 0
        assert "documents" in stats["collections"]
        assert set(stats["documents_by_file"]) == {"cargo.txt", "customs.txt"}


class TestIndexingService:
    def test_index_file_replaces_previous_version(
        self, manager: ChromaManager, indexer: IndexingService, docs_dir: Path
    ) -> None:
        target = docs_dir / "cargo.txt"
        indexer.index_file(target)
        before = manager.stats()["documents_by_file"]["cargo.txt"]

        # Shrink the file; re-index must not leave orphaned old chunks.
        target.write_text("Cargo note: short version.")
        indexer.index_file(target)
        after = manager.stats()["documents_by_file"]["cargo.txt"]
        assert after <= before
        results = manager.vector_store().similarity_search("cargo", k=3)
        assert all("unit load devices" not in r.page_content for r in results)

    def test_rebuild_removes_deleted_sources(
        self, manager: ChromaManager, indexer: IndexingService, docs_dir: Path
    ) -> None:
        indexer.index_directory()
        (docs_dir / "customs.txt").unlink()  # file removed from disk

        report = indexer.rebuild()
        assert report.rebuilt is True
        assert "customs.txt" not in manager.stats()["documents_by_file"]
        assert "cargo.txt" in manager.stats()["documents_by_file"]
