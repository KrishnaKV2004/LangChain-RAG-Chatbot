"""Unit tests for the IngestionService."""

from pathlib import Path

import pytest

from app.config.settings import ChunkingSettings
from app.loaders.ingestion import IngestionService
from app.utils.exceptions import DocumentLoadError, IngestionError


@pytest.fixture()
def service() -> IngestionService:
    return IngestionService(
        ChunkingSettings(chunk_size=300, chunk_overlap=50, min_chunk_chars=20)
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A small mixed directory: good files, an unsupported one, a corrupt one."""
    (tmp_path / "guide.txt").write_text("Air cargo must be weighed and labeled. " * 20)
    (tmp_path / "faq.md").write_text("# FAQ\n\n## AWB\n\nAn air waybill is a contract of carriage between shipper and carrier.\n")
    (tmp_path / "video.mp4").write_bytes(b"\x00\x01")          # unsupported
    (tmp_path / "broken.json").write_text("{not valid json")    # corrupt
    (tmp_path / ".hidden.txt").write_text("should be ignored")  # hidden
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "rates.csv").write_text("lane,rate\nICN-DXB,4.2\n")
    return tmp_path


class TestIngestFile:
    def test_file_becomes_chunks_with_ids(self, service: IngestionService, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("Customs clearance requires a commercial invoice. " * 30)
        chunks = service.ingest_file(path)
        assert len(chunks) > 1
        assert all("chunk_id" in c.metadata for c in chunks)
        assert all(c.metadata["filename"] == "doc.txt" for c in chunks)

    def test_unsupported_file_raises(self, service: IngestionService, tmp_path: Path) -> None:
        path = tmp_path / "img.png"
        path.write_bytes(b"\x89PNG")
        with pytest.raises(DocumentLoadError):
            service.ingest_file(path)


class TestIngestDirectory:
    def test_batch_survives_bad_files(self, service: IngestionService, corpus: Path) -> None:
        result = service.ingest_directory(corpus)
        assert sorted(result.files_processed) == ["faq.md", "guide.txt", "rates.csv"]
        assert result.files_skipped == ["video.mp4"]
        assert "broken.json" in result.errors
        assert result.chunk_count > 0
        assert result.duration_ms > 0

    def test_non_recursive_skips_subdirectories(
        self, service: IngestionService, corpus: Path
    ) -> None:
        result = service.ingest_directory(corpus, recursive=False)
        assert "rates.csv" not in result.files_processed

    def test_missing_directory_raises(self, service: IngestionService) -> None:
        with pytest.raises(IngestionError):
            service.ingest_directory(Path("/nonexistent/dir"))
