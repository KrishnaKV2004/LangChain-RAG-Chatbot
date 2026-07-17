"""Unit tests for DocumentChunker."""

from langchain_core.documents import Document

from app.config.settings import ChunkingSettings
from app.loaders.chunking import DocumentChunker


def make_chunker(**overrides: int) -> DocumentChunker:
    defaults = {"chunk_size": 200, "chunk_overlap": 40, "min_chunk_chars": 20}
    defaults.update(overrides)
    return DocumentChunker(ChunkingSettings(**defaults))


class TestChunking:
    def test_long_text_is_split_within_size(self) -> None:
        text = " ".join(f"sentence{i}." for i in range(300))
        chunks = make_chunker().chunk([Document(page_content=text)])
        assert len(chunks) > 1
        # Allow slight overshoot for separator handling, but no runaway chunks.
        assert all(len(c.page_content) <= 260 for c in chunks)

    def test_source_metadata_propagates_to_chunks(self) -> None:
        doc = Document(page_content="word " * 200, metadata={"filename": "a.txt", "page": 3})
        chunks = make_chunker().chunk([doc])
        assert all(c.metadata["filename"] == "a.txt" for c in chunks)
        assert all(c.metadata["page"] == 3 for c in chunks)

    def test_chunk_ids_are_deterministic(self) -> None:
        doc = Document(page_content="stable content " * 50, metadata={"source": "/x/a.txt"})
        first = make_chunker().chunk([doc])
        second = make_chunker().chunk(
            [Document(page_content="stable content " * 50, metadata={"source": "/x/a.txt"})]
        )
        assert [c.metadata["chunk_id"] for c in first] == [
            c.metadata["chunk_id"] for c in second
        ]

    def test_tiny_fragments_are_dropped(self) -> None:
        chunks = make_chunker().chunk([Document(page_content="ok")])
        assert chunks == []

    def test_markdown_sections_get_titles(self) -> None:
        md = (
            "# Manual\n\nIntro paragraph with enough length to survive filters.\n\n"
            "## Lithium Batteries\n\nUN3480 shipments need a class 9 label attached.\n"
        )
        doc = Document(page_content=md, metadata={"doc_type": "markdown"})
        chunks = make_chunker().chunk([doc])
        titles = {c.metadata.get("section_title") for c in chunks}
        assert "Lithium Batteries" in titles
