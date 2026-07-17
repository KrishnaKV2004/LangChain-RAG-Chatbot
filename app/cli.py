"""Command-line utilities for managing the knowledge base.

Usage::

    python -m app.cli ingest              # incremental index of documents/
    python -m app.cli ingest --dir path   # index a specific directory
    python -m app.cli rebuild             # drop the index and re-ingest
    python -m app.cli stats               # show vector-store statistics

Only the ingestion stack is constructed (embeddings + ChromaDB + security
tagging) — no LLM or web-search credentials are needed to index documents.
"""

import argparse
import json
import sys

from app.config.settings import Settings, get_settings
from app.database.chroma import ChromaManager
from app.database.indexer import IndexingService, IndexReport
from app.embeddings.factory import create_embeddings
from app.loaders.ingestion import IngestionService
from app.security.guard import SecurityGuard
from app.utils.exceptions import RAGChatbotError
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def build_indexer(settings: Settings) -> IndexingService:
    """Minimal composition for indexing (mirrors ``create_agent``'s wiring)."""
    settings.ensure_directories()
    embeddings = create_embeddings(settings)
    manager = ChromaManager(settings, embeddings)
    guard = SecurityGuard(settings.security)
    ingestion = IngestionService(settings.chunking)
    return IndexingService(
        ingestion,
        manager,
        settings.paths.documents_dir,
        chunk_processor=guard.tag_sensitivity,
    )


def _print_report(report: IndexReport) -> None:
    print(f"Files processed : {len(report.files_processed)}")
    for name in report.files_processed:
        print(f"  ✓ {name}")
    if report.files_skipped:
        print(f"Files skipped   : {', '.join(report.files_skipped)}")
    for name, error in report.errors.items():
        print(f"  ✗ {name}: {error}")
    print(f"Chunks indexed  : {report.chunks_indexed}")
    print(f"Duration        : {report.duration_ms / 1000:.1f}s")


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Index documents (incremental upsert)")
    ingest.add_argument("--dir", default=None, help="Directory to ingest (default: documents/)")
    subcommands.add_parser("rebuild", help="Drop the index and re-ingest everything")
    subcommands.add_parser("stats", help="Print vector-store statistics as JSON")
    chat = subcommands.add_parser("chat", help="Ask one question through the full pipeline")
    chat.add_argument("question", help="The question to ask")

    arguments = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    try:
        if arguments.command == "chat":
            from app.agents.agent import create_agent

            response = create_agent(settings).chat(arguments.question)
            print(f"\n[route: {response.route}]\n\n{response.answer}\n")
            internal = response.citations.get("internal", [])
            external = response.citations.get("external", [])
            if internal or external:
                print("Sources:")
                for label in internal:
                    print(f"  📄 {label}")
                for source in external:
                    print(f"  🌐 {source['title']} — {source['url']}")
            print(
                f"\n({response.total_latency_ms:.0f} ms, "
                f"{response.token_usage.get('total_tokens', 0)} tokens)"
            )
            return 0

        if arguments.command == "stats":
            settings.ensure_directories()
            manager = ChromaManager(settings, create_embeddings(settings))
            print(json.dumps(manager.stats(), indent=2))
            return 0

        indexer = build_indexer(settings)
        if arguments.command == "rebuild":
            _print_report(indexer.rebuild())
        else:
            _print_report(indexer.index_directory(arguments.dir))
        return 0
    except RAGChatbotError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
