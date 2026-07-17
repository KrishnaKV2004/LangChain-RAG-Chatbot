"""Reusable Streamlit rendering helpers (sidebar sections, message blocks).

Kept separate from ``streamlit_app.py`` so the main file reads as the page's
structure while the details live here.
"""

import uuid
from typing import Any, Dict, List

import streamlit as st

from api_client import APIError, RAGAPIClient

#: Sidebar search-mode label → API force_route value (None = automatic).
SEARCH_MODES = {
    "Auto (LLM router)": None,
    "Internal documents": "INTERNAL_ONLY",
    "Web search": "WEB_ONLY",
    "Hybrid": "HYBRID",
    "Chat only": "GENERAL_CHAT",
}

_ROUTE_BADGES = {
    "INTERNAL_ONLY": "📄 internal",
    "WEB_ONLY": "🌐 web",
    "HYBRID": "🔀 hybrid",
    "GENERAL_CHAT": "💬 chat",
}


# --------------------------------------------------------------------------- #
# Session-state conversation store
# --------------------------------------------------------------------------- #


def init_state() -> None:
    """Create the conversation store on first load."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
        new_conversation()


def new_conversation() -> None:
    conversation_id = str(uuid.uuid4())[:8]
    st.session_state.conversations[conversation_id] = {"title": "New chat", "messages": []}
    st.session_state.current_id = conversation_id


def current_messages() -> List[Dict[str, Any]]:
    return st.session_state.conversations[st.session_state.current_id]["messages"]


def set_title_from(query: str) -> None:
    """First user message names the conversation in the sidebar."""
    conversation = st.session_state.conversations[st.session_state.current_id]
    if conversation["title"] == "New chat":
        conversation["title"] = query[:40] + ("…" if len(query) > 40 else "")


# --------------------------------------------------------------------------- #
# Sidebar sections
# --------------------------------------------------------------------------- #


def sidebar_conversations() -> None:
    st.sidebar.subheader("Conversations")
    if st.sidebar.button("➕ New chat", use_container_width=True):
        new_conversation()
        st.rerun()
    for conversation_id, conversation in list(st.session_state.conversations.items()):
        is_current = conversation_id == st.session_state.current_id
        label = ("● " if is_current else "") + conversation["title"]
        if st.sidebar.button(label, key=f"conv-{conversation_id}", use_container_width=True):
            st.session_state.current_id = conversation_id
            st.rerun()


def sidebar_search_mode() -> str:
    st.sidebar.divider()
    st.sidebar.subheader("Search mode")
    label = st.sidebar.radio(
        "Where to look for answers",
        list(SEARCH_MODES),
        label_visibility="collapsed",
    )
    return SEARCH_MODES[label]


def sidebar_documents(client: RAGAPIClient) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Knowledge base")

    uploaded = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx", "pptx", "html", "json"],
        accept_multiple_files=True,
    )
    if uploaded and st.sidebar.button("Index uploaded files", use_container_width=True):
        for file in uploaded:
            try:
                report = client.upload(file.name, file.getvalue())
                st.sidebar.success(f"{file.name}: {report['chunks_indexed']} chunks")
            except APIError as exc:
                st.sidebar.error(f"{file.name}: {exc}")

    if st.sidebar.button("🔄 Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding the vector index…"):
            try:
                report = client.reindex(rebuild=True)
                st.sidebar.success(
                    f"Re-indexed {len(report['files_processed'])} files "
                    f"({report['chunks_indexed']} chunks)"
                )
            except APIError as exc:
                st.sidebar.error(str(exc))


def sidebar_stats(client: RAGAPIClient) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Vector statistics")
    try:
        health = client.health()
        store = health.get("vector_store", {})
        st.sidebar.metric("Total chunks", store.get("total_chunks", "—"))
        by_file = store.get("documents_by_file", {})
        if by_file:
            with st.sidebar.expander(f"{len(by_file)} indexed files"):
                for filename, count in sorted(by_file.items()):
                    st.write(f"• {filename} — {count} chunks")
        cache = client.cache_stats()
        st.sidebar.caption(
            f"Web cache: {cache['entries']} entries "
            f"({cache['expired_entries']} expired)"
        )
    except APIError:
        st.sidebar.warning("Backend offline — start the API first.")


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #


def render_assistant_extras(meta: Dict[str, Any]) -> None:
    """Citations, retrieved chunks and performance details for one answer."""
    citations = meta.get("citations", {})
    internal = citations.get("internal", [])
    external = citations.get("external", [])
    if internal or external:
        st.markdown("**Sources**")
        for label in internal:
            st.markdown(f"- 📄 {label}")
        for source in external:
            st.markdown(f"- 🌐 [{source['title']}]({source['url']})")

    chunks = meta.get("retrieved_chunks", [])
    if chunks:
        with st.expander(f"Retrieved chunks ({len(chunks)})"):
            for chunk in chunks:
                provenance = chunk.get("filename") or "unknown"
                if chunk.get("page"):
                    provenance += f", page {chunk['page']}"
                if chunk.get("section_title"):
                    provenance += f" — {chunk['section_title']}"
                st.markdown(f"**{provenance}**")
                st.text(chunk.get("text", ""))
                st.divider()

    badge = _ROUTE_BADGES.get(meta.get("route", ""), meta.get("route", ""))
    tokens = meta.get("token_usage", {}).get("total_tokens", 0)
    latency = meta.get("total_latency_ms", 0)
    cache_note = " · cache hit" if meta.get("web_cache_hit") else ""
    st.caption(f"{badge} · {latency:.0f} ms · {tokens} tokens{cache_note}")
