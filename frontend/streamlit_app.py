"""Hybrid RAG AI Agent — Streamlit frontend.

Run (after starting the API)::

    uvicorn app.api.main:app --port 8000      # backend
    streamlit run frontend/streamlit_app.py   # this UI

The page structure lives here; rendering details are in ``ui_components``.
"""

import streamlit as st
from api_client import APIError, RAGAPIClient
from ui_components import (
    current_messages,
    init_state,
    set_title_from,
    sidebar_conversations,
    sidebar_documents,
    sidebar_stats,
)

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_client() -> RAGAPIClient:
    return RAGAPIClient()


def main() -> None:
    client = get_client()
    init_state()

    # ---- Sidebar --------------------------------------------------------- #
    st.sidebar.title("🚚 Hybrid RAG")
    sidebar_conversations()
    sidebar_documents(client)
    sidebar_stats(client)

    # ---- Main chat area ---------------------------------------------------- #
    st.title("Logistics Knowledge Assistant")
    st.caption(
        "Ask about freight, customs, dangerous goods, SOPs, or anything else. "
        "Answers combine internal documents, live web search and LLM reasoning."
    )

    messages = current_messages()
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question…")
    if not prompt:
        return

    set_title_from(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # History = everything before this turn, as the API expects it.
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = client.chat(prompt, history=history)
            except APIError as exc:
                error_text = f"⚠️ {exc}"
                st.error(error_text)
                messages.append({"role": "assistant", "content": error_text})
                return

        st.markdown(response["answer"])

        # Freight quotes (RATES route) render as a sortable table under the answer.
        quotes = response.get("rate_quotes") or []
        if quotes:
            st.dataframe(
                [
                    {
                        # Present on door-to-door quotes; blank for single-leg.
                        "Leg": q.get("leg", ""),
                        "Carrier": q.get("carrier_name"),
                        # Which rate API this price came from.
                        "Source": q.get("source")
                        or {"7lfreight": "7LFreight", "mycarrier": "MyCarrier"}.get(
                            q.get("provider"), q.get("provider", "")
                        ),
                        "Mode": q.get("mode"),
                        "Price": f"{q.get('price')} {q.get('currency', '')}".strip(),
                        # Air quotes carry no transit time from 7L — show a dash,
                        # not a literal "None".
                        "Transit (days)": q.get("transit_days")
                        if q.get("transit_days") is not None else "—",
                        "Lane": f"{q.get('origin')} → {q.get('destination')}",
                    }
                    for q in quotes
                ],
                hide_index=True,
                use_container_width=True,
            )

    messages.append({"role": "assistant", "content": response["answer"]})


main()
