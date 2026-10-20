"""
app/main.py
───────────
Streamlit UI — calls the FastAPI backend at localhost:8000.

Run separately:
    streamlit run app/main.py

Start the API first:
    uvicorn app.api:app --port 8000
"""

import os

import httpx
import streamlit as st


# ── Configuration ─────────────────────────────────────────────────────────────

# When running via docker-compose:
# API_BASE=http://api:8000
API_BASE = os.getenv("API_BASE", "http://localhost:8000")


st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Agentic RAG Document Assistant")
st.caption(
    "Hybrid retrieval · Cross-encoder reranking · "
    "Corrective RAG · Web fallback"
)


# ── Helper: render unique sources ─────────────────────────────────────────────

def render_sources(sources):
    """
    Display unique document sources.

    Multiple retrieved chunks can come from the same PDF page.
    We deduplicate them using (source, page).
    """

    if not sources:
        return

    unique_sources = {
        (s.get("source", "Unknown"), s.get("page", "?"))
        for s in sources
    }

    with st.expander("📚 Sources"):
        for source, page in sorted(
            unique_sources,
            key=lambda x: (str(x[0]), str(x[1]))
        ):
            st.markdown(
                f"- **{source}** — Page {page}"
            )


# ── Sidebar — upload ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Documents")

    uploaded = st.file_uploader(
        "Choose a PDF",
        type="pdf",
    )

    if uploaded and st.button("Index Document"):
        with st.spinner("Ingesting and indexing …"):
            try:
                resp = httpx.post(
                    f"{API_BASE}/ingest",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=120,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    st.success(
                        f"Indexed {data['chunks']} chunks "
                        f"from {data['filename']}"
                    )

                else:
                    st.error(f"Error: {resp.text}")

            except httpx.ConnectError:
                st.error(
                    "Cannot reach API. Start it with: "
                    "uvicorn app.api:app --port 8000"
                )


    st.divider()

    # ── RAGAS Scores ──────────────────────────────────────────────────────────

    st.header("RAGAS Scores")

    if st.button("Load Latest Scores"):
        try:
            resp = httpx.get(
                f"{API_BASE}/metrics",
                timeout=10,
            )

            scores = resp.json()

            if "detail" not in scores:
                for k, v in scores.items():
                    st.metric(
                        k.replace("_", " ").title(),
                        f"{v:.3f}",
                    )
            else:
                st.info(scores["detail"])

        except httpx.ConnectError:
            st.error(
                "Cannot reach API. Make sure FastAPI is running."
            )


# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        # Render deduplicated sources
        render_sources(msg.get("sources", []))

        # Render metadata
        if msg.get("meta"):

            m = msg["meta"]

            cols = st.columns(3)

            cols[0].metric(
                "Latency",
                f"{m['latency_ms']} ms",
            )

            cols[1].metric(
                "Rewrites",
                m["rewrite_count"],
            )

            cols[2].metric(
                "Web used",
                "Yes" if m["used_web"] else "No",
            )


# ── User input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input(
    "Ask a question about your documents …"
):

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)


    # ── Assistant response ────────────────────────────────────────────────────

    with st.chat_message("assistant"):

        with st.spinner("Thinking …"):

            try:

                resp = httpx.post(
                    f"{API_BASE}/query",
                    json={
                        "question": prompt,
                        "history": [
                            {
                                "role": m["role"],
                                "content": m["content"],
                            }
                            for m in st.session_state.messages[:-1]
                        ],
                    },
                    timeout=120,
                )


                # ── Successful response ───────────────────────────────────────

                if resp.status_code == 200:

                    data = resp.json()

                    # Answer
                    st.markdown(data["answer"])

                    # Deduplicated sources
                    render_sources(
                        data.get("sources", [])
                    )

                    # Metrics
                    cols = st.columns(3)

                    cols[0].metric(
                        "Latency",
                        f"{data['latency_ms']} ms",
                    )

                    cols[1].metric(
                        "Rewrites",
                        data["rewrite_count"],
                    )

                    cols[2].metric(
                        "Web used",
                        "Yes" if data["used_web"] else "No",
                    )


                    # Save assistant response to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data.get(
                                "sources",
                                [],
                            ),
                            "meta": {
                                "latency_ms": data[
                                    "latency_ms"
                                ],
                                "rewrite_count": data[
                                    "rewrite_count"
                                ],
                                "used_web": data[
                                    "used_web"
                                ],
                            },
                        }
                    )


                # ── API error ────────────────────────────────────────────────

                else:

                    try:
                        err = resp.json().get(
                            "detail",
                            resp.text,
                        )
                    except Exception:
                        err = resp.text

                    st.error(
                        f"API error: {err}"
                    )


            # ── Connection error ──────────────────────────────────────────────

            except httpx.ConnectError:

                st.error(
                    "Cannot reach API. "
                    "Start it with: "
                    "uvicorn app.api:app --port 8000"
                )


            # ── Timeout ───────────────────────────────────────────────────────

            except httpx.TimeoutException:

                st.error(
                    "The request timed out. "
                    "The local model may still be processing."
                )
