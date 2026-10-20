"""
app/agent.py
------------
Corrective RAG Agent built with LangGraph.

Graph nodes
-----------
route_query       -> decides: documents OR web OR llm_only
retrieve          -> hybrid FAISS + BM25 search + cross-encoder rerank
grade_documents   -> batch relevance grading
rewrite_query     -> rewrites retrieval query if document retrieval fails
web_search        -> Tavily search for current information or fallback
generate          -> produces final grounded answer

Important:
- original_query always stores the user's real question.
- query is the working retrieval/search query and may be rewritten.
"""

from __future__ import annotations

import os
from typing import (
    Any,
    Dict,
    List,
    Annotated,
    Literal,
    Optional,
    TypedDict,
)

from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from loguru import logger

from vectorstore.store import hybrid_search


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.1,
)


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

web_search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
)


# Maximum number of corrective retrieval attempts
MAX_REWRITES = 2


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class _AgentStateRequired(TypedDict):

    messages: Annotated[List[BaseMessage], add_messages]

    # User's actual question.
    # This must NEVER be overwritten by query rewriting.
    original_query: str

    # Working retrieval/search query.
    # This may be rewritten during Corrective RAG.
    query: str

    rewrite_count: int

    # Retrieved document chunks
    context: List[Dict[str, Any]]

    # Tavily snippets
    web_results: List[str]

    answer: str


class AgentState(_AgentStateRequired, total=False):
    """
    LangGraph state.

    _route:
        documents
        web
        llm_only

    _grade:
        generate
        rewrite
    """

    _route: str
    _grade: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_human_query(state: AgentState) -> str:
    """
    Extract the most recent user query from conversation history.
    """

    for msg in reversed(state["messages"]):

        if isinstance(msg, HumanMessage):
            return str(msg.content)

    return state.get("original_query") or state.get("query") or ""


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> AgentState:
    """
    Fast deterministic router.

    Avoids an unnecessary LLM call for routing.
    """

    query = _last_human_query(state)
    q = query.lower()

    # -------------------------------------------------------
    # Document indicators
    # -------------------------------------------------------

    document_keywords = [
        "uploaded document",
        "uploaded pdf",
        "uploaded file",
        "my document",
        "my pdf",
        "my file",
        "the document",
        "the pdf",
        "the report",
        "the paper",
        "according to the document",
        "according to the pdf",
        "according to the report",
        "according to the paper",
        "in the document",
        "in the pdf",
        "from the document",
        "from the pdf",
        "summarize the document",
        "summarize the pdf",
    ]

    # -------------------------------------------------------
    # Current/live information indicators
    # -------------------------------------------------------

    web_keywords = [
        "today",
        "latest",
        "current",
        "currently",
        "recent",
        "recently",
        "yesterday",
        "this week",
        "this month",
        "right now",
        "live",
        "real-time",
        "real time",
        "stock price",
        "weather",
        "news",
    ]

    # Documents get priority.
    # Example:
    # "According to the uploaded PDF, what is the latest..."
    # should still be answered from the PDF.
    if any(keyword in q for keyword in document_keywords):
        decision = "documents"

    elif any(keyword in q for keyword in web_keywords):
        decision = "web"

    else:
        decision = "llm_only"

    logger.info(
        f"[route_query] fast route='{decision}' "
        f"for query='{query[:60]}'"
    )

    return {
        **state,
        "original_query": query,
        "query": query,
        "_route": decision,
    }

# ---------------------------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------------------------

def retrieve(state: AgentState) -> AgentState:
    """
    Hybrid retrieval pipeline:

        FAISS semantic retrieval
                +
        BM25 keyword retrieval
                ↓
        Reciprocal Rank Fusion
                ↓
        Cross-encoder reranking
                ↓
        Top contexts
    """

    query = state["query"]

    results = hybrid_search(
        query,
        top_k=5,
    )

    logger.info(
        f"[retrieve] got {len(results)} chunks"
    )

    return {
        **state,
        "context": results,
    }


# ---------------------------------------------------------------------------
# DOCUMENT GRADING
# ---------------------------------------------------------------------------

def grade_documents(state: AgentState) -> AgentState:
    """
    Grade all retrieved chunks in ONE LLM call.

    This is significantly faster than making one LLM call
    for every retrieved chunk.
    """

    query = state["query"]
    context = state["context"]

    if not context:
        return {
            **state,
            "_grade": "rewrite",
        }

    chunks_text = "\n\n".join(
        f"CHUNK {i}:\n{chunk['text']}"
        for i, chunk in enumerate(context)
    )

    prompt = (
        f"Question:\n{query}\n\n"

        f"{chunks_text}\n\n"

        "Determine which chunks contain information useful for "
        "answering the question.\n\n"

        "Important rules:\n"
        "- A chunk is relevant only if it contains information that "
        "actually helps answer the question.\n"
        "- Mere keyword overlap is not enough.\n"
        "- Do not mark unrelated chunks as relevant.\n\n"

        "Return ONLY the numbers of relevant chunks separated by commas.\n"
        "Example: 0,2,4\n"

        "If no chunks are relevant, reply exactly:\n"
        "NONE"
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    decision = str(response.content).strip()

    relevant = []

    if decision.upper() != "NONE":

        try:

            indices = [
                int(x.strip())
                for x in decision.split(",")
                if x.strip().isdigit()
            ]

            relevant = [
                context[i]
                for i in indices
                if 0 <= i < len(context)
            ]

        except Exception:
            relevant = []

    logger.info(
        f"[grade_documents] "
        f"{len(relevant)}/{len(context)} chunks relevant "
        f"(batch graded)"
    )

    if relevant:

        return {
            **state,
            "context": relevant,
            "_grade": "generate",
        }

    return {
        **state,
        "context": [],
        "_grade": "rewrite",
    }


# ---------------------------------------------------------------------------
# QUERY REWRITING
# ---------------------------------------------------------------------------

def rewrite_query(state: AgentState) -> AgentState:
    """
    Corrective RAG query rewriting.

    IMPORTANT:
    Only state["query"] is changed.

    state["original_query"] always remains the user's
    original question.
    """

    working_query = state["query"]
    original_query = state["original_query"]

    prompt = (
        "A retrieval query failed to find useful information "
        "inside the indexed documents.\n\n"

        f"Original user question:\n"
        f"{original_query}\n\n"

        f"Current retrieval query:\n"
        f"{working_query}\n\n"

        "Rewrite the retrieval query so that it is more precise, "
        "specific, and likely to match relevant passages in the "
        "indexed documents.\n\n"

        "Rules:\n"
        "- Preserve the user's original intent.\n"
        "- Do not invent facts.\n"
        "- Do not change the subject of the question.\n"
        "- Do not answer the question.\n"
        "- Return ONLY the rewritten retrieval query."
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    new_query = str(response.content).strip()

    rewrite_count = (
        int(state.get("rewrite_count") or 0) + 1
    )

    logger.info(
        f"[rewrite_query] attempt {rewrite_count}: "
        f"'{new_query[:80]}'"
    )

    return {
        **state,

        # Only modify the working query.
        "query": new_query,

        "rewrite_count": rewrite_count,
    }


# ---------------------------------------------------------------------------
# WEB SEARCH
# ---------------------------------------------------------------------------

def web_search(state: AgentState) -> AgentState:
    """
    Search the web using Tavily.

    Used in two situations:

    1. Router detects that the ORIGINAL question needs
       current information.

    2. Corrective document retrieval fails after MAX_REWRITES.

    The original user question remains preserved.
    """

    working_query = state["query"]
    original_query = state["original_query"]

    # For a normal direct web query, use the original user question.
    #
    # For corrective document fallback, use the rewritten working query
    # because it may be better optimized for retrieval.
    if state.get("_route") == "web":
        search_query = original_query
    else:
        search_query = working_query

    logger.info(
        f"[web_search] searching web for "
        f"'{search_query[:80]}'"
    )

    try:

        results = web_search_tool.invoke(
            search_query
        )

        snippets = []

        for result in results:

            if isinstance(result, dict):

                content = result.get("content")

                if content:
                    snippets.append(content)

        logger.info(
            f"[web_search] got {len(snippets)} web results"
        )

        return {
            **state,

            "web_results": snippets,

            # Important:
            # Never allow stale/irrelevant document context to
            # mix with web fallback results.
            "context": [],
        }

    except Exception as exc:

        logger.exception(
            f"[web_search] failed: {exc}"
        )

        return {
            **state,
            "web_results": [],
            "context": [],
        }


# ---------------------------------------------------------------------------
# ANSWER GENERATION
# ---------------------------------------------------------------------------

def generate(state: AgentState) -> AgentState:
    """
    Generate the final answer using one of:

        document context
        OR
        web results
        OR
        general LLM knowledge

    The final answer ALWAYS answers original_query.

    query is only the working retrieval/search query.
    """

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    original_query: str = state.get(
        "original_query",
        state["query"],
    )

    working_query: str = state["query"]


    # -----------------------------------------------------------------------
    # Context
    # -----------------------------------------------------------------------

    context: List[Dict[str, Any]] = (
        state.get("context") or []
    )

    web_results: List[str] = (
        state.get("web_results") or []
    )

    history: List[BaseMessage] = (
        state.get("messages") or []
    )


    # -----------------------------------------------------------------------
    # Build source/context instructions
    # -----------------------------------------------------------------------

    if context:

        ctx_block = "\n\n".join(
            (
                f"[Source: {chunk['source']}, "
                f"Page {chunk['page']}]\n"
                f"{chunk['text']}"
            )
            for chunk in context
        )

        source_note = (
            "The user is asking about uploaded/indexed documents.\n"
            "Answer ONLY using information supported by the document "
            "context below.\n\n"

            "GROUNDING RULES:\n"
            "1. Do not invent facts that are absent from the context.\n"
            "2. Do not use outside knowledge to fill missing information.\n"
            "3. If the supplied context does not support a claim, do not "
            "make that claim.\n"
            "4. Cite document evidence using [Source, Page X].\n"
            "5. Never claim that information appears in the document unless "
            "the provided context actually contains it."
        )


    elif web_results:

        ctx_block = "\n\n".join(
            f"[Web result {i + 1}]\n{result}"
            for i, result in enumerate(web_results)
        )


        # ---------------------------------------------------------------
        # IMPORTANT CASE:
        # User asked about documents, but document retrieval failed.
        # ---------------------------------------------------------------

        if state.get("_route") == "documents":

            source_note = (
                "The user originally asked a question about their uploaded "
                "documents.\n\n"

                "Document retrieval was attempted multiple times but no "
                "relevant document evidence was found.\n\n"

                "The context below comes ONLY from a WEB SEARCH. "
                "It does NOT come from the uploaded documents.\n\n"

                "STRICT RESPONSE RULES:\n"

                "1. First clearly state that the uploaded/indexed documents "
                "did not contain relevant information answering the user's "
                "question.\n"

                "2. Never imply that any web information came from the "
                "uploaded document.\n"

                "3. If the web results contain useful information, you may "
                "provide it only as a clearly separated supplemental section "
                "such as 'Web information:'.\n"

                "4. Do not create a relationship between the document topic "
                "and the web topic unless the evidence explicitly supports "
                "that relationship.\n"

                "5. Do not speculate.\n"

                "6. Answer the ORIGINAL user question, not merely the "
                "rewritten retrieval query."
            )


        # ---------------------------------------------------------------
        # Normal current/live web query
        # ---------------------------------------------------------------

        else:

            source_note = (
                "The user asked a question requiring current or recent "
                "information.\n"

                "Use the provided web search results to answer the question.\n"

                "Prioritize the provided web information over pretrained "
                "knowledge because the information may be time-sensitive.\n"

                "Do not invent details that are not supported by the "
                "provided web results."
            )


    # -----------------------------------------------------------------------
    # General knowledge
    # -----------------------------------------------------------------------

    else:

        ctx_block = ""

        # Special case:
        # document retrieval failed AND web search also failed
        if state.get("_route") == "documents":

            source_note = (
                "The user asked about uploaded documents, but no relevant "
                "document evidence was found and no useful web fallback "
                "information is available.\n\n"

                "Clearly tell the user that the uploaded/indexed documents "
                "do not contain enough relevant information to answer the "
                "question.\n"

                "Do NOT answer using general knowledge because the user "
                "specifically asked about the documents."
            )

        else:

            source_note = (
                "Answer using your general knowledge.\n"
                "If you do not know the answer, say that you do not know.\n"
                "Do not invent facts."
            )


    # -----------------------------------------------------------------------
    # Conversation history
    # -----------------------------------------------------------------------

    history_text = "\n".join(

        f"{'User' if isinstance(message, HumanMessage) else 'Assistant'}: "
        f"{message.content}"

        for message in history[-6:]
    )


    # -----------------------------------------------------------------------
    # Prompt
    # -----------------------------------------------------------------------

    system = (
        "You are a precise and helpful AI research assistant.\n\n"

        "Always preserve the intent of the user's original question.\n"
        "Never confuse retrieved document evidence with web information.\n"
        "Never present unsupported information as if it came from a source.\n\n"

        f"{source_note}"
    )


    user_prompt = (
        f"Conversation so far:\n"
        f"{history_text}\n\n"

        f"Evidence/context:\n"
        f"{ctx_block}\n\n"

        f"Original user question:\n"
        f"{original_query}\n\n"

        f"Working retrieval/search query:\n"
        f"{working_query}\n\n"

        "Answer the ORIGINAL user question."
    )


    response = llm.invoke(
        [
            HumanMessage(
                content=(
                    f"[SYSTEM]\n"
                    f"{system}\n\n"

                    f"[USER]\n"
                    f"{user_prompt}"
                )
            )
        ]
    )


    answer = str(response.content)

    logger.info(
        f"[generate] answer length="
        f"{len(answer)} chars"
    )


    return {
        **state,

        "answer": answer,

        "messages": (
            state["messages"]
            + [AIMessage(content=answer)]
        ),
    }


# ---------------------------------------------------------------------------
# CONDITIONAL ROUTING
# ---------------------------------------------------------------------------

def route_after_routing(
    state: AgentState,
) -> Literal[
    "retrieve",
    "web_search",
    "generate",
]:

    route = state.get("_route")

    if route == "documents":
        return "retrieve"

    if route == "web":
        return "web_search"

    return "generate"


def route_after_grading(
    state: AgentState,
) -> Literal[
    "generate",
    "rewrite_query",
    "web_search",
]:

    grade = state.get("_grade") or "generate"

    # Relevant documents found
    if grade == "generate":
        return "generate"

    rewrite_count = int(
        state.get("rewrite_count") or 0
    )

    # Retrieval failed repeatedly -> use web
    if rewrite_count >= MAX_REWRITES:

        logger.info(
            "[route_after_grading] "
            "maximum rewrites reached -> web search"
        )

        return "web_search"

    # Try another document query
    return "rewrite_query"


# ---------------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------------

def build_graph():

    graph = StateGraph(AgentState)


    # -----------------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------------

    graph.add_node(
        "route_query",
        route_query,
    )

    graph.add_node(
        "retrieve",
        retrieve,
    )

    graph.add_node(
        "grade_documents",
        grade_documents,
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query,
    )

    graph.add_node(
        "web_search",
        web_search,
    )

    graph.add_node(
        "generate",
        generate,
    )


    # -----------------------------------------------------------------------
    # Entry
    # -----------------------------------------------------------------------

    graph.set_entry_point(
        "route_query"
    )


    # -----------------------------------------------------------------------
    # Initial routing
    # -----------------------------------------------------------------------

    graph.add_conditional_edges(

        "route_query",

        route_after_routing,

        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "generate": "generate",
        },
    )


    # -----------------------------------------------------------------------
    # Document RAG
    # -----------------------------------------------------------------------

    graph.add_edge(
        "retrieve",
        "grade_documents",
    )


    graph.add_conditional_edges(

        "grade_documents",

        route_after_grading,

        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )


    # -----------------------------------------------------------------------
    # Corrective RAG loop
    # -----------------------------------------------------------------------

    graph.add_edge(
        "rewrite_query",
        "retrieve",
    )


    # -----------------------------------------------------------------------
    # Web -> generation
    # -----------------------------------------------------------------------

    graph.add_edge(
        "web_search",
        "generate",
    )


    # -----------------------------------------------------------------------
    # Generation -> finish
    # -----------------------------------------------------------------------

    graph.add_edge(
        "generate",
        END,
    )


    return graph.compile()


# ---------------------------------------------------------------------------
# Compile graph once
# ---------------------------------------------------------------------------

rag_agent = build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(
    query: str,
    history: Optional[List[BaseMessage]] = None,
) -> Dict[str, Any]:
    """
    Main entry point used by FastAPI / Streamlit.
    """

    messages = (
        history or []
    ) + [
        HumanMessage(content=query)
    ]


    initial_state: AgentState = {

        "messages": messages,

        # Permanent user question
        "original_query": query,

        # Working retrieval/search query
        "query": query,

        "rewrite_count": 0,

        "context": [],

        "web_results": [],

        "answer": "",
    }


    final_state: Dict[str, Any] = (
        rag_agent.invoke(initial_state)
    )


    return {

        "answer":
            final_state.get(
                "answer",
                "",
            ),

        "sources":
            final_state.get(
                "context",
                [],
            ),

        "rewrite_count":
            final_state.get(
                "rewrite_count",
                0,
            ),

        "used_web":
            bool(
                final_state.get(
                    "web_results"
                )
            ),
    }