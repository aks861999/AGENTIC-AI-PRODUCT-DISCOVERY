from __future__ import annotations
import asyncio, logging
from typing import Annotated, Literal, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from api.search import hybrid_search
from api.cart import create_checkout_session, complete_checkout
from shared.config.settings import get_settings
import api.dependencies as deps
from api.nl_query import parse_nl_query, _resolve_attribute_filters


import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from contextvars import ContextVar
import time

logger = logging.getLogger(__name__)
settings = get_settings()


# Per-request trace accumulator — safe with asyncio, no shared state
_request_trace: ContextVar[list] = ContextVar('_request_trace', default=None)

def _trace(msg: str, type: str = 'step'):
    """Append a log line to the current request's trace, if one is active."""
    t = _request_trace.get(None)
    if t is not None:
        t.append({'msg': msg, 'type': type, 'ts': time.time()})


# ── State ──────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    last_results: list[dict]   # products from last search, kept for follow-ups
    last_query: Optional[str]


# ── In-memory session store (replace with Redis for production) ────────────────
_sessions: dict[str, ChatState] = {}













SEARCH_ONLY_PROMPT = """You are a helpful home improvement product search assistant.
You have access to one tool:
- **search_products**: call this for ANY request to find, get, show, recommend,
  or browse products, including phrases like "get me X", "any good Y", "I need Z",
  "show me brackets", "galvanized steel connector", etc.

Always call search_products whenever the user expresses any product need.
When search_products returns a JSON object, present the products to the user using
the "text" field only. Do not output raw JSON. The "products" array is internal
UI data and should not be mentioned.

Keep responses concise and product-focused. Focus on attributes like material, gauge, dimensions,
and brand.

When presenting results, list them clearly and ask if the user wants to search further.
"""



# ── Tools , ALL defined at module level, NEVER redefined inside a function ─────

@tool
async def search_products(query: str) -> str:
    """Search the product catalogue using natural language."""

    client = deps.get_os_client()
    model  = deps.get_model()
    if client is None or model is None:
        _trace('Search unavailable — client or model not initialised', 'error')
        return json.dumps({"text": "Search is temporarily unavailable.", "products": []})

    try:
        _trace(f'LangGraph → chat_node → search_products tool called', 'step')
        _trace(f'NL parse: invoking LLM on query="{query[:80]}"', 'llm')

        parsed = await parse_nl_query(query)
        search_q = parsed.semantic_query or query
        raw_attrs = parsed.filters.attributes if parsed.filters.attributes else None

        if raw_attrs:
            _trace(f'LLM parsed intent: product="{search_q}", attrs={json.dumps(raw_attrs)}', 'llm')
        else:
            _trace(f'LLM parsed intent: product="{search_q}", no attributes detected', 'llm')

        logger.info(
            "search_products: query=%r parsed_semantic=%r filters=%s",
            query, search_q, raw_attrs,
        )

        vector = await asyncio.to_thread(
            lambda: model.encode(search_q, normalize_embeddings=True).tolist()
        )
        _trace(f'Embedding: BAAI/bge-small-en-v1.5 → {len(vector)}-dim vector', 'dim')

        resolved_attrs = None
        if raw_attrs:
            # Temporarily capture SchemaLinker output by reading logs
            # We instrument _resolve_attribute_filters to also return schema_links
            resolved_attrs, schema_links = await _resolve_with_trace(
                client=client,
                index_name=settings.index_name,
                semantic_query=search_q,
                model=model,
                required_attributes=raw_attrs,
            )
            if schema_links:
                for lnk in schema_links:
                    score = lnk.get('score', 0)
                    color = 'filter' if score >= 0.70 else 'warn'
                    _trace(
                        f"SchemaLinker: '{lnk['from']}' → '{lnk['to']}' (sim={score:.3f})"
                        + ('' if score >= 0.70 else ' ⚠ below threshold, attr dropped'),
                        color
                    )
            if resolved_attrs:
                _trace(f'Resolved attrs: {json.dumps(resolved_attrs)}', 'filter')
            else:
                _trace('⚠ No attrs survived SchemaLinker — running unfiltered search', 'warn')
        else:
            _trace('⊘ No attribute filters — running unfiltered hybrid search', 'step')

        num_filters = len(resolved_attrs) if resolved_attrs else 0
        _trace(
            f'Building hybrid query: KNN×{settings.hybrid_knn_weight} + '
            f'BM25×{settings.hybrid_bm25_weight}, '
            f'{num_filters} attr filter(s), k={settings.knn_k}',
            'step'
        )
        _trace(f'POST → AWS OpenSearch ({settings.opensearch_host.split(".")[2] if "." in settings.opensearch_host else "cluster"})', 'step')

        result = await hybrid_search(
            client=client,
            query=search_q,
            query_vector=vector,
            index_name=settings.index_name,
            required_attributes=resolved_attrs,
            size=5,
        )

        if not result.hits:
            logger.info("search_products: 0 hits for query=%r resolved_attrs=%s", search_q, resolved_attrs)
            _trace(f'0 hits for query="{search_q}" — filters may have been relaxed or dropped', 'warn')
            return json.dumps({"text": "No products found for that query.", "products": []})

        _trace(f'{len(result.hits)} hit(s) returned for query="{search_q}"', 'result')
        for i, h in enumerate(result.hits[:3]):
            _trace(f'  [{i+1}] #{h.product_uid}  {h.title[:50]}', 'result')

        _trace(f'Gemini/Groq LLM generating natural language reply…', 'llm')

        products = []
        text_rows = []
        for i, h in enumerate(result.hits):
            attrs = ", ".join(f"{a.name}: {a.value}" for a in h.attributes[:3]) if h.attributes else "N/A"
            products.append({
                "id": h.id,
                "productuid": h.product_uid,
                "title": h.title,
                "productdescription": h.product_description[:150],
                "bulletpoints": h.bullet_points[:2],
                "attributes": [{"name": a.name, "value": a.value} for a in h.attributes[:5]],
            })
            text_rows.append(f"{i+1}. {h.title}\n   {h.product_description[:120]}...\n   Specs: {attrs}")

        logger.info("search_products: %d hits returned for query=%r", len(result.hits), search_q)
        return json.dumps({"text": "\n".join(text_rows), "products": products}, ensure_ascii=False)

    except Exception as exc:
        logger.warning("search_products tool error: %s", exc)
        _trace(f'✗ search_products error: {exc}', 'error')
        return json.dumps({"text": "Search failed, please try again.", "products": []})



async def _resolve_with_trace(
    client, index_name: str, semantic_query: str, model, required_attributes: dict
) -> tuple[dict | None, list[dict]]:
    """
    Calls _resolve_attribute_filters AND captures per-attribute schema links.
    Returns (resolved_attrs, schema_links).
    schema_links: [{"from": "colour", "to": "Color/Finish", "score": 0.838}, ...]
    """
    from api.nl_query import _schema_linker

    schema_links = []
    # Call SchemaLinker directly to get scores per attribute
    for raw_key in required_attributes:
        try:
            # _schema_linker.resolve_one returns (canonical_name, score) or None
            result = await _schema_linker.resolve_one(
                client, index_name, model, raw_key
            )
            if result:
                canonical, score = result
                schema_links.append({"from": raw_key, "to": canonical, "score": score})
            else:
                schema_links.append({"from": raw_key, "to": None, "score": 0.0})
        except Exception:
            # resolve_one may not exist — fall back to full resolve
            pass

    # Still call the original for the actual resolved dict
    resolved = await _resolve_attribute_filters(
        client=client,
        index_name=index_name,
        semantic_query=semantic_query,
        model=model,
        required_attributes=required_attributes,
    )
    return resolved, schema_links






_SEARCH_ONLY_TOOLS = [search_products]


def _build_gemini_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.3,
        google_api_key=settings.google_api_key,
        max_retries=settings.max_retries, 
    )

def _build_groq_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.groq_model,
        temperature=0.3,
        openai_api_key=settings.groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )


# ── Chat node ─────────────────────────────────────────────────────────────────
async def chat_node(
    state: ChatState,
    client: OpenSearch,
    model: SentenceTransformer,
) -> ChatState:
    
    tools = _SEARCH_ONLY_TOOLS                 


    full_messages = (
        [SystemMessage(content=SEARCH_ONLY_PROMPT)]  # ← was SYSTEM_PROMPT
        + list(state["messages"])
    )

    providers = []
    if settings.google_api_key:
        providers.append(("gemini", lambda: _build_gemini_llm()))
    if settings.groq_api_key:
        providers.append(("groq", lambda: _build_groq_llm()))

    #await asyncio.sleep(10)
    response = None
    for name, llm_factory in providers:
        try:
            llm      = llm_factory().bind_tools(tools)
            response = await asyncio.to_thread(llm.invoke, full_messages)
            logger.info("chat_node: response from %s", name)
            break
        except Exception as exc:
            logger.warning("chat_node: %s failed (%s) — trying next provider.", name, exc)

    if response is None:
        raise RuntimeError("All LLM providers failed in chat_node.")

    return {**state, "messages": response}

# ── Routing: go to tools if LLM made a tool call, otherwise end ───────────────
def _route(state: ChatState) -> Literal["tools", "end"]:
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# ── Graph factory (rebuilt per request , cheap, stateless compile) ─────────────
def _build_graph(client: OpenSearch, model: SentenceTransformer):
    async def _chat(state): return await chat_node(state, client, model)

    tool_node = ToolNode(_SEARCH_ONLY_TOOLS)         

    g = StateGraph(ChatState)
    g.add_node("chat", _chat)
    g.add_node("tools", tool_node)


    g.set_entry_point("chat")
    g.add_conditional_edges("chat", _route, {"tools": "tools", "end": END})
    g.add_edge("tools", "chat")   # after tool execution → back to chat

    return g.compile()


def _extract_text(content) -> str:
    """Normalize AIMessage content , handles both str and list-of-parts (Gemini 2.5)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
            if not (isinstance(part, dict) and part.get("type") == "tool_use")
        )
    return str(content)


async def chat_turn(
    session_id: str,
    user_message: str,
    client: OpenSearch,
    model: SentenceTransformer,
) -> dict:
    # Activate per-request trace
    trace: list[dict] = []
    token = _request_trace.set(trace)

    t_start = time.time()
    _trace(f'Session: {session_id}', 'dim')
    _trace(f'LangGraph graph invoked', 'step')

    try:
        if session_id not in _sessions:
            _sessions[session_id] = {
                "messages": [],
                "session_id": session_id,
                "last_results": [],
                "last_query": None,
            }

        state = _sessions[session_id]
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_message)]

        graph = _build_graph(client, model)
        new_state = await graph.ainvoke(state)

        last_results: list[dict] = []
        for msg in reversed(new_state["messages"]):
            if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "search_products":
                try:
                    payload = json.loads(msg.content)
                    if isinstance(payload, dict):
                        last_results = payload.get("products", [])
                except Exception:
                    pass
                break

        # Check if this was a conversational query (no tool was called)
        tool_was_called = any(
            isinstance(m, ToolMessage) for m in new_state["messages"][-6:]
        )
        if not tool_was_called:
            _trace('LangGraph: query classified as conversational — search skipped', 'step')
            _trace('⊘ OpenSearch search skipped', 'dim')
            _trace('LLM generating conversational reply…', 'llm')

        new_state["last_results"] = last_results
        _sessions[session_id] = new_state

        last_ai_text = next(
            (
                _extract_text(m.content)
                for m in reversed(new_state["messages"])
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
            ),
            "",
        )

        elapsed_ms = int((time.time() - t_start) * 1000)
        _trace(f'Done in {elapsed_ms / 1000:.1f}s · {len(last_results)} product(s)', 'dim')

        return {
            "session_id": session_id,
            "reply": last_ai_text,
            "results": last_results,
            "debug": {
                "trace": trace,           # full ordered event list
                "hits": len(last_results),
                "elapsed_ms": elapsed_ms,
            },
        }
    finally:
        _request_trace.reset(token)   # always clean up ContextVar