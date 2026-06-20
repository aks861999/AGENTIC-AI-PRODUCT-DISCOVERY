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

logger = logging.getLogger(__name__)
settings = get_settings()


# ── State ──────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    last_results: list[dict]   # products from last search, kept for follow-ups
    last_query: Optional[str]


# ── In-memory session store (replace with Redis for production) ────────────────
_sessions: dict[str, ChatState] = {}


# ── LLM factory ───────────────────────────────────────────────────────────────
def _get_llm() -> ChatGoogleGenerativeAI:
    """Return LLM instance. Tries Gemini first, falls back to Groq."""
    # 1. Gemini (primary)
    if settings.google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info("Chatbot LLM: using Google Gemini (%s)", settings.llm_model)
            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                temperature=0.3,
                google_api_key=settings.google_api_key,
            )
        except Exception as exc:
            logger.warning("Chatbot: Gemini init failed (%s) — trying Groq.", exc)

    # 2. Groq (fallback)
    if settings.groq_api_key:
        try:
            from langchain_openai import ChatOpenAI
            logger.info("Chatbot LLM: using Groq (%s)", settings.groq_model)
            return ChatOpenAI(
                model=settings.groq_model,
                temperature=0.3,
                openai_api_key=settings.groq_api_key,
                openai_api_base="https://api.groq.com/openai/v1",
            )
        except Exception as exc:
            logger.warning("Chatbot: Groq init failed (%s).", exc)

    raise RuntimeError(
        "No LLM provider available. Set GOOGLE_API_KEY or GROQ_API_KEY in .env"
    )




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

FULL_COMMERCE_PROMPT = """You are a helpful product search assistant.
You have access to three tools:
- **search_products**: call this for ANY request to find, get, show, recommend,
  or browse products , including phrases like "get me X", "any good Y", "I need Z",
  "something for winter", "show me cream", etc.
- **ucp_add_to_cart**: call this when a user wants to add a specific product to their cart.
- **ucp_checkout**: call this when the user wants to pay or complete their order.

Always call search_products first whenever the user expresses any product need.
When search_products returns a JSON object, present the products to the user using
the "text" field only. Do not output raw JSON. The "products" array is internal
UI data and should not be mentioned.

When presenting results, list them clearly and ask if the user wants to add any to their cart.
When a user says "add the first one" or "buy it", call ucp_add_to_cart with the correct details.
Keep responses concise and product-focused.

If the price shows as N/A or is unknown, pass price_cents=0.
Never refuse to add due to a missing price; use price_cents=0 as a placeholder.
"""

def _active_system_prompt() -> str:
    return FULL_COMMERCE_PROMPT if settings.agent_mode == "full_commerce" else SEARCH_ONLY_PROMPT



# ── Tools , ALL defined at module level, NEVER redefined inside a function ─────

@tool
async def search_products(query: str) -> str:
    """Search the product catalogue using natural language. Call this whenever the
    user wants to find, get, show, browse, discover, or get recommendations for
    any product , including casual phrases like 'get me X' or 'something for Y'."""

    client = deps.get_os_client()
    model  = deps.get_model()
    if client is None or model is None:
        return json.dumps({"text": "Search is temporarily unavailable.", "products": []})

    try:
        # Extract filters, boost terms, and semantic core from the NL query
        parsed = await parse_nl_query(query)
        search_q = parsed.semantic_query or query


        logger.info(
            "search_products: query=%r parsed_semantic=%r filters=%s",
            query, search_q, parsed.filters.attributes,
        )

        vector = await asyncio.to_thread(
            lambda: model.encode(search_q, normalize_embeddings=True).tolist()
        )
        raw_attrs = parsed.filters.attributes if parsed.filters.attributes else None

        resolved_attrs = await _resolve_attribute_filters(
            client=client,
            index_name=settings.index_name,
            semantic_query=search_q,
            model=model,
            required_attributes=raw_attrs,
        ) if raw_attrs else None

        logger.info("DEBUG chatbot: raw_attrs=%s | resolved_attrs=%s", raw_attrs, resolved_attrs)


        result = await hybrid_search(
            client=client,
            query=search_q,
            query_vector=vector,
            index_name=settings.index_name,
            required_attributes=resolved_attrs,
            size=5,
        )
        if not result.hits:
            logger.info("search_products: 0 hits returned for query=%r resolved_attrs=%s", search_q, resolved_attrs)
            return json.dumps({"text": "No products found for that query.", "products": []})

        products = []
        text_rows = []
        for i, h in enumerate(result.hits):
            # Top 3 attributes as a readable string
            attrs = ", ".join(
                f"{a.name}: {a.value}" for a in h.attributes[:3]
            ) if h.attributes else "N/A"

            products.append({
                "id": h.id,
                "product_uid": h.product_uid,
                "title": h.title,
                "product_description": h.product_description[:150],
                "bullet_points": h.bullet_points[:2],
                "attributes": [{"name": a.name, "value": a.value} for a in h.attributes[:5]],
                "primary_image_url": None,   # not in Home Depot dataset
            })
            text_rows.append(
                f"{i+1}. {h.title}\n"
                f"   {h.product_description[:120]}...\n"
                f"   Specs: {attrs}"
            )

        logger.info("search_products: %d hits returned for query=%r", len(result.hits), search_q)

        return json.dumps({
            "text": "\n".join(text_rows),
            "products": products
        }, ensure_ascii=False)

    except Exception as exc:
        logger.warning("search_products tool error: %s", exc)
        return json.dumps({"text": "Search failed, please try again.", "products": []})


@tool
def ucp_add_to_cart(product_id: str, title: str, price_cents: int = 0) -> str:
    """Add a product to a UCP checkout session.
    price_cents is the price in cents (e.g. 1999 = €19.99).
    """
    session = create_checkout_session(product_id, title, price_cents)
    total = sum(li.item.price * li.quantity for li in session.line_items) / 100
    return (
        f"✅ Added '{title}' to cart.\n"
        f"UCP Session ID: {session.id}\n"
        f"Cart total: €{total:.2f}"
    )


@tool
def ucp_checkout(session_id: str) -> str:
    """Complete a UCP checkout session and place the order."""
    order = complete_checkout(session_id)
    return (
        f"🎉 Order confirmed!\n"
        f"Order ID : {order.id}\n"
        f"Status   : {order.status}\n"
        f"Total    : €{order.total:.2f}"
    )



_SEARCH_ONLY_TOOLS = [search_products]
_FULL_COMMERCE_TOOLS = [search_products, ucp_add_to_cart, ucp_checkout]

def _active_tools() -> list:
    return _FULL_COMMERCE_TOOLS if settings.agent_mode == "full_commerce" else _SEARCH_ONLY_TOOLS


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
    
    tools = _active_tools()                      
    llm = _get_llm().bind_tools(tools)


    full_messages = (
        [SystemMessage(content=_active_system_prompt())]  # ← was SYSTEM_PROMPT
        + list(state["messages"])
    )

    providers = []
    if settings.google_api_key:
        providers.append(("gemini", lambda: _build_gemini_llm()))
    if settings.groq_api_key:
        providers.append(("groq", lambda: _build_groq_llm()))

    await asyncio.sleep(10)
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

    tool_node = ToolNode(_active_tools())         

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

    # ── NEW: extract product cards from the last search tool call ──
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

    return {
        "session_id": session_id,
        "reply": last_ai_text,
        "results": last_results,
    }