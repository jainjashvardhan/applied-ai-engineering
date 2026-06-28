"""
langgraph_05a_subgraphs.py

PURPOSE  : Replace deterministic worker nodes with compiled LangGraph subgraphs.
           Each worker is now an independent agent with its own:
           - State definition (isolated from the parent)
           - Nodes and conditional routing
           - LLM selected for its domain strengths

MULTI-LLM ROLE ASSIGNMENT:
  Claude Sonnet 4.6    → Supervisor (routing decisions + final synthesis)
  OpenAI gpt-5.4-mini  → HR Worker (policy retrieval + structured reasoning)
  Gemini 2.5 Flash     → Analytics Worker (data analysis + comparison)

WHAT CHANGES FROM langgraph_04_multiagent.py:
  Session 4 workers: plain Python functions — one deterministic code path
  Session 5A workers: compiled StateGraphs — own state, own nodes, own routing

KEY PATTERN — Wrapper Node:
  The parent graph cannot invoke a subgraph directly via edges if the state types
  differ. A wrapper node handles the translation:
    1. Extract what the subgraph needs from parent state
    2. Invoke the compiled subgraph
    3. Map subgraph output back to parent state format
  This makes state boundaries explicit and each layer independently testable.

GRAPH STRUCTURE:

  Parent (Supervisor — Claude):
    START → classify → dispatch_workers (Send API, parallel)
               ↓                ↓
          hr_wrapper      analytics_wrapper
          (invokes         (invokes Analytics
           HR subgraph)     subgraph)
               ↓                ↓
             worker_outputs (reducer merges both)
               ↓
            synthesize (Claude) → END

  HR Worker Subgraph (OpenAI):
    START → retrieve_context → route_coverage
                ↓ (found)         ↓ (not found)
          generate_answer      hr_fallback
                ↓                 ↓
                └────────────────→ END

  Analytics Worker Subgraph (Gemini):
    START → resolve_and_fetch → generate_insight → END

RUN      : python langgraph_05a_subgraphs.py
REQUIRES : pip install langgraph langchain-openai langchain-google-genai langchain-anthropic
           OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY in .env
"""

import json
import logging
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel

load_dotenv()

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── LLM INSTANCES ─────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: LLM assignment by role, not by default.
# This is a real cost + capability optimisation pattern used in production:
# - Expensive/capable model for tasks requiring deep reasoning (supervisor, synthesis)
# - Efficient model for high-volume structured tasks (policy retrieval + answer)
# - Fast model for data-heavy analytical tasks (store metrics + comparison)

claude  = ChatAnthropic(
    model      = "claude-sonnet-4-6",
    api_key    = os.getenv("ANTHROPIC_API_KEY"),
)

openai_llm = ChatOpenAI(
    model   = "gpt-5.4-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
)

gemini = ChatOpenAI(
    model   = "gpt-5.4-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
)


# ── DATA ──────────────────────────────────────────────────────────────────────
STORE_NAME_TO_ID: dict[str, str] = {
    "london": "HM-LON-042", "lon": "HM-LON-042",
    "new york": "HM-NYC-018", "nyc": "HM-NYC-018", "ny": "HM-NYC-018",
    "berlin": "HM-BER-007", "ber": "HM-BER-007",
}

INVENTORY_DATA: dict[str, dict] = {
    "HM-LON-042": {"polo_shirt_m": 45, "jeans_w32": 12, "summer_dress": 3},
    "HM-NYC-018": {"polo_shirt_m": 0,  "jeans_w32": 89, "summer_dress": 23},
    "HM-BER-007": {"polo_shirt_m": 12, "jeans_w32": 0,  "summer_dress": 78},
}

PERFORMANCE_DATA: dict[str, dict] = {
    "HM-LON-042": {"daily_sales_usd": 12400, "conversion_rate": 0.082, "avg_basket_usd": 45.2},
    "HM-NYC-018": {"daily_sales_usd": 8900,  "conversion_rate": 0.061, "avg_basket_usd": 38.7},
    "HM-BER-007": {"daily_sales_usd": 15600, "conversion_rate": 0.094, "avg_basket_usd": 52.1},
}

HR_POLICIES: dict[str, str] = {
    "leave"  : "24 days paid annual leave/year. Min 3 days advance notice. Carry forward max 12 days. Sick leave: 12 days/year separate. Probation (first 90 days): 12 pro-rated days, no carry forward.",
    "wfh"    : "WFH up to 2 days/week. Manager approval required. NOT permitted first 90 days. Employees on PIP need HR approval. Core hours: 10am–5pm IST.",
    "expense": "Travel reimbursed within 7 working days. Receipts required above Rs 500. Meal allowance Rs 1500/meal for client visits. International travel needs Finance + dept head pre-approval.",
}


# ═════════════════════════════════════════════════════════════════════════════
# HR WORKER SUBGRAPH  —  powered by OpenAI gpt-5.4-mini
# ═════════════════════════════════════════════════════════════════════════════

class HRWorkerState(TypedDict):
    """
    Isolated state for the HR worker subgraph.
    Parent graph knows nothing about these fields except question (input) and answer (output).
    This isolation is what makes the subgraph independently testable and deployable.
    """
    question   : str
    context    : str   # retrieved policy text
    has_context: bool  # routing signal — did we find relevant policies?
    answer     : str


def hr_retrieve_context(state: HRWorkerState) -> dict:
    """
    NODE (HR subgraph): retrieve_context
    Keyword search across HR policies — deterministic, no LLM needed for retrieval.
    Sets context and has_context routing signal.
    """
    logger.info(f"[HR Subgraph] retrieve_context | q='{state['question'][:50]}'")

    question_lower = state["question"].lower()
    found: list[str] = []

    keyword_map = {
        "leave"  : ["leave", "annual", "sick", "vacation", "carry", "holiday", "days off"],
        "wfh"    : ["wfh", "work from home", "remote", "office", "hybrid"],
        "expense": ["expense", "reimburse", "travel", "meal", "receipt", "allowance"],
    }

    for policy_key, keywords in keyword_map.items():
        if any(kw in question_lower for kw in keywords):
            found.append(HR_POLICIES[policy_key])

    if not found:
        # Broad fallback — return all policies when no keywords match
        # Better to over-retrieve than to return empty context to the LLM
        found = list(HR_POLICIES.values())

    context     = "\n\n".join(found)
    has_context = bool(found)

    return {"context": context, "has_context": has_context}


def hr_generate_answer(state: HRWorkerState) -> dict:
    """
    NODE (HR subgraph): generate_answer
    OpenAI generates a grounded answer from retrieved policy context.
    """
    logger.info("[HR Subgraph] generate_answer")

    response = openai_llm.invoke([
        SystemMessage(content=(
            "You are an HR Policy Specialist. Answer using ONLY the policy context provided. "
            "Cite the policy name. Be concise — employees need quick, actionable answers. "
            "If the answer is not in the context, say so clearly."
        )),
        HumanMessage(content=f"Policy context:\n{state['context']}\n\nQuestion: {state['question']}"),
    ])

    return {"answer": response.content}


def hr_fallback(state: HRWorkerState) -> dict:
    """
    NODE (HR subgraph): fallback
    Reached when no policy context found — returns a clean "not available" answer
    rather than hallucinating. No LLM call needed.
    """
    logger.info("[HR Subgraph] fallback — no relevant policies found")

    return {
        "answer": (
            f"I couldn't find specific policy information for: '{state['question']}'. "
            "Please contact HR directly or check the internal policy portal."
        )
    }


def hr_route_coverage(state: HRWorkerState) -> str:
    """Routing function (HR subgraph): found context → generate, no context → fallback."""
    return "hr_generate_answer" if state["has_context"] else "hr_fallback"


def build_hr_subgraph() -> StateGraph:
    """
    Builds and compiles the HR worker subgraph.

    Topology:
        START → retrieve_context → route_coverage
                                       ↓ (found)     ↓ (not found)
                                  generate_answer    fallback
                                       ↓                ↓
                                       └────────────────→ END

    SUBGRAPH DESIGN PRINCIPLE:
    This graph is independently runnable:
      hr_graph.invoke({"question": "How many leave days?"})
    No knowledge of the parent supervisor required.
    This is the testability benefit of the subgraph pattern.
    """
    builder = StateGraph(HRWorkerState)

    builder.add_node("retrieve_context", hr_retrieve_context)
    builder.add_node("hr_generate_answer", hr_generate_answer)
    builder.add_node("hr_fallback",        hr_fallback)

    builder.set_entry_point("retrieve_context")

    builder.add_conditional_edges(
        "retrieve_context",
        hr_route_coverage,
        {"hr_generate_answer": "hr_generate_answer", "hr_fallback": "hr_fallback"}
    )

    builder.add_edge("hr_generate_answer", END)
    builder.add_edge("hr_fallback",        END)

    return builder.compile()


# Compiled once at module level — reused across all invocations
hr_subgraph = build_hr_subgraph()


# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS WORKER SUBGRAPH  —  powered by Gemini 2.5 Flash
# ═════════════════════════════════════════════════════════════════════════════

class AnalyticsWorkerState(TypedDict):
    """Isolated state for the Analytics worker subgraph."""
    question  : str
    store_ids : list[str]   # resolved from city names or direct IDs
    raw_data  : str         # JSON blob of inventory + performance data
    answer    : str


def analytics_resolve_and_fetch(state: AnalyticsWorkerState) -> dict:
    """
    NODE (Analytics subgraph): resolve_and_fetch
    Resolves city names → store IDs, then fetches inventory + performance data.
    Pure Python — no LLM call. Two responsibilities in one node here because
    they're tightly coupled (resolution output directly feeds the fetch).
    """
    import re
    logger.info(f"[Analytics Subgraph] resolve_and_fetch | q='{state['question'][:50]}'")

    text_lower = state["question"].lower()
    found_ids: list[str] = []

    for name, sid in STORE_NAME_TO_ID.items():
        if name in text_lower and sid not in found_ids:
            found_ids.append(sid)

    for direct_id in re.findall(r"HM-[A-Z]{3}-\d{3}", state["question"]):
        if direct_id not in found_ids:
            found_ids.append(direct_id)

    # Fall back to all stores if nothing resolved
    if not found_ids:
        found_ids = list(INVENTORY_DATA.keys())

    # Fetch data for all resolved stores
    data_blocks: list[str] = []
    for sid in found_ids:
        inv  = INVENTORY_DATA.get(sid, {})
        perf = PERFORMANCE_DATA.get(sid, {})
        data_blocks.append(
            f"{sid}:\n"
            f"  performance : {json.dumps(perf)}\n"
            f"  inventory   : {json.dumps(inv)}"
        )

    logger.info(f"[Analytics Subgraph] resolved store_ids={found_ids}")

    return {
        "store_ids": found_ids,
        "raw_data" : "\n\n".join(data_blocks),
    }


def analytics_generate_insight(state: AnalyticsWorkerState) -> dict:
    """
    NODE (Analytics subgraph): generate_insight
    Gemini analyzes the fetched store data and generates an insight.
    Gemini is well-suited to this: structured data → natural language summary.
    """
    logger.info(f"[Analytics Subgraph] generate_insight | stores={state['store_ids']}")

    response = gemini.invoke([
        SystemMessage(content=(
            "You are a Retail Analytics Specialist for GreyOrange's gStore platform. "
            "Analyse the provided store data and give a direct, data-backed answer. "
            "Always include specific numbers. Highlight the key differentiator when comparing stores. "
            "Be concise — store managers need actionable insights, not essays."
        )),
        HumanMessage(content=f"Store data:\n{state['raw_data']}\n\nQuestion: {state['question']}"),
    ])

    return {"answer": response.content}


def build_analytics_subgraph() -> StateGraph:
    """
    Builds and compiles the Analytics worker subgraph.

    Topology:
        START → resolve_and_fetch → generate_insight → END

    Simpler than HR subgraph — analytics always has data (stores are known).
    The complexity is in insight generation, not routing.
    """
    builder = StateGraph(AnalyticsWorkerState)

    builder.add_node("resolve_and_fetch",    analytics_resolve_and_fetch)
    builder.add_node("generate_insight",     analytics_generate_insight)

    builder.set_entry_point("resolve_and_fetch")
    builder.add_edge("resolve_and_fetch", "generate_insight")
    builder.add_edge("generate_insight",  END)

    return builder.compile()


analytics_subgraph = build_analytics_subgraph()


# ═════════════════════════════════════════════════════════════════════════════
# SUPERVISOR GRAPH  —  powered by Claude Sonnet 4.6
# ═════════════════════════════════════════════════════════════════════════════

def merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
    """Reducer: append worker outputs from parallel subgraphs."""
    return existing + new


class SupervisorState(TypedDict):
    question      : str
    route         : str                                    # set by classify node
    worker_outputs: Annotated[list[dict], merge_results]   # reducer for parallel results
    final_answer  : str
    messages      : Annotated[list, add_messages]


class RoutingDecision(BaseModel):
    route    : Literal["hr", "analytics", "both", "general"]
    reasoning: str


# Structured output on Claude — same pattern as OpenAI but on Anthropic
supervisor_router = claude.with_structured_output(RoutingDecision)


def classify(state: SupervisorState) -> dict:
    """
    NODE (Supervisor): classify
    Claude routes the question. Resets worker_outputs at start of each run.

    DESIGN: Claude is chosen here because routing requires nuanced understanding
    of question intent — distinguishing "how many leave days" (HR) from
    "how is London performing" (analytics) from "I'm a new Berlin joiner — WFH
    policy AND store numbers?" (both). Claude's instruction-following is strong here.
    """
    logger.info(f"Node: classify | q='{state['question'][:60]}'")

    decision: RoutingDecision = supervisor_router.invoke([
        SystemMessage(content=(
            "You are a routing supervisor for a GreyOrange gStore analytics platform.\n"
            "Route to:\n"
            "  'hr'        — leave, WFH, expenses, HR procedures\n"
            "  'analytics' — store performance, inventory, sales, conversion rates\n"
            "  'both'      — requires HR policy AND store analytics data simultaneously\n"
            "  'general'   — out of scope, greetings, clarifications\n"
            "Be decisive. Prefer single-worker routes unless both are genuinely required."
        )),
        HumanMessage(content=state["question"]),
    ])

    logger.info(f"Claude routing decision | route={decision.route} | reason={decision.reasoning}")

    return {
        "route"         : decision.route,
        "worker_outputs": [],   # reset — critical for multi-turn correctness
    }


def hr_wrapper(state: SupervisorState) -> dict:
    """
    NODE (Supervisor): hr_wrapper
    Wrapper node — translates between SupervisorState and HRWorkerState.

    PATTERN: The wrapper is the interface contract between parent and subgraph.
    Parent graph knows: "I give it a question, I get back an answer."
    Parent graph does NOT know: how HR subgraph retrieves context, what LLM it uses,
    how its internal routing works.
    This is why subgraphs are independently deployable and testable.
    """
    logger.info("Node: hr_wrapper → invoking HR subgraph (OpenAI)")

    result = hr_subgraph.invoke({
        "question"   : state["question"],
        "context"    : "",
        "has_context": False,
        "answer"     : "",
    })

    return {
        "worker_outputs": [{"source": "hr (OpenAI gpt-5.4-mini)", "answer": result["answer"]}]
    }


def analytics_wrapper(state: SupervisorState) -> dict:
    """
    NODE (Supervisor): analytics_wrapper
    Wrapper node — translates between SupervisorState and AnalyticsWorkerState.
    """
    logger.info("Node: analytics_wrapper → invoking Analytics subgraph (Gemini)")

    result = analytics_subgraph.invoke({
        "question" : state["question"],
        "store_ids": [],
        "raw_data" : "",
        "answer"   : "",
    })

    return {
        "worker_outputs": [{"source": "analytics (Gemini 2.5 Flash)", "answer": result["answer"]}]
    }


def general_handler(state: SupervisorState) -> dict:
    """NODE (Supervisor): general_handler — Claude handles out-of-scope questions directly."""
    logger.info("Node: general_handler")
    response = claude.invoke([HumanMessage(content=state["question"])])
    return {
        "worker_outputs": [{"source": "general (Claude Sonnet 4.6)", "answer": response.content}]
    }


def synthesize(state: SupervisorState) -> dict:
    """
    NODE (Supervisor): synthesize
    Claude synthesizes worker outputs into the final answer.

    Single worker → return directly (no synthesis call needed, saves cost).
    Multiple workers → Claude combines them into one coherent response.

    DESIGN: Claude is used for synthesis because combining two specialist outputs
    — one from OpenAI (HR) and one from Gemini (analytics) — requires the ability
    to understand both and produce a coherent unified response.
    """
    logger.info(f"Node: synthesize | workers={len(state['worker_outputs'])}")

    outputs = state["worker_outputs"]

    if not outputs:
        return {"final_answer": "No response generated."}

    if len(outputs) == 1:
        # Single worker — return directly, no extra LLM call
        return {"final_answer": outputs[0]["answer"]}

    # Multiple workers — Claude synthesises
    combined = "\n\n---\n\n".join(
        f"[{out['source'].upper()}]\n{out['answer']}"
        for out in outputs
    )

    response = claude.invoke([
        SystemMessage(content=(
            "Combine the two specialist responses below into one clear, well-structured answer. "
            "Preserve all key information from both. Remove redundancy. "
            "Do not add information that isn't in either response."
        )),
        HumanMessage(content=f"Specialist outputs:\n{combined}\n\nOriginal question: {state['question']}"),
    ])

    return {"final_answer": response.content}


def dispatch_workers(state: SupervisorState) -> list[Send] | str:
    """
    Routing function: reads Claude's route decision, dispatches via Send API.
    Same pattern as langgraph_04_multiagent.py — Send API unchanged.
    The only difference: the nodes being dispatched are now subgraph wrappers.
    """
    route = state["route"]
    logger.info(f"Dispatching | route={route}")

    if route == "both":
        return [Send("hr_wrapper", state), Send("analytics_wrapper", state)]
    elif route == "hr":
        return [Send("hr_wrapper", state)]
    elif route == "analytics":
        return [Send("analytics_wrapper", state)]
    else:
        return "general_handler"


def build_supervisor() -> StateGraph:
    builder = StateGraph(SupervisorState)

    builder.add_node("classify",           classify)
    builder.add_node("hr_wrapper",         hr_wrapper)
    builder.add_node("analytics_wrapper",  analytics_wrapper)
    builder.add_node("general_handler",    general_handler)
    builder.add_node("synthesize",         synthesize)

    builder.set_entry_point("classify")

    builder.add_conditional_edges(
        "classify",
        dispatch_workers,
        {"general_handler": "general_handler"}
    )

    builder.add_edge("hr_wrapper",        "synthesize")
    builder.add_edge("analytics_wrapper", "synthesize")
    builder.add_edge("general_handler",   "synthesize")
    builder.add_edge("synthesize",        END)

    return builder.compile()


supervisor = build_supervisor()


# ── PUBLIC INTERFACE ──────────────────────────────────────────────────────────
def ask(question: str) -> dict:
    import time
    start = time.time()

    result = supervisor.invoke({
        "question"      : question,
        "route"         : "",
        "worker_outputs": [],
        "final_answer"  : "",
        "messages"      : [HumanMessage(content=question)],
    })

    return {
        "question": question,
        "route"   : result["route"],
        "answer"  : result["final_answer"],
        "sources" : [o["source"] for o in result["worker_outputs"]],
        "latency_ms": int((time.time() - start) * 1000),
    }


# ── TESTS ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    questions = [
        # HR only — OpenAI answers
        "What's the WFH policy for someone still in their probation period?",

        # Analytics only — Gemini answers — city names, not store IDs
        "Which store has better conversion rate — London or Berlin?",

        # Both — Claude routes, OpenAI + Gemini work in parallel, Claude synthesises
        "I'm a new joiner starting at the Berlin store. What's the WFH policy for me, and how is the store performing?",

        # General — Claude answers directly
        "What does RFID stand for and how does it work in retail settings?",
    ]

    print("\n" + "=" * 72)
    print("LANGGRAPH 05A — Subgraph Workers + Multi-LLM Architecture")
    print(f"  Supervisor / Synthesis : Claude Sonnet 4.6")
    print(f"  HR Worker              : OpenAI gpt-5.4-mini")
    print(f"  Analytics Worker       : Gemini 2.5 Flash")
    print("=" * 72)

    for q in questions:
        print(f"\nQ: {q}")
        result = ask(q)
        print(f"Route   : {result['route']}")
        print(f"Sources : {result['sources']}")
        print(f"Answer  : {result['answer']}")
        print(f"Latency : {result['latency_ms']}ms")
        print("-" * 72)
