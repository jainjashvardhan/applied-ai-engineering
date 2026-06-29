"""
langgraph_04_multiagent.py

PURPOSE  : Build a Supervisor + Worker multi-agent system.

PROBLEM SOLVED:
  A single generalist agent with many tools makes poor routing decisions,
  runs tasks sequentially that could be parallel, and conflates domain-specific
  knowledge across a single bloated system prompt.
  Multi-agent systems solve this with specialisation and parallel execution.

WHAT THIS DEMONSTRATES:
  1. Supervisor routing via structured LLM output (typed, no string parsing)
  2. Specialist workers — each with focused tools and domain knowledge
  3. Send API — parallel dispatch of multiple workers
  4. Aggregation reducer — collecting results from parallel workers safely
  5. How a specialist analytics worker resolves "London" → "HM-LON-042"
     (the exact problem observed in langgraph_02_agent.py)

ARCHITECTURE:
    Question
        ↓
    supervisor  (routes via structured output)
        ↓
    dispatch_workers  (returns list[Send] for parallel execution)
        ↓              ↓
   hr_worker     analytics_worker      ← run in parallel via Send API
        ↓              ↓
    worker_outputs (merged by reducer — both results preserved)
        ↓
    synthesize    (combines results into final answer)
        ↓
       END

SPECIALIST vs GENERALIST:
  Generalist (langgraph_02_agent.py) — one agent, all tools, one system prompt.
    → asks for store IDs when given city names (doesn't know the mapping)
  Specialist analytics_worker         — focused tools, domain knowledge baked in.
    → resolves "London" → "HM-LON-042" internally, calls tools without asking

RUN      : python langgraph_04_multiagent.py
REQUIRES : pip install langgraph langchain-openai langchain-google-genai pydantic
           OPENAI_API_KEY and GEMINI_API_KEY in .env
"""

import json
import logging
import os
import time
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
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


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-5.4-mini"
GEMINI_MODEL = "gemini-2.5-flash"


# ── DOMAIN KNOWLEDGE: Store Name → ID Resolution ──────────────────────────────
# DOMAIN KNOWLEDGE: This is specialisation in action.
# The generalist agent in langgraph_02_agent.py asked for store IDs because
# it had no mapping — it was instructed to ask when unsure.
# A specialist analytics worker bakes in domain context it always needs.
# In production this lookup would be a database call to a store registry.

STORE_NAME_TO_ID: dict[str, str] = {
    "london"  : "HM-LON-042",
    "lon"     : "HM-LON-042",
    "new york": "HM-NYC-018",
    "nyc"     : "HM-NYC-018",
    "ny"      : "HM-NYC-018",
    "berlin"  : "HM-BER-007",
    "ber"     : "HM-BER-007",
}

def resolve_store_ids(text: str) -> list[str]:
    """
    Extract store IDs from a question that may use city names or direct IDs.
    Returns a deduplicated list of store IDs found in the text.
    """
    import re
    text_lower = text.lower()
    found: list[str] = []

    # Resolve city name aliases
    for name, store_id in STORE_NAME_TO_ID.items():
        if name in text_lower and store_id not in found:
            found.append(store_id)

    # Also capture direct IDs written in the question (e.g. "HM-LON-042")
    for direct_id in re.findall(r"HM-[A-Z]{3}-\d{3}", text):
        if direct_id not in found:
            found.append(direct_id)

    return found


# ── FAKE DATA (same shape as langgraph_02_agent.py) ───────────────────────────
INVENTORY_DATA: dict[str, dict[str, int]] = {
    "HM-LON-042": {"polo_shirt_m": 45, "jeans_w32": 12, "summer_dress": 3},
    "HM-NYC-018": {"polo_shirt_m": 0,  "jeans_w32": 89, "summer_dress": 23},
    "HM-BER-007": {"polo_shirt_m": 12, "jeans_w32": 0,  "summer_dress": 78},
}

PERFORMANCE_DATA: dict[str, dict[str, float]] = {
    "HM-LON-042": {"daily_sales_usd": 12400.0, "conversion_rate": 0.082, "avg_basket_usd": 45.2},
    "HM-NYC-018": {"daily_sales_usd": 8900.0,  "conversion_rate": 0.061, "avg_basket_usd": 38.7},
    "HM-BER-007": {"daily_sales_usd": 15600.0, "conversion_rate": 0.094, "avg_basket_usd": 52.1},
}

HR_POLICIES: dict[str, str] = {
    "leave": (
        "Full-time employees receive 24 days paid annual leave per year. "
        "Minimum 3 days advance notice required. Carry forward max 12 days. "
        "Sick leave: 12 days/year (separate from annual leave). "
        "Probationary period (first 90 days): 12 pro-rated days, no carry forward."
    ),
    "wfh": (
        "WFH permitted up to 2 days/week. Manager approval required by Thursday for next week. "
        "NOT permitted in first 90 days of employment. "
        "Employees on PIP need explicit HR approval. Core hours: 10am–5pm IST."
    ),
    "expense": (
        "Travel expenses reimbursed within 7 working days of receipt submission. "
        "All expenses above Rs 500 require receipt. "
        "Meal allowance: Rs 1500/meal for client visits. "
        "International travel requires pre-approval from Finance and department head."
    ),
}


# ── STATE DEFINITION ──────────────────────────────────────────────────────────
# CRITICAL: worker_outputs uses a reducer.
# Without reducer: Worker B's result overwrites Worker A's (last-write-wins).
# With reducer: both results are appended — synthesizer sees outputs from ALL workers.
# This is the multi-agent equivalent of the handler_log reducer in langgraph_01_basics.py.

def merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
    """Reducer: append worker outputs rather than overwrite."""
    return existing + new


class AgentState(TypedDict):
    question      : str
    route         : str                               # "hr" | "analytics" | "both" | "general"
    worker_outputs: Annotated[list[dict], merge_results]  # parallel results accumulated here
    final_answer  : str
    messages      : Annotated[list, add_messages]     # conversation history (for multi-turn extension)


# ── LLM SETUP ─────────────────────────────────────────────────────────────────
# PYTHON CONCEPT: .with_fallbacks([...]) wraps the primary LLM so that if it
# raises any exception (rate limit, network error, service down), LangChain
# automatically retries with the fallback model transparently.

_openai = ChatOpenAI(
    model   = OPENAI_MODEL,
    api_key = os.getenv("OPENAI_API_KEY"),
)

_gemini = ChatGoogleGenerativeAI(
    model          = GEMINI_MODEL,
    google_api_key = os.getenv("GEMINI_API_KEY"),   # GEMINI_API_KEY — not GOOGLE_API_KEY
)

# General LLM: OpenAI primary, Gemini fallback
llm = _openai.with_fallbacks([_gemini])

# ── STRUCTURED OUTPUT LLM ─────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: .with_structured_output() forces the LLM to return a Pydantic
# object instead of free text. This is critical for supervisor routing:
#   Without it: LLM returns "I think this is an HR question" — fragile string parsing
#   With it:    LLM returns RoutingDecision(route="hr", reasoning="...") — type-safe
# Under the hood: uses OpenAI's function calling / response format JSON mode.

class RoutingDecision(BaseModel):
    route    : Literal["hr", "analytics", "both", "general"]
    reasoning: str   # brief explanation — useful for debugging and observability

supervisor_llm = _openai.with_structured_output(RoutingDecision)


# ── SYSTEM PROMPTS ─────────────────────────────────────────────────────────────
# Each agent gets a focused system prompt — not one bloated prompt for everything.
# This is why specialised agents make better decisions than a generalist.

SUPERVISOR_PROMPT = """You are a routing supervisor for a GreyOrange gStore analytics platform.
Your ONLY job is to classify incoming questions and route them correctly.

Route to:
- "hr"        : questions about leave, WFH policy, expenses, HR procedures
- "analytics" : questions about store performance, inventory, sales, conversion rates, stock levels
- "both"      : questions that require BOTH HR policy AND analytics data simultaneously
- "general"   : everything else (greetings, clarifications, questions outside scope)

Be decisive. When in doubt between "analytics" and "both", choose "both".
"""

HR_WORKER_PROMPT = """You are an HR Policy Specialist for a retail company using GreyOrange's gStore platform.
You answer questions about internal company policies based ONLY on the context provided.
Always cite the policy name. If information isn't in the context, say so clearly.
Be concise — employees need quick, actionable answers.
"""

ANALYTICS_WORKER_PROMPT = """You are an Analytics Specialist for GreyOrange's gStore platform.
You have access to real-time inventory and performance data for all stores.
You know that: London = HM-LON-042, New York = HM-NYC-018, Berlin = HM-BER-007.
When comparing stores, always include specific numbers and highlight the key differentiator.
Be direct — store managers need clear, data-backed answers.
"""

SYNTHESIZER_PROMPT = """You are synthesizing the outputs of multiple specialist agents into one coherent answer.
Combine the provided specialist responses into a single, well-structured response.
Do not lose any key information. Avoid redundancy. Keep it concise.
"""


# ── HELPER: call LLM with system prompt + question ────────────────────────────
def _generate(system_prompt: str, context: str, question: str) -> str:
    """Single LLM call: system prompt + context + question → answer string."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    return response.content


# ── NODE: SUPERVISOR ──────────────────────────────────────────────────────────
def supervisor(state: AgentState) -> dict:
    """
    NODE: supervisor
    Routes the question to the correct specialist(s).

    Uses .with_structured_output() so the LLM returns a RoutingDecision Pydantic
    object — typed, validated, no string parsing.

    Writes: route (str), messages (adds decision to history for observability)
    """
    logger.info(f"Node: supervisor | question='{state['question'][:60]}'")

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=state["question"]),
    ]

    decision: RoutingDecision = supervisor_llm.invoke(messages)

    logger.info(f"Supervisor decision | route={decision.route} | reason={decision.reasoning}")

    return {
        "route"         : decision.route,
        "worker_outputs": [],    # reset outputs at the start of each run
    }


# ── NODE: HR WORKER ───────────────────────────────────────────────────────────
def hr_worker(state: AgentState) -> dict:
    """
    NODE: hr_worker
    Specialist in HR policies. Deterministic tool use — always searches
    relevant policy based on keywords in the question.

    PATTERN: Deterministic worker
    This worker always retrieves context first, then generates an answer.
    It doesn't use LLM tool calling to DECIDE whether to search —
    it just searches, because that's its job.

    CONTRAST with agentic worker (Session 5):
    An agentic worker would itself be a compiled LangGraph subgraph with
    its own ReAct loop, making its own decisions about which tools to call.
    """
    logger.info(f"Node: hr_worker | question='{state['question'][:60]}'")
    start = time.time()

    # ── Deterministic context retrieval ────────────────────────────────────────
    question_lower = state["question"].lower()
    relevant_policies: list[str] = []

    keyword_map = {
        "leave"  : ["leave", "annual", "sick", "holiday", "vacation", "carry forward"],
        "wfh"    : ["wfh", "work from home", "remote", "office"],
        "expense": ["expense", "reimburse", "travel", "meal", "receipt", "allowance"],
    }

    for policy_key, keywords in keyword_map.items():
        if any(kw in question_lower for kw in keywords):
            relevant_policies.append(HR_POLICIES[policy_key])

    if not relevant_policies:
        # Fall back to all policies if no keywords matched
        relevant_policies = list(HR_POLICIES.values())

    context = "\n\n".join(relevant_policies)

    # ── Generate answer from retrieved context ──────────────────────────────
    answer = _generate(HR_WORKER_PROMPT, context, state["question"])

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"hr_worker done | latency_ms={elapsed_ms}")

    # CRITICAL: Return a LIST for worker_outputs — the reducer appends this list
    # to whatever is already in worker_outputs from other workers.
    # If you return a dict, the reducer tries to append a dict to list[dict] — wrong.
    return {
        "worker_outputs": [{"source": "hr", "answer": answer, "latency_ms": elapsed_ms}]
    }


# ── NODE: ANALYTICS WORKER ────────────────────────────────────────────────────
def analytics_worker(state: AgentState) -> dict:
    """
    NODE: analytics_worker
    Specialist in store performance and inventory data.
    Resolves city names to store IDs — no need to ask the user.

    This directly addresses the observed limitation in langgraph_02_agent.py:
    the generalist asked for store IDs when given "London" and "Berlin"
    because it lacked domain knowledge. This specialist has it baked in.
    """
    logger.info(f"Node: analytics_worker | question='{state['question'][:60]}'")
    start = time.time()

    # ── Resolve store IDs from question ────────────────────────────────────────
    store_ids = resolve_store_ids(state["question"])
    logger.info(f"analytics_worker resolved store_ids={store_ids}")

    if not store_ids:
        # If still no IDs found, query all stores
        store_ids = list(INVENTORY_DATA.keys())

    # ── Retrieve data for each resolved store ──────────────────────────────────
    data_sections: list[str] = []

    for store_id in store_ids:
        inv_data  = INVENTORY_DATA.get(store_id, {})
        perf_data = PERFORMANCE_DATA.get(store_id, {})

        if not inv_data and not perf_data:
            data_sections.append(f"{store_id}: No data available.")
            continue

        data_sections.append(
            f"{store_id}:\n"
            f"  Performance : {json.dumps(perf_data)}\n"
            f"  Inventory   : {json.dumps(inv_data)}"
        )

    context = "\n\n".join(data_sections)

    # ── Generate answer from retrieved data ────────────────────────────────────
    answer = _generate(ANALYTICS_WORKER_PROMPT, context, state["question"])

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"analytics_worker done | stores={store_ids} | latency_ms={elapsed_ms}")

    return {
        "worker_outputs": [{"source": "analytics", "answer": answer, "latency_ms": elapsed_ms}]
    }


# ── NODE: GENERAL HANDLER ─────────────────────────────────────────────────────
def general_handler(state: AgentState) -> dict:
    """
    NODE: general_handler
    Handles questions that don't need specialist agents — greetings,
    out-of-scope questions, or simple clarifications.
    Direct LLM call, no tools or context.
    """
    logger.info("Node: general_handler")

    response = llm.invoke([HumanMessage(content=state["question"])])

    return {
        "worker_outputs": [{"source": "general", "answer": response.content, "latency_ms": 0}]
    }


# ── NODE: SYNTHESIZE ──────────────────────────────────────────────────────────
def synthesize(state: AgentState) -> dict:
    """
    NODE: synthesize
    Collects all worker_outputs and combines them into a single final answer.

    For single-worker routes ("hr", "analytics", "general"):
      Returns the worker's answer directly — no synthesis LLM call needed.
    For "both" routes:
      Makes one additional LLM call to combine the two specialist answers.

    PRODUCTION CONSIDERATION: An extra LLM call for synthesis adds latency and cost.
    Evaluate whether rule-based synthesis (simple concatenation with headers) is
    sufficient for your use case before defaulting to LLM synthesis.
    """
    logger.info(f"Node: synthesize | worker_outputs={len(state['worker_outputs'])}")

    outputs = state["worker_outputs"]

    if not outputs:
        return {"final_answer": "No response generated."}

    # Single worker — return its answer directly, no synthesis overhead
    if len(outputs) == 1:
        return {"final_answer": outputs[0]["answer"]}

    # Multiple workers — synthesize into one coherent response
    combined_context = "\n\n---\n\n".join(
        f"[{out['source'].upper()} SPECIALIST]\n{out['answer']}"
        for out in outputs
    )

    final = _generate(
        system_prompt = SYNTHESIZER_PROMPT,
        context       = combined_context,
        question      = state["question"],
    )

    return {"final_answer": final}


# ── ROUTING FUNCTION ──────────────────────────────────────────────────────────
# CRITICAL CONCEPT: The Send API
#
# Old pattern (single destination):
#   def route(state) -> str:
#       return "hr_worker"                  # returns ONE node name
#
# New pattern (parallel dispatch):
#   def route(state) -> list[Send]:
#       return [Send("a", state), Send("b", state)]   # dispatches to BOTH
#
# LangGraph sees the list[Send] return and runs both targets.
# Each Send gets its own copy of state at that point.
# Both write to worker_outputs; the reducer merges their results.
#
# For single-route cases, returning a list[Send] with one element still works —
# it's just sequential under the hood. The Send API is backward compatible.

def dispatch_workers(
    state: AgentState,
) -> list[Send] | str:
    """
    Routing function: reads supervisor's route decision, dispatches workers.
    Called via add_conditional_edges after the supervisor node.

    Returns list[Send] for parallel execution, or str for single-node routing.
    Both return types are valid for add_conditional_edges.
    """
    route = state["route"]
    logger.info(f"Routing | route={route}")

    if route == "both":
        # Dispatch BOTH workers simultaneously — parallel execution
        # Total latency = max(hr_time, analytics_time) instead of hr_time + analytics_time
        return [
            Send("hr_worker",        state),
            Send("analytics_worker", state),
        ]
    elif route == "hr":
        return [Send("hr_worker", state)]
    elif route == "analytics":
        return [Send("analytics_worker", state)]
    else:
        return "general_handler"   # string routing for non-Send path


# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────
def build_supervisor_agent() -> StateGraph:
    """
    Builds the multi-agent supervisor graph.

    Topology:
        START
          ↓
        supervisor  ← classifies question, sets route
          ↓ (dispatch_workers conditional edge)
          ├── "both"      → [Send("hr_worker"), Send("analytics_worker")]  PARALLEL
          ├── "hr"        → [Send("hr_worker")]
          ├── "analytics" → [Send("analytics_worker")]
          └── other       → "general_handler"
               ↓               ↓              ↓
          hr_worker    analytics_worker  general_handler 
               ↓               ↓              ↓
               └───────────────┴──────────────┘
                               ↓
                           synthesize
                               ↓
                             END
    """
    builder = StateGraph(AgentState)

    builder.add_node("supervisor",        supervisor)
    builder.add_node("hr_worker",         hr_worker)
    builder.add_node("analytics_worker",  analytics_worker)
    builder.add_node("general_handler",   general_handler)
    builder.add_node("synthesize",        synthesize)

    builder.set_entry_point("supervisor")

    # Conditional edge from supervisor — can return list[Send] or str
    # When it returns list[Send], LangGraph dispatches all in parallel.
    # When it returns a str, LangGraph routes to that node name directly.
    builder.add_conditional_edges(
        "supervisor",
        dispatch_workers,
        # PYTHON CONCEPT: The mapping dict here is ONLY needed when you return
        # node name strings. When you return Send objects, the graph doesn't need
        # this mapping — the Send objects carry the destination node name directly.
        # Including a partial mapping is fine; LangGraph handles both cases.
        {
            "general_handler": "general_handler",
        }
    )

    # All workers and the general handler converge at synthesize
    builder.add_edge("hr_worker",        "synthesize")
    builder.add_edge("analytics_worker", "synthesize")
    builder.add_edge("general_handler",  "synthesize")
    builder.add_edge("synthesize",        END)

    return builder.compile()


# ── MODULE-LEVEL AGENT ─────────────────────────────────────────────────────────
supervisor_agent = build_supervisor_agent()


# ── PUBLIC INTERFACE ───────────────────────────────────────────────────────────
def ask(question: str) -> dict:
    """
    Ask the multi-agent system a question.

    Returns:
        dict with final_answer, route taken, worker latencies, total latency
    """
    start = time.time()

    initial_state: AgentState = {
        "question"      : question,
        "route"         : "",
        "worker_outputs": [],
        "final_answer"  : "",
        "messages"      : [HumanMessage(content=question)],
    }

    result     = supervisor_agent.invoke(initial_state)
    total_ms   = int((time.time() - start) * 1000)

    return {
        "question"      : question,
        "route"         : result["route"],
        "answer"        : result["final_answer"],
        "worker_sources": [o["source"] for o in result["worker_outputs"]],
        "worker_latencies": {o["source"]: o.get("latency_ms", 0) for o in result["worker_outputs"]},
        "total_latency_ms": total_ms,
    }


# ── TESTS ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_questions = [
        # Route: hr — single worker
        "How many annual leave days do I get, and can I carry them forward?",

        # Route: analytics — single worker, WITH city names (not store IDs)
        # This is the question that broke the generalist agent in langgraph_02_agent.py
        "Which store has better conversion rate — London or Berlin or India?",

        # Route: analytics — multi-store comparison resolved from city names
        "Compare inventory levels across all three stores",

        # Route: both — requires HR policy AND store data simultaneously
        "I'm a new joiner in Berlin. Can I WFH and how is the store performing?",

        # Route: general — out of scope
        "What is the capital of France?",
    ]

    print("\n" + "=" * 72)
    print("LANGGRAPH 04 — Supervisor + Worker Multi-Agent System")
    print("=" * 72)

    for question in test_questions:
        print(f"\nQ: {question}")

        result = ask(question)

        print(f"Route    : {result['route']}  →  workers: {result['worker_sources']}")
        print(f"Answer   : {result['answer']}")
        print(f"Latency  : total={result['total_latency_ms']}ms | workers={result['worker_latencies']}")
        print("-" * 72)
