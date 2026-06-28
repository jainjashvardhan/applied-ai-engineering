"""
langgraph_03_memory.py

PURPOSE  : Add cross-invocation memory to a LangGraph agent using checkpointing.
           Build the HR Policy Assistant — directly replaces langchain_rag_with_memory.py
           and the deprecated RunnableWithMessageHistory pattern.

WHAT THIS ADDS over langgraph_02_agent.py:
  - MemorySaver checkpointer  → state persists ACROSS .invoke() calls
  - thread_id session scoping → each user/session gets an isolated history
  - search_hr_policies tool   → wraps your existing ChromaDB RAG system
  - Multi-turn conversation   → follow-up questions resolve correctly

ARCHITECTURE:
  manager question
       ↓
  chat(question, thread_id="manager_001")
       ↓
  hr_agent.invoke(state, config={"configurable": {"thread_id": ...}})
       ↓                              ↑
  LangGraph loads persisted state ────┘
       ↓
  [call_llm → execute_tools → call_llm → ...] → END
       ↓
  LangGraph saves updated state to MemorySaver
       ↓
  returns final answer

REPLACES:
  langchain_rag_with_memory.py — built on RunnableWithMessageHistory (deprecated)

RUN      : python langgraph_03_memory.py
REQUIRES : pip install langgraph langchain-google-genai
           rag_with_metadata.py in the same directory
           GOOGLE_API_KEY in .env
"""

import logging
import os
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# ── LOGGING SETUP ────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
LLM_MODEL      = "gpt-5.4-mini"
MAX_ITERATIONS = 5     # safety cap per turn — prevents runaway tool loops


# ── RAG TOOL SETUP ────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: This is where your previous work plugs into LangGraph.
# The search() and ensure_indexed() functions from rag_with_metadata.py become
# the implementation layer for the tool below.
#
# We import at runtime (inside the tool) rather than at module level to handle
# the case where rag_with_metadata.py isn't available gracefully.
# In production, this would be a direct import at the top of the file.

def _load_rag_search():
    """
    Load the search function from the existing RAG system.
    Falls back to a stub if rag_with_metadata.py is not available.
    """
    try:
        from rag_with_metadata import ensure_indexed, search as rag_search
        ensure_indexed()   # idempotent — only indexes if collection is empty
        logger.info("RAG system loaded from rag_with_metadata.py ✅")
        return rag_search
    except ImportError:
        logger.warning(
            "rag_with_metadata.py not found — using stub data. "
            "Place rag_with_metadata.py in the same directory to use live ChromaDB."
        )
        return None


# Run the load once at module startup, not per tool call
_rag_search_fn = _load_rag_search()

# ── STUB DATA ──────────────────────────────────────────────────────────────────
# Used only when ChromaDB is unavailable. Mirrors the structure
# that rag_with_metadata.py's search() returns.

STUB_POLICIES: dict[str, str] = {
    "leave": """
[Leave Policy v2.1]
Full-time employees receive 24 days of paid annual leave per year.
Leave must be requested at least 3 days in advance.
Unused leave carries forward up to 12 days maximum.
Sick Leave: 12 days per year, separate from annual leave.
""".strip(),
    "wfh": """
[Work From Home Policy v1.3]
Eligible employees may work from home up to 2 days per week.
WFH is not permitted in the first 90 days of employment.
Core hours: employees must be reachable 10am–5pm IST.
""".strip(),
    "expense": """
[Expense and Reimbursement Policy v3.0]
Travel expenses reimbursed within 7 working days.
All expenses above Rs 500 require a receipt.
Meal allowance during client visits: up to Rs 1500 per meal.
""".strip(),
}


def _stub_search(query: str, department: str | None = None) -> list[dict]:
    """Minimal stub when ChromaDB isn't available — returns hardcoded policy text."""
    query_lower = query.lower()
    results     = []

    keyword_map = {
        "leave": ["leave", "annual", "sick", "vacation", "carry"],
        "wfh"  : ["wfh", "work from home", "remote", "office"],
        "expense": ["expense", "reimburse", "travel", "meal", "receipt"],
    }

    for policy_key, keywords in keyword_map.items():
        if any(kw in query_lower for kw in keywords):
            results.append({
                "text"      : STUB_POLICIES[policy_key],
                "metadata"  : {"document_title": STUB_POLICIES[policy_key].split("\n")[0], "version": "stub"},
                "similarity": 0.85,
            })

    return results[:3]   # cap at 3 like real search


# ── TOOL DEFINITION ────────────────────────────────────────────────────────────
# The @tool decorator wraps this function and exposes:
#   - The function name          → tool name (LLM uses to call it)
#   - The docstring              → tool description (LLM reads to decide WHEN to call it)
#   - The type hints             → argument schema (LLM uses to structure the call)
#
# Write the docstring as instructions TO the LLM — be explicit about when to use it.

@tool
def search_hr_policies(query: str, department: str | None = None) -> str:
    """
    Search the company HR and Finance policy knowledge base.

    Use this whenever the user asks about:
    - Leave (annual leave, sick leave, carry forward, probation period)
    - Work from home (WFH eligibility, core hours, restrictions)
    - Expenses and reimbursements (travel, meal allowance, receipts)
    - Any internal company policy or procedure

    Do NOT use this for general knowledge questions (e.g. "what is GDPR") —
    only for questions about THIS company's internal policies.

    Args:
        query      : Natural language question or search keywords
        department : Optional. "hr" for HR policies, "finance" for finance/expense policies.
                     Leave as None to search all departments.

    Returns:
        Relevant policy text with source and version labels.
        If nothing is found, returns a message saying so.
    """
    logger.info(f"Tool: search_hr_policies | query='{query[:60]}' | dept={department}")

    # Use live ChromaDB if available, fall back to stub
    if _rag_search_fn is not None:
        filters = {"department": department} if department else None
        chunks  = _rag_search_fn(
            query       = query,
            top_k       = 3,
            filters     = filters,
            active_only = True
        )
    else:
        chunks = _stub_search(query, department)

    if not chunks:
        return "No relevant policy information found for this query."

    # Format chunks for LLM consumption — source label on each chunk
    # so the LLM can cite the correct document in its answer
    formatted = []
    for chunk in chunks:
        meta  = chunk["metadata"]
        title = meta.get("document_title", "Unknown Policy")
        ver   = meta.get("version", "?")
        formatted.append(f"[{title} v{ver}]\n{chunk['text']}")

    return "\n\n---\n\n".join(formatted)


TOOLS   = [search_hr_policies]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── STATE ─────────────────────────────────────────────────────────────────────
# CRITICAL: The same AgentState from langgraph_02_agent.py.
# Nothing about the state definition changes when you add checkpointing.
# add_messages reducer still appends — but now those accumulated messages
# also get persisted across .invoke() calls by the MemorySaver.

class AgentState(TypedDict):
    messages  : Annotated[list, add_messages]   # full conversation history (reducer + persisted)
    iteration : int                             # loop counter — reset each turn (no reducer)


# ── LLM SETUP ──────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model          = LLM_MODEL,
    api_key = os.getenv("OPENAI_API_KEY"),
)
llm_with_tools = llm.bind_tools(TOOLS)


# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an HR Policy Assistant for a company that uses GreyOrange's gStore platform.

You help employees understand company policies on leave, work from home, expenses, and procedures.

You have one tool available:
- search_hr_policies(query, department): searches the internal policy knowledge base

Instructions:
1. Search before answering — do not answer HR policy questions from general knowledge.
   Company policies are specific and internal data must be retrieved, not guessed.
2. Always cite the source document and version in your answer.
3. For follow-up questions ("what about X?" or "and sick leave?"), use the conversation
   history to understand context before searching.
4. If the answer is not in the knowledge base, say so clearly — do not fabricate.
5. Keep answers concise and employee-friendly.
"""


# ── NODE: CALL LLM ─────────────────────────────────────────────────────────────
def call_llm(state: AgentState) -> dict:
    """
    NODE: call_llm
    Identical logic to langgraph_02_agent.py.
    What's different: state["messages"] now contains the FULL conversation history
    across all previous turns (loaded by MemorySaver from the thread's checkpoint).
    The LLM therefore sees all prior turns automatically — no extra code needed.
    """
    logger.info(f"Node: call_llm | iteration={state['iteration']} | messages={len(state['messages'])}")

    if state["iteration"] >= MAX_ITERATIONS:
        logger.warning(f"MAX_ITERATIONS ({MAX_ITERATIONS}) reached — forcing exit")
        return {
            "messages"  : [AIMessage(content="I've hit the tool call limit for this turn. Please try a more specific question.")],
            "iteration" : state["iteration"],
        }

    # System prompt injected fresh per LLM call — NOT stored in state.
    # This means the full conversation history in state["messages"] is clean:
    # only HumanMessages, AIMessages, and ToolMessages — no SystemMessage noise.
    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    response: AIMessage = llm_with_tools.invoke(messages_for_llm)

    logger.info(
        f"LLM response | tool_calls={len(response.tool_calls)} | has_text={bool(response.content)}"
    )

    return {
        "messages"  : [response],         # reducer appends this to history
        "iteration" : state["iteration"] + 1,
    }


# ── NODE: EXECUTE TOOLS ────────────────────────────────────────────────────────
def execute_tools(state: AgentState) -> dict:
    """
    NODE: execute_tools
    Identical to langgraph_02_agent.py.
    Executes all tool calls from the last AIMessage.
    """
    logger.info("Node: execute_tools")

    last_message     = state["messages"][-1]
    tool_result_msgs : list[ToolMessage] = []

    for tool_call in last_message.tool_calls:
        tool_name    = tool_call["name"]
        tool_args    = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info(f"Executing | tool={tool_name} | args={tool_args}")

        if tool_name not in TOOL_MAP:
            result_content = f"Error: unknown tool '{tool_name}'"
        else:
            try:
                result_content = TOOL_MAP[tool_name].invoke(tool_args)
            except Exception as e:
                logger.error(f"Tool failed | tool={tool_name} | error={e}")
                result_content = f"Tool error: {str(e)}"

        tool_result_msgs.append(
            ToolMessage(content=str(result_content), tool_call_id=tool_call_id)
        )

    return {"messages": tool_result_msgs}


# ── ROUTING FUNCTION ───────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_tools"
    return END


# ── GRAPH CONSTRUCTION ─────────────────────────────────────────────────────────
def build_hr_agent() -> StateGraph:
    """
    Build the HR Policy Assistant graph.

    THE ONE CHANGE FROM LANGGRAPH_02_AGENT.PY:
    compile(checkpointer=MemorySaver())

    Every node and edge is identical. Memory is infrastructure that lives at the
    compilation layer — your business logic nodes know nothing about it.

    MEMORY TYPES (dev → production progression):
      MemorySaver        → in-memory Python dict, lost on process restart (dev only)
      SqliteSaver        → on-disk SQLite, survives restarts, single server
      PostgresSaver      → cloud database, survives restarts, multi-server, production

    To switch from MemorySaver to SqliteSaver:
      from langgraph.checkpoint.sqlite import SqliteSaver
      with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
          agent = builder.compile(checkpointer=checkpointer)

    You would make this swap for production deployment with zero changes to
    your nodes, edges, or routing logic.
    """
    builder = StateGraph(AgentState)

    builder.add_node("call_llm",      call_llm)
    builder.add_node("execute_tools", execute_tools)

    builder.set_entry_point("call_llm")

    builder.add_conditional_edges(
        "call_llm",
        should_continue,
        {"execute_tools": "execute_tools", END: END}
    )

    builder.add_edge("execute_tools", "call_llm")

    # ── THE KEY CHANGE: add checkpointer at compile time ──────────────────────
    # This single argument transforms a stateless graph into a stateful one.
    # MemorySaver is instantiated once and shared across all sessions.
    # Sessions are differentiated by thread_id in the config dict.
    return builder.compile(checkpointer=MemorySaver())


# ── MODULE-LEVEL AGENT ─────────────────────────────────────────────────────────
hr_agent = build_hr_agent()


# ── SESSION INTERFACE ─────────────────────────────────────────────────────────
# DESIGN DECISION: thread_id is the session key.
# Format: any string. Convention: "{user_id}_{context}" or just "{user_id}".
# Same thread_id = same memory. Different thread_id = isolated memory.
# This maps directly to session_id in your old langchain_rag_with_memory.py:
#   OLD: session_store[session_id] = InMemoryChatMessageHistory()
#   NEW: thread_id in config — MemorySaver handles the storage

def chat(question: str, thread_id: str) -> str:
    """
    Send a message to the HR assistant within a named session.

    Passing iteration=0 explicitly resets the loop counter each turn.
    Without this, the iteration count from the PREVIOUS turn (persisted in
    state) would carry over and eat into the current turn's MAX_ITERATIONS.
    No reducer on iteration → last write wins → 0 overrides stored value.

    Args:
        question  : Employee's question in natural language
        thread_id : Session identifier — same ID = same conversation history

    Returns:
        Agent's answer as a plain string
    """
    config = {"configurable": {"thread_id": thread_id}}

    # iteration=0 resets the per-turn loop counter (last-write-wins, no reducer)
    # Messages-only input is sufficient for subsequent turns — LangGraph loads
    # the rest of the state from the checkpoint automatically.
    initial_state: AgentState = {
        "messages"  : [HumanMessage(content=question)],
        "iteration" : 0,
    }

    logger.info(f"chat() | thread={thread_id} | question='{question[:70]}'")
    start_time = time.time()

    result = hr_agent.invoke(initial_state, config=config)

    final_message = result["messages"][-1]
    answer        = final_message.content if hasattr(final_message, "content") else str(final_message)
    latency_ms    = int((time.time() - start_time) * 1000)

    # Summarise tool calls made in this turn (for observability)
    tools_called = [
        tc["name"]
        for msg in result["messages"]
        if isinstance(msg, AIMessage)
        for tc in msg.tool_calls
    ]

    logger.info(
        f"chat() done | thread={thread_id} | tools={tools_called} | "
        f"total_msgs={len(result['messages'])} | latency_ms={latency_ms}"
    )

    return answer


def get_conversation_history(thread_id: str) -> list[dict]:
    """
    Inspect the full conversation history stored for a thread.
    Useful for debugging — shows exactly what the agent sees on the next call.

    This uses LangGraph's get_state() API to read from the checkpointer
    without executing the graph.

    Returns:
        List of {role, content} dicts for each message in history
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state_snapshot = hr_agent.get_state(config)
        messages       = state_snapshot.values.get("messages", [])
    except Exception:
        return []

    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user",      "content": msg.content})
        elif isinstance(msg, AIMessage):
            if msg.content:
                history.append({"role": "assistant", "content": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    history.append({"role": "tool_call", "content": f"{tc['name']}({tc['args']})"})
        elif isinstance(msg, ToolMessage):
            history.append({"role": "tool_result", "content": str(msg.content)[:120] + "..."})

    return history


# ── TESTS ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 72)
    print("LANGGRAPH 03 — HR Policy Assistant with Memory")
    print("=" * 72)

    # ── TEST 1: Multi-turn conversation ──────────────────────────────────────
    # Tests whether follow-up questions resolve correctly across turns.
    # Turn 2's "what about sick leave?" only makes sense with Turn 1's context.
    # Turn 3's "can I combine both?" requires both previous turns in context.

    print("\n── TEST 1: Multi-turn conversation (same thread_id) ──")
    SESSION_A = "employee_jj_001"

    turn1_q = "How many paid leave days do I get per year?"
    print(f"\nTurn 1 | Q: {turn1_q}")
    print(f"       | A: {chat(turn1_q, SESSION_A)}\n")

    turn2_q = "And what about sick leave — is it separate?"
    print(f"Turn 2 | Q: {turn2_q}")
    print(f"       | A: {chat(turn2_q, SESSION_A)}\n")

    turn3_q = "Can I carry forward unused sick leave like annual leave?"
    print(f"Turn 3 | Q: {turn3_q}")
    print(f"       | A: {chat(turn3_q, SESSION_A)}\n")

    # ── INSPECT STATE ─────────────────────────────────────────────────────────
    # This shows exactly what the agent has accumulated in memory for SESSION_A.
    # In production this would be your debugging and audit endpoint.
    print("── Memory state for SESSION_A ──")
    history = get_conversation_history(SESSION_A)
    for i, entry in enumerate(history):
        role    = entry["role"]
        content = entry["content"][:100].replace("\n", " ")
        print(f"  [{i:02d}] {role:<14} | {content}")

    # ── TEST 2: Session isolation ─────────────────────────────────────────────
    # A different thread_id gets a completely fresh history.
    # SESSION_B should not know anything SESSION_A discussed.

    print("\n── TEST 2: Session isolation (different thread_id) ──")
    SESSION_B = "employee_pinky_002"

    isolation_q = "What is our WFH policy for new joiners?"
    print(f"\nSession B | Q: {isolation_q}")
    print(f"          | A: {chat(isolation_q, SESSION_B)}\n")

    # Session B's history should only contain its own messages
    history_b = get_conversation_history(SESSION_B)
    print(f"Session A has {len(get_conversation_history(SESSION_A))} messages in memory")
    print(f"Session B has {len(history_b)} messages in memory")
    print("Sessions are fully isolated ✅" if len(history_b) < len(get_conversation_history(SESSION_A)) else "")

    # ── TEST 3: Resume a session ──────────────────────────────────────────────
    # Simulates closing the app and reopening — MemorySaver persists in memory
    # for the process lifetime. SqliteSaver/PostgresSaver would survive restarts.

    print("\n── TEST 3: Resume SESSION_A (memory persists) ──")
    resume_q = "Given what we discussed, when is the earliest I can take WFH?"
    print(f"\nTurn 4 | Q: {resume_q}")
    print(f"       | A: {chat(resume_q, SESSION_A)}")
