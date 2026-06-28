"""
langgraph_02_agent.py

PURPOSE  : Build your first LLM-powered agent with tool use and a ReAct loop.
           This is the fundamental pattern behind every production AI agent.

WHAT THIS DEMONSTRATES:
  - add_messages reducer  → conversation history accumulates correctly
  - LLM node              → decides: answer directly OR call a tool
  - Tool execution node   → runs the requested tools, returns results
  - Conditional routing   → loops back if tools were called, exits if done
  - Why this cannot be built with LCEL

USE CASE : gStore Analytics Assistant
           Store managers ask questions → agent decides to query inventory or
           performance data → synthesizes results → gives actionable answer.

RUN      : python langgraph_02_agent.py
REQUIRES : pip install langgraph langchain-google-genai
           GOOGLE_API_KEY in .env
"""

import json
import logging
import os
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# ── LOGGING SETUP ───────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── CONSTANTS ────────────────────────────────────────────────────────────────
LLM_MODEL  = "gemini-2.5-flash"
MAX_ITERATIONS = 5   # production safety: cap agent loops to prevent runaway costs


# ── FAKE DATA STORE ──────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: In production, these tool implementations would query
# BigQuery aggregate tables you built for gStore. For this lesson the data
# is hardcoded — the tool-calling mechanics are identical.

INVENTORY_DATA: dict[str, dict[str, int]] = {
    "HM-LON-042": {"polo_shirt_m": 45, "jeans_w32": 12, "summer_dress": 3},
    "HM-NYC-018": {"polo_shirt_m": 0,  "jeans_w32": 89, "summer_dress": 23},
    "HM-BER-007": {"polo_shirt_m": 12, "jeans_w32": 0,  "summer_dress": 78},
}

PERFORMANCE_DATA: dict[str, dict[str, float]] = {
    "HM-LON-042": {"daily_sales_usd": 12400, "conversion_rate": 0.082, "avg_basket_usd": 45.2},
    "HM-NYC-018": {"daily_sales_usd": 8900,  "conversion_rate": 0.061, "avg_basket_usd": 38.7},
    "HM-BER-007": {"daily_sales_usd": 15600, "conversion_rate": 0.094, "avg_basket_usd": 52.1},
}


# ── TOOL DEFINITIONS ─────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Tools are functions the LLM can call to get real data.
# The @tool decorator wraps the function and exposes its docstring + type hints
# to the LLM as a description. This docstring IS the prompt for when to use it.
# Write docstrings that tell the LLM exactly when and how to call this tool.

@tool
def get_inventory(store_id: str) -> str:
    """
    Retrieve current inventory counts for a given gStore location.
    Use this when the user asks about stock levels, item counts, availability,
    or anything related to physical inventory in a store.

    Args:
        store_id: Store identifier (e.g. "HM-LON-042", "HM-NYC-018", "HM-BER-007")

    Returns:
        JSON string with SKU names mapped to current quantity counts.
        Returns an error string if the store_id is not found.
    """
    data = INVENTORY_DATA.get(store_id)
    if not data:
        return f"No inventory data found for store '{store_id}'. Valid IDs: {list(INVENTORY_DATA.keys())}"
    return json.dumps({"store_id": store_id, "inventory": data, "unit": "items"})


@tool
def get_store_performance(store_id: str) -> str:
    """
    Retrieve sales performance metrics for a given gStore location.
    Use this when the user asks about sales figures, revenue, conversion rates,
    basket size, or any financial/performance KPIs for a store.

    Args:
        store_id: Store identifier (e.g. "HM-LON-042", "HM-NYC-018", "HM-BER-007")

    Returns:
        JSON string with daily_sales_usd, conversion_rate, and avg_basket_usd.
        Returns an error string if the store_id is not found.
    """
    data = PERFORMANCE_DATA.get(store_id)
    if not data:
        return f"No performance data found for store '{store_id}'. Valid IDs: {list(PERFORMANCE_DATA.keys())}"
    return json.dumps({"store_id": store_id, "performance": data})


# Collect tools in a list — used to bind to the LLM and to build the execution map
TOOLS = [get_inventory, get_store_performance]

# TOOL_MAP: name → callable. Used in execute_tools node to dispatch calls.
# Built once at module level — no repeated dict construction per request.
TOOL_MAP: dict[str, object] = {t.name: t for t in TOOLS}


# ── STATE DEFINITION ─────────────────────────────────────────────────────────
# CRITICAL CONCEPT: add_messages reducer
#
# Without reducer:
#   Node A sets messages = [msg1, msg2]
#   Node B sets messages = [msg3]
#   Final state: messages = [msg3]    ← WRONG: history lost
#
# With add_messages reducer:
#   Node A sets messages = [msg1, msg2]
#   Node B returns messages = [msg3]
#   Final state: messages = [msg1, msg2, msg3]  ← CORRECT: appended
#
# This is why agents can "remember" what they've seen — the reducer keeps
# all messages from all previous turns in the list.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history, accumulates
    iteration: int                             # tracks loop count — safety guard


# ── LLM SETUP ─────────────────────────────────────────────────────────────────
# bind_tools() does two things:
# 1. Adds tool schemas to the LLM's context (so it knows what tools exist)
# 2. Configures the LLM to output tool_calls in a structured format when needed

# llm = ChatAnthropic(
#     model       = "claude-sonnet-4-20250514",
#     api_key     = os.getenv("ANTHROPIC_API_KEY"),
# )
llm = ChatOpenAI(
    model       = "gpt-5.4-mini",
    api_key     = os.getenv("OPENAI_API_KEY"),
)

llm_with_tools = llm.bind_tools(TOOLS)


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# DESIGN DECISION: System prompt is NOT stored in state.
# It is injected fresh on every LLM call. This is correct because:
# - System prompt is static — no reason to carry it through state transitions
# - Storing it in state means it would appear in history and add noise
# - Injecting it lets you change it without touching state schema

SYSTEM_PROMPT = """You are the gStore Analytics Assistant for GreyOrange's retail intelligence platform.
You help store managers and operations teams understand inventory and sales performance.

You have access to two tools:
- get_inventory(store_id): retrieves current stock counts by SKU
- get_store_performance(store_id): retrieves daily sales, conversion rate, and basket size

Rules:
- Call tools to get real data. Never fabricate numbers.
- After receiving tool results, synthesize the data into a clear, actionable answer.
- If a store_id is needed but not provided, ask for it before calling tools.
- Be concise. Store managers need fast, actionable answers — not paragraphs.
- If the question has no relevance to inventory or sales, answer from general knowledge.
"""


# ── NODE: CALL LLM ────────────────────────────────────────────────────────────
def call_llm(state: AgentState) -> dict:
    """
    NODE: call_llm

    Sends the full conversation history to the LLM (with system prompt prepended).
    The LLM responds with one of two things:
      1. A direct text answer (no tool needed) → agent will EXIT after this
      2. A tool call request                   → agent will execute tools and LOOP BACK

    This is the decision-making center of the agent.
    It never knows in advance which path the LLM will take.
    """
    logger.info(f"Node: call_llm | iteration={state['iteration']}")

    # Safety: don't loop forever
    if state["iteration"] >= MAX_ITERATIONS:
        logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached — forcing exit")
        return {
            "messages"  : [AIMessage(content=f"I've reached the maximum number of tool calls ({MAX_ITERATIONS}). Please try a more specific question.")],
            "iteration" : state["iteration"],
        }

    # Inject system prompt fresh on every call — do NOT store it in state
    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    response: AIMessage = llm_with_tools.invoke(messages_for_llm)

    logger.info(
        f"LLM response | "
        f"tool_calls={len(response.tool_calls)} | "
        f"has_text={bool(response.content)}"
    )

    # add_messages reducer will APPEND this AIMessage to state["messages"]
    # (not replace the list — that's the whole point of the reducer)
    return {
        "messages"  : [response],
        "iteration" : state["iteration"] + 1,
    }


# ── NODE: EXECUTE TOOLS ───────────────────────────────────────────────────────
def execute_tools(state: AgentState) -> dict:
    """
    NODE: execute_tools

    Reads the last AIMessage in state (which has tool_calls populated).
    Executes each requested tool and packages results as ToolMessages.

    ToolMessages are what the LLM reads as "tool results" on the next call.
    The tool_call_id links each ToolMessage to the specific call that triggered it.
    This matters for parallel tool calls — the LLM needs to match results to calls.

    After this node runs, execution returns to call_llm to process the results.
    """
    logger.info("Node: execute_tools")

    # The last message is always an AIMessage when we reach this node
    # (because only call_llm precedes execute_tools and our router ensures tool_calls exist)
    last_message: AIMessage = state["messages"][-1]

    tool_results: list[ToolMessage] = []

    for tool_call in last_message.tool_calls:
        tool_name    = tool_call["name"]
        tool_args    = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info(f"Executing tool | name={tool_name} | args={tool_args}")

        if tool_name not in TOOL_MAP:
            result_content = f"Error: unknown tool '{tool_name}'"
        else:
            try:
                # .invoke() accepts the args dict — LangChain handles argument mapping
                result_content = TOOL_MAP[tool_name].invoke(tool_args)
            except Exception as e:
                logger.error(f"Tool execution failed | tool={tool_name} | error={e}")
                result_content = f"Error executing {tool_name}: {str(e)}"

        # ToolMessage MUST include the tool_call_id so the LLM can correlate
        # this result with the original call (critical for multi-tool calls)
        tool_results.append(
            ToolMessage(
                content      = str(result_content),
                tool_call_id = tool_call_id,
            )
        )

    logger.info(f"Tools executed | count={len(tool_results)}")

    # add_messages reducer will APPEND these ToolMessages to the conversation history
    return {"messages": tool_results}


# ── ROUTING FUNCTION ──────────────────────────────────────────────────────────
# This is the router that creates the ReAct loop.
#
# ReAct pattern: Reason → Act → Observe → Reason → Act → Observe → ... → Answer
#
# "Reason"  = call_llm (LLM thinks about what to do)
# "Act"     = execute_tools (agent takes action in the world)
# "Observe" = the ToolMessages fed back to call_llm
#
# The loop only exits when the LLM reasons "I have enough — here is the answer"
# and produces no tool calls.

def should_continue(state: AgentState) -> str:
    """
    Routing function called after call_llm.

    Checks if the last message has tool_calls:
    - Yes → continue to execute_tools (loop continues)
    - No  → exit to END (agent has its final answer)
    """
    last_message = state["messages"][-1]

    # isinstance check is more robust than hasattr — AIMessage always has tool_calls attr
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Router: tool calls present → execute_tools (loop continues)")
        return "execute_tools"

    logger.info("Router: no tool calls → END (agent done)")
    return END


# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────
def build_gstore_agent() -> StateGraph:
    """
    Build the gStore analytics agent graph.

    Graph topology (contains a CYCLE — impossible in LCEL):

        START
          ↓
        call_llm  ←─────────────────────┐
          ↓ (should_continue)           │
          ├── tool_calls? → execute_tools ─┘
          └── no tools?  → END

    The cycle (call_llm → execute_tools → call_llm) is what makes this an AGENT.
    Each loop iteration adds tool results to the conversation history.
    The agent exits when the LLM decides it has enough information to answer.

    PRODUCTION CONSIDERATION: The MAX_ITERATIONS guard in call_llm prevents
    runaway loops when the LLM keeps requesting tools without making progress.
    In production you'd also track cost per invocation and alert on high-iteration calls.
    """
    builder = StateGraph(AgentState)

    builder.add_node("call_llm",      call_llm)
    builder.add_node("execute_tools", execute_tools)

    builder.set_entry_point("call_llm")

    # Conditional edge: loop or exit based on whether LLM requested tools
    builder.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "execute_tools": "execute_tools",   # loop back
            END            : END,               # exit
        }
    )

    # After tool execution, unconditionally return to LLM to process results
    builder.add_edge("execute_tools", "call_llm")

    return builder.compile()


# ── MODULE-LEVEL AGENT INSTANCE ──────────────────────────────────────────────
gstore_agent = build_gstore_agent()


# ── PUBLIC INTERFACE ──────────────────────────────────────────────────────────
def ask_agent(question: str) -> dict:
    """
    Ask the gStore analytics agent a question.

    Args:
        question: Natural language question about store inventory or performance

    Returns:
        dict with:
          - answer       : str   — the agent's final answer
          - iterations   : int   — how many LLM calls were made
          - tool_calls   : list  — which tools were called, in order
          - latency_ms   : int   — total wall-clock time
    """
    start_time = time.time()

    initial_state: AgentState = {
        "messages"  : [HumanMessage(content=question)],
        "iteration" : 0,
    }

    logger.info(f"Agent invoked | question={question[:80]}")
    final_state = gstore_agent.invoke(initial_state)

    # ── Extract answer ──────────────────────────────────────────────────────
    final_message = final_state["messages"][-1]
    answer        = final_message.content if hasattr(final_message, "content") else str(final_message)

    # ── Extract tool calls made ─────────────────────────────────────────────
    # Walk the message history and collect all tool calls (for observability)
    tools_called = []
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tools_called.extend([tc["name"] for tc in msg.tool_calls])

    latency_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Agent done | "
        f"iterations={final_state['iteration']} | "
        f"tools={tools_called} | "
        f"latency_ms={latency_ms}"
    )

    return {
        "answer"    : answer,
        "iterations": final_state["iteration"],
        "tool_calls": tools_called,
        "latency_ms": latency_ms,
    }


# ── TESTS ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Test cases designed to exercise different agent paths:
    # 1. Single tool call path     (inventory query)
    # 2. Dual tool call path       (inventory + performance for same store)
    # 3. Multi-store path          (LLM may call tools multiple times)
    # 4. No-tool path              (question answerable from LLM knowledge)
    # 5. Missing store_id path     (agent should ask for clarification)

    test_questions = [
        # "What's the current inventory at store HM-LON-042?",
        # "Is HM-NYC-018 out of polo shirts? And how are their sales today?",
        "Which store has better conversion rate — London or Berlin?",
        # "What does RFID stand for and how does it work in retail?",
        # "What's the inventory at our New York store?",
    ]

    print("\n" + "=" * 72)
    print("LANGGRAPH 02 — gStore Analytics Agent (LLM + Tools + ReAct Loop)")
    print("=" * 72)

    for question in test_questions:
        print(f"\nQ: {question}")
        result = ask_agent(question)
        print(f"A: {result['answer']}")
        print(
            f"   [iterations={result['iterations']} | "
            f"tools={result['tool_calls']} | "
            f"latency={result['latency_ms']}ms]"
        )
        print("-" * 72)
