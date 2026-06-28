"""
langgraph_05_hitl.py

PURPOSE  : Human-in-the-Loop (HITL) — pause agent mid-execution for human review.

USE CASE : gStore Alert Review System
           High/critical store alerts require ops manager approval before escalation.
           Agent analyzes → proposes action → PAUSES → human decides → agent executes.

NEW PRIMITIVES:
  interrupt(value)          — pauses execution inside a node; value is shown to human
  Command(resume=value)     — resumes interrupted graph carrying human's input back
  graph.get_state(config)   — inspect paused state and pending interrupt data
  graph.update_state(...)   — modify state BEFORE resuming (human edits agent's proposal)

EXECUTION MODEL (key conceptual shift from all previous sessions):

  Without HITL:
    graph.invoke(state, config) ──────────────────────────→ final state

  With HITL:
    graph.invoke(state, config) ──→ PAUSES at interrupt()
                                         ↓
                                   [human reviews]
                                         ↓
    graph.invoke(Command(resume=...), config) ──→ resumes ──→ final state

  The paused state lives in the MemorySaver checkpointer.
  The process can restart — the graph picks up exactly where it stopped.

FOUR DEMO SCENARIOS (covers every HITL pattern in production):
  1. Low severity   → auto-handled, no interrupt ever fires
  2. High severity  → interrupt fires, human approves
  3. High severity  → interrupt fires, human rejects with reason
  4. Critical       → interrupt fires, human EDITS proposed action, then approves

RUN      : python langgraph_05_hitl.py
REQUIRES : pip install langgraph langchain-openai langchain-google-genai pydantic
           OPENAI_API_KEY and GEMINI_API_KEY in .env
"""

import logging
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel

load_dotenv()

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── LLM SETUP ─────────────────────────────────────────────────────────────────
_openai = ChatOpenAI(
    model   = "gpt-5.4-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
)
_gemini = ChatGoogleGenerativeAI(
    model          = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
)
llm = _openai.with_fallbacks([_gemini])


# ── STRUCTURED OUTPUT: Alert Analysis ─────────────────────────────────────────
# DOMAIN KNOWLEDGE: Structured output forces the LLM to return a typed Pydantic
# object. Critical for routing decisions — you cannot afford string parsing errors
# when the output determines whether a human gets paged at 3am.

class AlertAnalysis(BaseModel):
    severity        : Literal["low", "high", "critical"]
    summary         : str   # 1–2 sentences for the ops manager to read at a glance
    proposed_action : str   # specific action the agent recommends
    reasoning       : str   # why this severity was assigned

analysis_llm = _openai.with_structured_output(AlertAnalysis)


# ── STATE ─────────────────────────────────────────────────────────────────────
class AlertState(TypedDict):
    # Input fields (set at start, immutable)
    alert_text      : str
    store_id        : str

    # Filled by analyze_alert node
    severity        : str
    alert_summary   : str
    proposed_action : str

    # Filled by request_approval node (after interrupt resumes)
    approved        : bool
    human_decision  : dict   # full response from human: approved, reviewer, notes, modified_action

    # Filled by terminal nodes
    execution_result: str

    # Conversation history — not used for routing, but useful for audit trails
    messages        : Annotated[list, add_messages]


# ── NODE: ANALYZE ALERT ───────────────────────────────────────────────────────
def analyze_alert(state: AlertState) -> dict:
    """
    NODE: analyze_alert
    LLM analyzes the raw alert and produces a structured assessment.
    Sets severity, summary, and proposed_action — all three used downstream.
    """
    logger.info(f"Node: analyze_alert | store={state['store_id']}")

    messages = [
        SystemMessage(content="""You are a gStore ops analyst for GreyOrange's retail platform.
Analyze store alerts and determine:
- severity: "low" (informational, no immediate action), "high" (needs action within hours),
  "critical" (needs immediate action, potential revenue impact)
- summary: 1-2 sentences a manager can read in 5 seconds
- proposed_action: specific, actionable recommendation
- reasoning: brief justification for your severity classification"""),
        HumanMessage(content=f"Store: {state['store_id']}\nAlert: {state['alert_text']}"),
    ]

    analysis: AlertAnalysis = analysis_llm.invoke(messages)

    logger.info(
        f"Analysis complete | severity={analysis.severity} | "
        f"action='{analysis.proposed_action[:60]}'"
    )

    return {
        "severity"       : analysis.severity,
        "alert_summary"  : analysis.summary,
        "proposed_action": analysis.proposed_action,
    }


# ── NODE: AUTO LOG ────────────────────────────────────────────────────────────
def auto_log(state: AlertState) -> dict:
    """
    NODE: auto_log
    Handles low-severity alerts automatically — no human involvement.
    This path never triggers an interrupt.
    """
    logger.info(f"Node: auto_log | store={state['store_id']} | severity=low")

    result = (
        f"[AUTO-HANDLED] {state['store_id']} | LOW | "
        f"{state['alert_summary']} | "
        f"Action: logged to dashboard, no escalation required."
    )

    return {"execution_result": result}


# ── NODE: REQUEST APPROVAL ────────────────────────────────────────────────────
def request_approval(state: AlertState) -> dict:
    """
    NODE: request_approval
    The HITL node — pauses execution and waits for human decision.

    HOW interrupt() WORKS:
      1. interrupt(value) is called — execution PAUSES here
      2. The value dict is what the human sees (surfaced via get_state())
      3. The human calls graph.invoke(Command(resume=their_response), config)
      4. Execution RESUMES from this exact line — interrupt() RETURNS their_response
      5. The rest of the node runs with their_response in hand

    WHAT THE HUMAN SENDS BACK (expected schema):
      {
        "approved"       : bool,              required
        "reviewer"       : str,               required — who made the decision
        "notes"          : str,               optional — free text reasoning
        "modified_action": str | None,        optional — human-edited version of proposed_action
      }

    If the human called graph.update_state() to change proposed_action before resuming,
    that change is already in state["proposed_action"] when this node runs.
    """
    logger.info(f"Node: request_approval | store={state['store_id']} | severity={state['severity']}")

    # Package everything the ops manager needs to make a decision
    review_request = {
        "store_id"        : state["store_id"],
        "severity"        : state["severity"].upper(),
        "alert_text"      : state["alert_text"],
        "agent_summary"   : state["alert_summary"],
        "proposed_action" : state["proposed_action"],
        "instructions"    : (
            "Respond with: "
            "{approved: bool, reviewer: str, notes: str (optional), "
            "modified_action: str | None (optional — override agent's proposed action)}"
        ),
    }

    # ── EXECUTION PAUSES HERE ─────────────────────────────────────────────────
    # Everything above this line ran before the pause.
    # Everything below runs AFTER the human sends Command(resume=...).
    # The 'human_response' variable contains exactly what they passed to resume.
    human_response: dict = interrupt(review_request)
    # ── EXECUTION RESUMES HERE ────────────────────────────────────────────────

    logger.info(
        f"Approval received | approved={human_response.get('approved')} | "
        f"reviewer={human_response.get('reviewer', 'unknown')}"
    )

    # If the human provided a modified action, update proposed_action
    # (They could also have done this via graph.update_state() before resuming —
    # both patterns are valid; this handles the in-band modification case)
    final_action = human_response.get("modified_action") or state["proposed_action"]

    return {
        "approved"       : human_response.get("approved", False),
        "human_decision" : human_response,
        "proposed_action": final_action,   # may be human-modified
    }


# ── NODE: EXECUTE APPROVED ────────────────────────────────────────────────────
def execute_approved(state: AlertState) -> dict:
    """
    NODE: execute_approved
    Runs after human approval. Executes the (potentially modified) proposed_action.
    In production: sends PagerDuty alert, creates Linear ticket, pages store team, etc.
    """
    logger.info(f"Node: execute_approved | store={state['store_id']}")

    reviewer = state["human_decision"].get("reviewer", "unknown")
    notes    = state["human_decision"].get("notes", "")

    result = (
        f"[ESCALATED] {state['store_id']} | {state['severity'].upper()} | "
        f"Action executed: {state['proposed_action']} | "
        f"Approved by: {reviewer}"
        + (f" | Notes: {notes}" if notes else "")
    )

    return {"execution_result": result}


# ── NODE: LOG REJECTION ───────────────────────────────────────────────────────
def log_rejection(state: AlertState) -> dict:
    """
    NODE: log_rejection
    Runs when human rejects the proposed action.
    Logs the rejection with reason for audit trail — important for compliance.
    """
    logger.info(f"Node: log_rejection | store={state['store_id']}")

    reviewer = state["human_decision"].get("reviewer", "unknown")
    notes    = state["human_decision"].get("notes", "No reason given")

    result = (
        f"[REJECTED] {state['store_id']} | {state['severity'].upper()} | "
        f"Proposed action rejected by: {reviewer} | "
        f"Reason: {notes} | "
        f"Alert logged for record — no escalation sent."
    )

    return {"execution_result": result}


# ── ROUTING FUNCTIONS ─────────────────────────────────────────────────────────
def route_by_severity(state: AlertState) -> str:
    """Routes after analyze_alert: low → auto_log, everything else → request_approval."""
    if state["severity"] == "low":
        logger.info("Routing: severity=low → auto_log (no human required)")
        return "auto_log"
    logger.info(f"Routing: severity={state['severity']} → request_approval (human required)")
    return "request_approval"


def route_after_approval(state: AlertState) -> str:
    """Routes after request_approval: approved → execute, rejected → log."""
    return "execute_approved" if state["approved"] else "log_rejection"


# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────
def build_alert_review_agent() -> StateGraph:
    """
    Builds the alert review graph.

    Topology:
        START
          ↓
        analyze_alert
          ↓ (route_by_severity)
          ├── "low"           → auto_log ─────────────────────┐
          └── "high/critical" → request_approval              │
                                     ↓                        │
                              [ interrupt() ]                 │
                                     ↓ (human resumes)        │
                              (route_after_approval)          │
                              ↓               ↓               │
                      execute_approved   log_rejection        │
                              ↓               ↓               │
                              └───────────────┘               │
                                      ↓       ←───────────────┘
                                     END

    CRITICAL: MemorySaver is REQUIRED for HITL.
    interrupt() saves the paused state to the checkpointer.
    Without a checkpointer, interrupt() raises an error — there's nowhere to save.
    """
    builder = StateGraph(AlertState)

    builder.add_node("analyze_alert",   analyze_alert)
    builder.add_node("auto_log",        auto_log)
    builder.add_node("request_approval",request_approval)
    builder.add_node("execute_approved",execute_approved)
    builder.add_node("log_rejection",   log_rejection)

    builder.set_entry_point("analyze_alert")

    builder.add_conditional_edges("analyze_alert",    route_by_severity,    {"auto_log": "auto_log", "request_approval": "request_approval"})
    builder.add_conditional_edges("request_approval", route_after_approval, {"execute_approved": "execute_approved", "log_rejection": "log_rejection"})

    builder.add_edge("auto_log",         END)
    builder.add_edge("execute_approved", END)
    builder.add_edge("log_rejection",    END)

    # MemorySaver required — interrupt() stores paused state here between invoke() calls
    return builder.compile(checkpointer=MemorySaver())


# ── MODULE-LEVEL AGENT ─────────────────────────────────────────────────────────
alert_agent = build_alert_review_agent()


# ── HELPER: Inspect and display paused state ──────────────────────────────────
def inspect_interrupt(config: dict) -> dict | None:
    """
    After an invoke() that triggered interrupt(), call this to see:
    - Which node is pending
    - What data the human needs to review
    Returns the interrupt value dict, or None if graph is not paused.
    """
    snapshot = alert_agent.get_state(config)

    if not snapshot.next:
        return None   # graph is not paused

    interrupt_values = []
    for task in snapshot.tasks:
        for intr in task.interrupts:
            interrupt_values.append(intr.value)

    return {
        "paused_at"      : snapshot.next,          # tuple of pending node names
        "interrupt_data" : interrupt_values[0] if interrupt_values else {},
    }


# ── PUBLIC INTERFACE ──────────────────────────────────────────────────────────
def submit_alert(alert_text: str, store_id: str, thread_id: str) -> dict:
    """
    Step 1: Submit a new alert. May return paused (awaiting approval) or completed.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AlertState = {
        "alert_text"      : alert_text,
        "store_id"        : store_id,
        "severity"        : "",
        "alert_summary"   : "",
        "proposed_action" : "",
        "approved"        : False,
        "human_decision"  : {},
        "execution_result": "",
        "messages"        : [HumanMessage(content=alert_text)],
    }

    alert_agent.invoke(initial_state, config)

    pending = inspect_interrupt(config)
    if pending:
        return {"status": "PAUSED", "thread_id": thread_id, **pending}

    final = alert_agent.get_state(config)
    return {"status": "COMPLETED", "thread_id": thread_id, "result": final.values["execution_result"]}


def resume_with_decision(thread_id: str, decision: dict) -> dict:
    """
    Step 2: Resume a paused alert with the human's decision.

    decision dict:
      approved        : bool                — required
      reviewer        : str                 — required
      notes           : str                 — optional
      modified_action : str | None          — optional; overrides agent's proposed action

    To edit proposed_action via update_state instead of passing modified_action here:
      alert_agent.update_state(config, {"proposed_action": "new text"})
      then call resume_with_decision(thread_id, {"approved": True, "reviewer": "..."})
    """
    config = {"configurable": {"thread_id": thread_id}}
    alert_agent.invoke(Command(resume=decision), config)

    final = alert_agent.get_state(config)
    return {"status": "COMPLETED", "thread_id": thread_id, "result": final.values["execution_result"]}


# ── TESTS ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    SEP = "─" * 72

    print("\n" + "=" * 72)
    print("LANGGRAPH 05 — Human-in-the-Loop Alert Review")
    print("=" * 72)

    # ── SCENARIO 1: Low severity — no interrupt fires ─────────────────────────
    print(f"\n{'━'*72}")
    print("SCENARIO 1: Low severity → auto-handled (no human needed)")
    print("━" * 72)

    result = submit_alert(
        alert_text = "Routine daily stock count report ready for review",
        store_id   = "HM-LON-042",
        thread_id  = "alert_low_001",
    )
    print(f"Status : {result['status']}")
    print(f"Result : {result.get('result', 'N/A')}")

    # ── SCENARIO 2: High severity → interrupt → human approves ───────────────
    print(f"\n{'━'*72}")
    print("SCENARIO 2: High severity → PAUSE → human approves")
    print("━" * 72)

    result = submit_alert(
        alert_text = "RFID scanner offline — zero inventory reads for 2 hours. Revenue impact estimated at £8,000.",
        store_id   = "HM-LON-042",
        thread_id  = "alert_high_001",
    )
    print(f"Status     : {result['status']}")
    print(f"Paused at  : {result.get('paused_at')}")

    if result["status"] == "PAUSED":
        review = result["interrupt_data"]
        print(f"\n── Ops Manager sees: ──────────────────────────────────────────────")
        for k, v in review.items():
            if k != "instructions":
                print(f"  {k:<20}: {v}")

        print(f"\n── Ops Manager approves ───────────────────────────────────────────")
        final = resume_with_decision(
            thread_id = "alert_high_001",
            decision  = {
                "approved" : True,
                "reviewer" : "ops_manager_priya",
                "notes"    : "Confirmed — RFID team dispatched, critical revenue window.",
            }
        )
        print(f"Result : {final['result']}")

    # ── SCENARIO 3: High severity → interrupt → human rejects ────────────────
    print(f"\n{'━'*72}")
    print("SCENARIO 3: High severity → PAUSE → human rejects")
    print("━" * 72)

    result = submit_alert(
        alert_text = "Sales conversion rate dropped 15% vs yesterday between 2pm–4pm.",
        store_id   = "HM-BER-007",
        thread_id  = "alert_high_002",
    )
    print(f"Status     : {result['status']}")

    if result["status"] == "PAUSED":
        print(f"\n── Ops Manager sees proposed action: ──────────────────────────────")
        print(f"  {result['interrupt_data'].get('proposed_action')}")

        print(f"\n── Ops Manager rejects — normal fluctuation ───────────────────────")
        final = resume_with_decision(
            thread_id = "alert_high_002",
            decision  = {
                "approved" : False,
                "reviewer" : "ops_manager_priya",
                "notes"    : "Checked manually — afternoon dip is within normal variance for Tuesday.",
            }
        )
        print(f"Result : {final['result']}")

    # ── SCENARIO 4: Critical → interrupt → human EDITS proposed action ───────
    print(f"\n{'━'*72}")
    print("SCENARIO 4: Critical → PAUSE → human edits proposed action → approves")
    print("━" * 72)

    THREAD_4 = "alert_critical_001"
    result = submit_alert(
        alert_text = "CRITICAL: Zero inventory detected across ALL SKUs — H&M Berlin. Dashboard shows no data for 3 hours. Possible pipeline failure.",
        store_id   = "HM-BER-007",
        thread_id  = THREAD_4,
    )
    print(f"Status     : {result['status']}")

    if result["status"] == "PAUSED":
        config = {"configurable": {"thread_id": THREAD_4}}

        print(f"\n── Agent proposed: ────────────────────────────────────────────────")
        original_action = result["interrupt_data"].get("proposed_action", "")
        print(f"  {original_action}")

        # Human disagrees with urgency level — edits via update_state BEFORE resuming
        # This is the update_state pattern: modify state while graph is paused,
        # then resume. The node will see the updated proposed_action when it finishes.
        modified_action = "Escalate to data engineering on-call (not store team) — likely BigQuery pipeline issue, not physical inventory. Check Cloud Run job status first."
        alert_agent.update_state(config, {"proposed_action": modified_action})

        print(f"\n── Ops Manager edited action to: ─────────────────────────────────")
        print(f"  {modified_action}")

        print(f"\n── Ops Manager approves with edit ─────────────────────────────────")
        final = resume_with_decision(
            thread_id = THREAD_4,
            decision  = {
                "approved" : True,
                "reviewer" : "ops_manager_priya",
                "notes"    : "Redirected to data eng — pipeline issue not store hardware.",
            }
        )
        print(f"Result : {final['result']}")

    print(f"\n{'='*72}\n")
