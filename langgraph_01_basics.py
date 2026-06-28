"""
langgraph_01_basics.py

PURPOSE  : Understand LangGraph StateGraph mechanics — no LLM involved.
           Master State, Nodes, Edges, Conditional Routing, and Reducers
           before adding LLM complexity.

USE CASE : gStore alert routing pipeline (GreyOrange domain)
           Alerts arrive from stores → classify → route to correct team → format → done.

RUN      : python langgraph_01_basics.py
REQUIRES : pip install langgraph
"""

import logging
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, StateGraph

# ── LOGGING SETUP ───────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── STATE DEFINITION ────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: State is the "whiteboard" shared across all nodes.
# Every node reads from it and returns ONLY the fields it changed.
# The LangGraph runtime merges those changes back into the full state.
#
# TypedDict enforces the shape — every key is a field nodes can read/update.

# PYTHON CONCEPT: Annotated[type, reducer_fn] tells LangGraph how to merge
# updates when multiple nodes write to the same key.
#
# Without reducer  → last write wins   (normal for single-writer fields)
# With reducer     → function decides  (needed for fields multiple nodes write to)
#
# handler_log accumulates entries from multiple nodes, so it needs a reducer.

def append_to_list(existing: list[str], new: list[str]) -> list[str]:
    """
    Custom reducer for handler_log.
    LangGraph calls this with (existing_value, new_value) whenever
    a node returns an update to a field with this reducer attached.
    """
    # PYTHON CONCEPT: The + operator on lists produces a NEW list (immutable merge).
    # Never mutate 'existing' in-place — reducers must be pure functions.
    return existing + new


class AlertState(TypedDict):
    alert_text  : str                              # raw alert from gStore (immutable — set once)
    store_id    : str                              # store that triggered the alert
    alert_type  : str                              # "inventory" | "performance" | "system" | "unknown"
    severity    : str                              # "critical" | "high" | "medium" | "low"
    handler_log : Annotated[list[str], append_to_list]  # accumulates log entries (reducer applied)
    response    : str                              # final formatted response (last write wins)


# ── NODE FUNCTIONS ──────────────────────────────────────────────────────────
# CRITICAL PATTERN:
#   Input  → full AlertState
#   Output → dict of ONLY the fields this node changed
#
# Do NOT return the full state. LangGraph merges your partial return
# into the existing state. Returning unchanged fields is wasted bandwidth
# and can cause subtle bugs with reducers.

def classify_alert(state: AlertState) -> dict:
    """
    NODE: classify_alert
    Reads  : alert_text
    Writes : alert_type, severity, handler_log

    DOMAIN KNOWLEDGE: gStore alerts fall into three categories:
    - Inventory  → RFID scanner issues, stock count anomalies (critical if data gap)
    - Performance → sales drops, conversion declines (impacts H&M revenue reporting)
    - System      → pipeline failures, BigQuery timeouts, dashboard staleness
    """
    logger.info(f"Node: classify_alert | store={state['store_id']}")

    text = state["alert_text"].lower()

    # ── CLASSIFY ────────────────────────────────────────────────────────────
    if any(kw in text for kw in ["rfid", "scan", "stock", "inventory", "item count", "sku"]):
        alert_type = "inventory"
        severity   = "critical" if any(x in text for x in ["zero", "missing", "none"]) else "high"

    elif any(kw in text for kw in ["sales", "revenue", "conversion", "drop", "decline", "basket"]):
        alert_type = "performance"
        severity   = "critical" if any(x in text for x in ["40%", "50%", "zero", "drop"]) else "medium"

    elif any(kw in text for kw in ["pipeline", "bigquery", "dashboard", "timeout", "stale", "error"]):
        alert_type = "system"
        severity   = "critical" if any(x in text for x in ["down", "timeout", "failed"]) else "high"

    else:
        alert_type = "unknown"
        severity   = "medium"

    # PATTERN: Return ONLY the fields this node changes.
    # handler_log uses a reducer — returning a list means "append these entries".
    return {
        "alert_type"  : alert_type,
        "severity"    : severity,
        "handler_log" : [f"classify_alert: tagged as [{alert_type} / {severity}]"],
    }


def handle_inventory_alert(state: AlertState) -> dict:
    """
    NODE: handle_inventory_alert
    Handles RFID scanner issues and stock count anomalies.
    Reads  : store_id, severity, alert_text
    Writes : response, handler_log
    """
    logger.info(f"Node: handle_inventory_alert | store={state['store_id']}")

    response = (
        f"[INVENTORY TEAM] {state['store_id']} | {state['severity'].upper()} | "
        f"Dispatching RFID diagnostic protocol. "
        f"Alert: {state['alert_text'][:80]}"
    )

    return {
        "response"    : response,
        "handler_log" : ["handle_inventory: RFID diagnostic queued"],
    }


def handle_performance_alert(state: AlertState) -> dict:
    """
    NODE: handle_performance_alert
    Handles sales and revenue metric anomalies.
    """
    logger.info(f"Node: handle_performance_alert | store={state['store_id']}")

    response = (
        f"[PERFORMANCE TEAM] {state['store_id']} | {state['severity'].upper()} | "
        f"Flagging for growth team review. "
        f"Alert: {state['alert_text'][:80]}"
    )

    return {
        "response"    : response,
        "handler_log" : ["handle_performance: escalated to growth team"],
    }


def handle_system_alert(state: AlertState) -> dict:
    """
    NODE: handle_system_alert
    Handles pipeline and infrastructure issues.
    """
    logger.info(f"Node: handle_system_alert | store={state['store_id']}")

    response = (
        f"[DATA ENGINEERING] {state['store_id']} | {state['severity'].upper()} | "
        f"Pipeline incident raised. BigQuery/Kafka diagnostic initiated. "
        f"Alert: {state['alert_text'][:80]}"
    )

    return {
        "response"    : response,
        "handler_log" : ["handle_system: data engineering paged"],
    }


def handle_unknown_alert(state: AlertState) -> dict:
    """NODE: handle_unknown_alert — fallback for unclassified alerts."""
    logger.info(f"Node: handle_unknown_alert | store={state['store_id']}")

    return {
        "response"    : (
            f"[TRIAGE] {state['store_id']} | {state['severity'].upper()} | "
            f"Unclassified alert. Manual review required. "
            f"Alert: {state['alert_text'][:80]}"
        ),
        "handler_log" : ["handle_unknown: sent to manual triage"],
    }


def format_final_response(state: AlertState) -> dict:
    """
    NODE: format_final_response
    Shared terminal node — ALL alert types pass through here.
    This is the "fan-in" point of the graph.

    Demonstrates: multiple upstream nodes can converge on a single downstream node.
    The response field here is a last-write-wins field — format just adds metadata.
    """
    logger.info(f"Node: format_final_response | store={state['store_id']}")

    # handler_log is now fully accumulated (all previous nodes have appended to it)
    audit_trail = " → ".join(state["handler_log"])

    formatted = (
        f"{state['response']}\n"
        f"  Audit: {audit_trail}"
    )

    return {
        "response"    : formatted,
        "handler_log" : ["format_final_response: response finalized"],
    }


# ── ROUTING FUNCTION ────────────────────────────────────────────────────────
# CRITICAL DISTINCTION: This is NOT a node. It is a ROUTER.
# Routers are plain functions that read state and return the NAME of the next node.
# They have NO side effects and do NOT write to state.
#
# Return type annotation must match the keys in add_conditional_edges mapping.

def route_by_alert_type(
    state: AlertState
) -> Literal["inventory", "performance", "system", "unknown"]:
    """
    Routes execution to the correct handler based on the alert_type set by classify_alert.
    Called after classify_alert. Returns a node name — not a state update.
    """
    return state["alert_type"]   # the graph uses this string to pick the next node


# ── GRAPH CONSTRUCTION ──────────────────────────────────────────────────────
def build_alert_router() -> StateGraph:
    """
    Builds and compiles the gStore alert routing graph.

    Graph topology (fan-out → fan-in):

        START
          ↓
        classify_alert
          ↓ (conditional based on alert_type)
          ├── "inventory"   ──→  handle_inventory_alert   ──┐
          ├── "performance" ──→  handle_performance_alert ──┤
          ├── "system"      ──→  handle_system_alert      ──┤
          └── "unknown"     ──→  handle_unknown_alert     ──┘
                                                            ↓
                                              format_final_response
                                                            ↓
                                                          END
    """
    builder = StateGraph(AlertState)

    # ── ADD NODES ────────────────────────────────────────────────────────────
    # First argument = name (string used in edges). Second = the function.
    # Node name and function name don't have to match — name is just a label.
    builder.add_node("classify",    classify_alert)
    builder.add_node("inventory",   handle_inventory_alert)
    builder.add_node("performance", handle_performance_alert)
    builder.add_node("system",      handle_system_alert)
    builder.add_node("unknown",     handle_unknown_alert)
    builder.add_node("format",      format_final_response)

    # ── ADD EDGES ────────────────────────────────────────────────────────────
    builder.set_entry_point("classify")   # execution starts here

    # Conditional edge: classify → one of [inventory, performance, system, unknown]
    # add_conditional_edges(source_node, routing_function, mapping_dict)
    # mapping_dict: {routing_fn_return_value → destination_node_name}
    builder.add_conditional_edges(
        "classify",
        route_by_alert_type,
        {
            "inventory"  : "inventory",
            "performance": "performance",
            "system"     : "system",
            "unknown"    : "unknown",
        }
    )

    # Fan-in: all four handlers converge on the shared format node
    builder.add_edge("inventory",   "format")
    builder.add_edge("performance", "format")
    builder.add_edge("system",      "format")
    builder.add_edge("unknown",     "format")

    # Terminal edge to END (imported from langgraph.graph)
    builder.add_edge("format", END)

    # compile() validates the graph (catches missing nodes, disconnected subgraphs)
    # and produces a runnable CompiledGraph object.
    # PRODUCTION NOTE: Call compile() ONCE at module load, not per request.
    return builder.compile()


# ── MODULE-LEVEL GRAPH INSTANCE ─────────────────────────────────────────────
# PYTHON CONCEPT: Create the compiled graph once. Compilation validates the graph
# structure and is not cheap — instantiating per-request is wasteful.

alert_router = build_alert_router()


# ── PUBLIC INTERFACE ─────────────────────────────────────────────────────────
def process_alert(alert_text: str, store_id: str) -> AlertState:
    """
    Process a gStore alert through the routing graph.

    Args:
        alert_text : raw alert message from the store
        store_id   : store identifier (e.g. "HM-LON-042")

    Returns:
        Final AlertState after all graph nodes have executed
    """

    # The initial state — must include ALL required TypedDict keys.
    # Fields that will be filled by nodes can start as empty strings / lists.
    initial_state: AlertState = {
        "alert_text"  : alert_text,
        "store_id"    : store_id,
        "alert_type"  : "",    # will be set by classify_alert
        "severity"    : "",    # will be set by classify_alert
        "handler_log" : [],    # empty — reducer will accumulate entries across nodes
        "response"    : "",    # will be set by handler node
    }

    logger.info(f"Processing alert | store={store_id} | text={alert_text[:60]}")
    final_state = alert_router.invoke(initial_state)
    logger.info(
        f"Alert done | type={final_state['alert_type']} | severity={final_state['severity']}"
    )

    return final_state


# ── TESTS ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_cases = [
        # (alert_text, store_id, expected_type)
        ("RFID scanner showing zero item count for 3 SKUs in aisle 4", "HM-LON-042", "inventory"),
        ("Revenue drop of 40% in last 2 hours vs yesterday baseline",  "HM-NYC-018", "performance"),
        ("BigQuery pipeline timeout — dashboard showing stale data",   "HM-BER-007", "system"),
        ("Customer reported something seems off with the app",         "HM-PAR-033", "unknown"),
    ]

    print("\n" + "=" * 72)
    print("LANGGRAPH 01 — gStore Alert Router (No LLM)")
    print("=" * 72)

    for alert_text, store_id, expected_type in test_cases:
        print(f"\n[INPUT]")
        print(f"  store_id   : {store_id}")
        print(f"  alert_text : {alert_text}")
        print(f"  expected   : {expected_type}")

        result = process_alert(alert_text, store_id)

        print(f"[OUTPUT]")
        print(f"  alert_type : {result['alert_type']}")
        print(f"  severity   : {result['severity']}")
        print(f"  response   :")
        print(f"    {result['response']}")
        print("-" * 72)
