"""
langsmith_06_observability.py
═══════════════════════════════════════════════════════════════════════════════

PURPOSE  : AI Observability — trace every LLM call, node, and graph invocation
           so you can debug failures, measure latency, and audit costs in
           production multi-agent systems.

USE CASE : gStore Replenishment Alert Classifier
           Every alert analysis is traced end-to-end in LangSmith.
           When the classifier routes wrong or costs spike, you can see
           the exact prompt and LLM response that caused it — in the UI.

WHAT'S NEW vs Session 5A:
  1. @traceable decorator  — wraps plain Python functions into the trace tree
  2. Enriched config={}    — run_name, metadata, tags make traces searchable
  3. RunCollectorCallbackHandler — captures run IDs for programmatic feedback
  4. Feedback API          — log whether a run was correct (foundation for eval)

SETUP (once):
  Add to .env:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=gstore-ai-dev
  Then just run — traces appear automatically at smith.langchain.com

RUN      : python langsmith_06_observability.py
REQUIRES : pip install langsmith langchain-openai langgraph
           OPENAI_API_KEY, LANGCHAIN_API_KEY in .env
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────
import json
import logging
import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import Client as LangSmithClient
from langsmith import traceable
from typing_extensions import TypedDict

# ── ENVIRONMENT + LOGGING ──────────────────────────────────────────────────────
# IMPORTANT: load_dotenv() must run before anything reads os.getenv().
# LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT are read automatically
# by LangChain at startup — you don't pass them anywhere in code.
# Setting LANGCHAIN_TRACING_V2=true in .env is all you need to enable tracing.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── CONSTANTS ──────────────────────────────────────────────────────────────────
MODEL_NAME       = "gpt-5.4-mini"
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "gstore-ai-dev")

# ── LLM CLIENT ────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Why ChatOpenAI (not the direct openai SDK)?
# LangSmith auto-traces LangChain wrappers (ChatOpenAI, ChatAnthropic, etc.)
# because LangChain injects a callback system into every call it makes.
# The raw openai.chat.completions.create() bypasses that system entirely —
# LangSmith never sees those calls.
# Rule: if you want it traced without extra work, use the LangChain wrapper.
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ── STATE SCHEMA ───────────────────────────────────────────────────────────────
# PYTHON CONCEPT: TypedDict
# A TypedDict declares the shape of a dict — what keys it can hold and what
# type each value is. It's a documentation and IDE hint, not a runtime check.
# LangGraph reads this to know what your state looks like.
# Nodes return partial dicts (only the keys they update), and LangGraph
# merges them into the existing state. So a node can return just
# {"classification": "data_quality"} and only that key changes.
class AlertState(TypedDict):
    # ── INPUTS (provided by caller) ────────────────────────────
    alert_text: str   # raw alert message from the replenishment system
    store_id:   str   # e.g. "PET-NYC-001"
    client:     str   # e.g. "PetSmart"
    # ── SET BY classify NODE ────────────────────────────────────
    parsed_fields:  dict  # structured fields extracted before the LLM runs
    classification: str   # data_quality | ops_failure | config_error | false_positive
    severity:       str   # low | medium | high | critical
    # ── SET BY recommend NODE ───────────────────────────────────
    recommendation: str   # concrete action for the on-call team


# ── @traceable: CUSTOM PYTHON FUNCTION IN THE TRACE TREE ──────────────────────
# DOMAIN KNOWLEDGE: Why trace a pure Python function?
# parse_alert_fields is deterministic Python — no LLM involved.
# Without @traceable, it is invisible in LangSmith: you see the classify node
# start, then an LLM call, but you never see what structured data the LLM
# actually received. If the parsed store_id is wrong, you have no way to know
# that the problem happened BEFORE the LLM, not inside it.
# With @traceable, it appears as a child run inside the classify node's run.
# You can click it in the UI and see exactly what it returned.
#
# PYTHON CONCEPT: The @traceable decorator and context awareness
# A decorator is a function that wraps another function to add behaviour.
# @traceable adds LangSmith recording to any function it wraps.
# Context-awareness: LangSmith uses Python's `contextvars` module to track
# whether there is an active parent trace. When called from inside a LangGraph
# node, @traceable automatically nests this run under that node's run — no
# extra configuration needed. Call it from outside a graph and it creates its
# own root trace. The nesting is automatic and context-driven.
@traceable(name="parse_alert_fields")
def parse_alert_fields(alert_text: str) -> dict:
    """
    Extract structured fields from raw alert text using regex matching.

    Runs BEFORE the LLM and makes extracted data visible in LangSmith.
    This is the "thick tools" principle: encode deterministic domain
    knowledge in a testable Python function, not in the LLM prompt.
    The LLM receives clean structured inputs, not raw text to interpret.
    """
    fields: dict = {
        "store_id_parsed":   "unknown",
        "sku_count_actual":  None,
        "sku_count_expected": None,
        "raw_text_length":   len(alert_text),
    }

    # Match patterns like "Store: PET-NYC-001" or "store_id=HM-LON-042"
    store_match = re.search(
        r"(?:store[_\s]*id[:\s=]+|store[:\s]+)([A-Z]{2,5}-[A-Z]{2,5}-\d{2,4})",
        alert_text,
        re.IGNORECASE,
    )
    if store_match:
        fields["store_id_parsed"] = store_match.group(1).upper()

    # Match "actual: 1,234" or "1234 actual SKUs"
    actual_match = re.search(r"actual[:\s]*([\d,]+)", alert_text, re.IGNORECASE)
    if actual_match:
        fields["sku_count_actual"] = int(actual_match.group(1).replace(",", ""))

    # Match "expected: 1,450"
    expected_match = re.search(r"expected[:\s]*([\d,]+)", alert_text, re.IGNORECASE)
    if expected_match:
        fields["sku_count_expected"] = int(expected_match.group(1).replace(",", ""))

    return fields


# ── NODE 1: CLASSIFY ───────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Classify before you act.
# Every alert goes through classification first, for two reasons:
#   1. It prevents sending an ops_failure alert to the data engineering team
#      and a config_error to the ops team — routing requires classification first.
#   2. It makes the second node's prompt shorter and more focused — the LLM
#      generating a recommendation already knows the alert type and severity,
#      so it doesn't have to figure that out from raw text.
# In LangSmith, this separation means you can evaluate classification quality
# independently from recommendation quality — two different failure modes,
# two different feedback loops.
def classify_alert(state: AlertState) -> dict:
    """
    Classify the alert type and severity.
    Calls parse_alert_fields (@traceable) first — visible as a child run.
    """
    # ── STEP 1: EXTRACT STRUCTURED FIELDS ─────────────────────────────────────
    # parse_alert_fields appears in LangSmith as a child run inside this node.
    # Its output is visible: you can see what store_id was parsed, what
    # sku counts were extracted, BEFORE the LLM ever sees the data.
    parsed = parse_alert_fields(state["alert_text"])
    logger.info(f"classify | parsed_fields={parsed}")

    # ── STEP 2: LLM CLASSIFICATION ─────────────────────────────────────────────
    # DOMAIN KNOWLEDGE: Why require JSON output at the prompt level?
    # The classify node feeds structured data into the recommend node via state.
    # If the LLM returns prose ("I think this is a data quality issue..."),
    # we'd need fragile string parsing that breaks on any wording change.
    # Requiring JSON and parsing it is the standard production pattern.
    # The LangSmith trace will show you both the prompt and the raw JSON string
    # the model returned — which is exactly what you need to debug parse failures.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a gStore replenishment QA engineer. "
                "Classify the incoming alert.\n"
                "Return ONLY valid JSON — no markdown, no explanation:\n"
                '{{"classification": "data_quality|ops_failure|config_error|false_positive", '
                '"severity": "low|medium|high|critical", '
                '"reason": "<one concise sentence>"}}'
            ),
        ),
        (
            "human",
            "Alert text: {alert_text}\nParsed metadata: {parsed_fields}",
        ),
    ])

    # PYTHON CONCEPT: LCEL pipe operator |
    # prompt | llm | StrOutputParser() chains three steps in sequence:
    # prompt formats the template → llm calls the model → StrOutputParser
    # extracts the text content from the response object.
    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "alert_text":    state["alert_text"],
            "parsed_fields": json.dumps(parsed),   # dict → JSON string for the template
        })
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        # DOMAIN KNOWLEDGE: Fail safe — an unknown classification is better
        # than a crash. In LangSmith, you'll see this logged as an error
        # and can click the LLM call run to see exactly what the model returned.
        logger.error(f"classify | LLM returned non-JSON: {raw[:200]}")
        result = {
            "classification": "data_quality",
            "severity":       "medium",
            "reason":         "Classification failed — defaulting to data_quality",
        }

    return {
        "parsed_fields":  parsed,
        "classification": result.get("classification", "data_quality"),
        "severity":       result.get("severity", "medium"),
    }


# ── NODE 2: RECOMMEND ──────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Why separate classification from recommendation?
# Two failure modes, two feedback loops.
# If the recommendation is wrong but the classification was right, the problem
# is in the recommendation prompt — not the classification prompt.
# Without the separation, you can't diagnose which node is responsible.
# In LangSmith, you'll see two distinct LLM call runs with their own
# prompts, latency, and token counts — independently auditable.
def generate_recommendation(state: AlertState) -> dict:
    """Generate a concrete action recommendation for the on-call team."""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a gStore operations lead writing runbooks for the on-call team. "
                "Given a classified replenishment alert, produce a concise action recommendation. "
                "2–3 sentences. Be specific: name the system to check, the team to contact, "
                "and the expected resolution time. Do not use bullet points."
            ),
        ),
        (
            "human",
            (
                "Store: {store_id} | Client: {client}\n"
                "Alert: {alert_text}\n"
                "Classification: {classification} | Severity: {severity}"
            ),
        ),
    ])

    chain = prompt | llm | StrOutputParser()

    recommendation = chain.invoke({
        "store_id":       state["store_id"],
        "client":         state["client"],
        "alert_text":     state["alert_text"],
        "classification": state["classification"],
        "severity":       state["severity"],
    })

    return {"recommendation": recommendation.strip()}


# ── GRAPH ASSEMBLY ─────────────────────────────────────────────────────────────
# Linear graph: classify → recommend → END
# Intentionally simple — the observability concepts are the focus, not
# agent routing. Once you understand what a trace looks like for a 2-node
# graph, reading Session 5A's 8-node subgraph trace is straightforward.
def build_graph() -> StateGraph:
    builder = StateGraph(AlertState)
    builder.add_node("classify", classify_alert)
    builder.add_node("recommend", generate_recommendation)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "recommend")
    builder.add_edge("recommend", END)
    return builder.compile()


graph = build_graph()


# ── RUN WITH ENRICHED TRACING ──────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: The three trace enrichment keys
#
# "run_name" : Human-readable label for the root trace in the UI.
#              Without it every trace is labelled "StateGraph" — useless at scale.
#              Best practice: include a searchable identifier (store_id, alert_id,
#              request_id). At 288,000 traces/day you need to be able to find one.
#
# "metadata" : A dict of key-value pairs attached to the trace and indexed.
#              Searchable and filterable in the UI.
#              Best practice: store_id, client, environment, anything you'd
#              filter on during an incident at 2am.
#
# "tags"     : A list of string labels — categorical filters in the UI.
#              Best practice: "env:dev" / "env:prod", feature name, system name.
#              Tags are how you answer "show me all dev traces from today".
#
# "callbacks": A list of callback handler objects.
#              We attach RunCollectorCallbackHandler to capture run IDs —
#              needed for programmatic feedback logging after invocation.
def run_alert(
    alert_text: str,
    store_id:   str,
    client:     str,
) -> tuple[AlertState, str]:
    """
    Invoke the alert classifier with fully enriched LangSmith tracing.

    Returns (final_state, root_run_id).
    root_run_id is used to log feedback on this specific trace.
    """
    initial_state: AlertState = {
        "alert_text":     alert_text,
        "store_id":       store_id,
        "client":         client,
        "parsed_fields":  {},
        "classification": "",
        "severity":       "",
        "recommendation": "",
    }

    # PYTHON CONCEPT: Instantiating a class
    # RunCollectorCallbackHandler() creates an instance — a Python object
    # with internal state. As the graph runs, LangGraph calls methods on
    # this object each time a run starts or ends. By the time graph.invoke()
    # returns, collector.traced_runs holds every run that was created.
    collector = RunCollectorCallbackHandler()

    result: AlertState = graph.invoke(
        initial_state,
        config={
            # ── TRACE IDENTITY ─────────────────────────────────────────────
            "run_name": f"alert-{store_id}-{client.lower().replace(' ', '_')}",
            # ── SEARCHABLE METADATA ────────────────────────────────────────
            "metadata": {
                "store_id":    store_id,
                "client":      client,
                "model":       MODEL_NAME,
                "environment": "dev",
            },
            # ── CATEGORICAL TAGS ───────────────────────────────────────────
            "tags": ["gstore", "replenishment", "alert-classifier", "dev"],
            # ── RUN COLLECTOR ──────────────────────────────────────────────
            # Attaches our collector so we retrieve the run_id after invoke.
            # Without this, we'd have to query the LangSmith API by timestamp —
            # fragile and slow. The collector is the clean way.
            "callbacks": [collector],
        },
    )

    # The root run (the graph invocation itself) is always added first —
    # before any of its child node runs. So traced_runs[0] is the root.
    root_run_id = (
        str(collector.traced_runs[0].id)
        if collector.traced_runs
        else "unknown"
    )

    logger.info(
        f"run_alert done | store={store_id} | "
        f"classification={result.get('classification')} | "
        f"severity={result.get('severity')} | "
        f"run_id={root_run_id[:16]}..."
    )

    return result, root_run_id


# ── FEEDBACK API ───────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Why log feedback?
# Tracing is passive — it records what happened.
# Feedback is active — it records whether what happened was correct.
# This distinction matters enormously for production AI systems.
#
# Feedback is the foundation of your evaluation loop:
#   Trace (what happened) + Feedback (was it right?) = Ground truth dataset
# That dataset is what you use to:
#   - Build LangSmith evaluation datasets ("here are 50 good runs")
#   - Catch regressions when you change prompts
#   - Fine-tune models on correct examples
#   - Build dashboards showing system quality over time
#
# In production, feedback comes from three sources:
#   1. Human reviewers (ops team marks each alert as correctly classified)
#   2. Automated evaluators (LLM judges another LLM's output)
#   3. Downstream signals (alert was escalated = classification was right)
#
# key   : the feedback dimension — "correctness", "severity_accuracy", etc.
#         Use separate keys for separate quality dimensions.
# score : 0.0 (wrong) to 1.0 (correct). 0.5 = uncertain.
def log_feedback(
    run_id:  str,
    correct: bool,
    comment: str = "",
) -> None:
    """Attach human feedback to a completed trace in LangSmith."""
    if run_id == "unknown":
        logger.warning("log_feedback | run_id unknown — skipping")
        return

    langsmith_client = LangSmithClient()
    langsmith_client.create_feedback(
        run_id=run_id,
        key="correctness",
        score=1.0 if correct else 0.0,
        comment=comment,
    )
    logger.info(
        f"log_feedback | run_id={run_id[:16]}... | "
        f"correct={correct} | comment='{comment[:60]}'"
    )


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "═" * 65)
    print("gStore Replenishment Alert Classifier — with LangSmith Tracing")
    print(f"Project : {LANGSMITH_PROJECT}")
    print(f"Model   : {MODEL_NAME}")
    print("═" * 65)

    # ── ALERT 1: SKU COUNT MISMATCH ────────────────────────────────────────────
    # Expected: classification=data_quality, severity=medium
    # In LangSmith look for:
    #   → trace named "alert-PET-NYC-001-petsmart"
    #   → classify node → parse_alert_fields child run (shows sku counts extracted)
    #   → classify node → ChatOpenAI call (exact prompt sent, exact JSON returned)
    #   → recommend node → ChatOpenAI call (recommendation prose)
    #   → metadata panel: store_id="PET-NYC-001", client="PetSmart"
    print("\n[ALERT 1] SKU Count Mismatch — PetSmart PET-NYC-001")
    print("─" * 65)

    result_1, run_id_1 = run_alert(
        alert_text=(
            "REPLENISHMENT ALERT — Store: PET-NYC-001 | Client: PetSmart\n"
            "SKU count mismatch detected in the 14:30 replenishment run.\n"
            "Actual: 1,234 SKUs processed. Expected: 1,450 SKUs.\n"
            "Discrepancy: 216 SKUs. Ingestion job completed without errors.\n"
            "Possible source: GCP bucket sync delay or filter rule change."
        ),
        store_id="PET-NYC-001",
        client="PetSmart",
    )

    print(f"  Classification : {result_1['classification']}")
    print(f"  Severity       : {result_1['severity']}")
    print(f"  Recommendation : {result_1['recommendation']}")
    print(f"  Run ID         : {run_id_1[:20]}...")

    log_feedback(
        run_id_1,
        correct=True,
        comment="Correct — GCP bucket sync delay confirmed by data engineering",
    )

    # ── ALERT 2: ISOLATED STORE OFFLINE ───────────────────────────────────────
    # Expected: classification=ops_failure, severity=high or critical
    # Feedback: marking as incorrect if severity is "high" instead of "critical"
    # — store is actively failing, not at risk. This is the feedback loop in action.
    print("\n[ALERT 2] Store Offline — HMGroup HM-LON-042")
    print("─" * 65)

    result_2, run_id_2 = run_alert(
        alert_text=(
            "CRITICAL — store_id=HM-LON-042 has not received a replenishment file "
            "for the past 3 cycles (15 minutes). Store is currently active and open.\n"
            "Last successful file: 14:15 UTC. Current time: 14:30 UTC.\n"
            "All other London stores (HM-LON-039, HM-LON-040) received files normally.\n"
            "This is an isolated failure — store-level routing or file generation issue."
        ),
        store_id="HM-LON-042",
        client="HMGroup",
    )

    print(f"  Classification : {result_2['classification']}")
    print(f"  Severity       : {result_2['severity']}")
    print(f"  Recommendation : {result_2['recommendation']}")
    print(f"  Run ID         : {run_id_2[:20]}...")

    # Intentionally logging negative feedback if severity is wrong —
    # this is how you build a feedback dataset that captures failure modes.
    severity_correct = result_2["severity"] == "critical"
    log_feedback(
        run_id_2,
        correct=severity_correct,
        comment=(
            "Severity correct — store is actively impacted"
            if severity_correct
            else "Severity WRONG — should be critical, not high; store is open and impacted"
        ),
    )

    # ── ALERT 3: KNOWN FALSE POSITIVE ─────────────────────────────────────────
    # Expected: classification=false_positive, severity=low
    # The seasonal_mode flag is the critical signal — a human would catch this
    # from the config context. The LLM should too. In LangSmith, you can see
    # exactly what prompt the model received and whether it "saw" the flag.
    print("\n[ALERT 3] Seasonal Mode False Positive — FootLocker FL-CHI-007")
    print("─" * 65)

    result_3, run_id_3 = run_alert(
        alert_text=(
            "ALERT — FootLocker store FL-CHI-007: SKU count below threshold.\n"
            "Actual: 980 SKUs. Threshold: 1,000 SKUs. Delta: -20 SKUs.\n"
            "NOTE: This store is running seasonal_mode=true (summer inventory profile).\n"
            "Reduced SKU threshold config has NOT been applied to the alert rules.\n"
            "Expected behaviour — alert threshold should be updated in store config."
        ),
        store_id="FL-CHI-007",
        client="FootLocker",
    )

    print(f"  Classification : {result_3['classification']}")
    print(f"  Severity       : {result_3['severity']}")
    print(f"  Recommendation : {result_3['recommendation']}")
    print(f"  Run ID         : {run_id_3[:20]}...")

    log_feedback(
        run_id_3,
        correct=True,
        comment="Correctly identified as false positive — seasonal_mode flag was recognised",
    )

    # ── FLUSH TRACES ───────────────────────────────────────────────────────────
    # DOMAIN KNOWLEDGE: Why wait_for_all_tracers()?
    # LangSmith uploads traces in background threads — your script can finish
    # and exit before all uploads complete. wait_for_all_tracers() blocks until
    # every background upload thread has finished.
    # Without this: short-lived scripts (like this one) may have traces
    # missing from the UI even though the code ran correctly.
    # In a long-running FastAPI server this isn't needed — the process stays
    # alive. In scripts: always add it at the end.
    wait_for_all_tracers()

    # ── SUMMARY ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("✅  3 alerts processed, traced, and feedback logged.")
    print(f"    View traces : https://smith.langchain.com")
    print(f"    Project     : {LANGSMITH_PROJECT}")
    print("    Filter tips :")
    print("      • Tag = 'gstore'        → all gStore traces")
    print("      • metadata.client       → filter by client name")
    print("      • metadata.store_id     → find one store's traces")
    print("      • Feedback = 0          → all incorrect classifications")
    print("═" * 65)
