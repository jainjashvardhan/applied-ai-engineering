"""
langsmith_07_evaluation.py
═══════════════════════════════════════════════════════════════════════════════

PURPOSE  : AI Evaluation — measure system quality systematically using
           labeled datasets, rule-based evaluators, and LLM-as-judge.

USE CASE : gStore Alert Classifier Evaluation
           Ground truth dataset of 6 labeled alerts.
           3 evaluators: classification_accuracy, severity_accuracy,
           recommendation_quality (LLM-as-judge).
           Two experiments to show eval-driven development loop.

WHY EVALUATION (not just tracing):
  Tracing records what happened.
  Evaluation answers: did this change improve or hurt quality?
  Without a dataset, every prompt change is a guess.
  With eval, quality is a number you can compare across versions.

WORKFLOW:
  1. Create dataset (once)  →  ground truth in LangSmith
  2. Run Experiment A        →  baseline scores
  3. Improve the prompt      →  edit CLASSIFICATION_SYSTEM_PROMPT
  4. Run Experiment B        →  new scores
  5. Compare side-by-side   →  ship only if scores improved

RUN      : python langsmith_07_evaluation.py
REQUIRES : pip install langsmith langchain-openai langgraph pandas
           OPENAI_API_KEY, LANGCHAIN_API_KEY in .env
           LANGCHAIN_TRACING_V2=true, LANGCHAIN_PROJECT=gstore-ai-dev in .env
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────
import json
import logging
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import Client as LangSmithClient
from langsmith.evaluation import evaluate
from typing_extensions import TypedDict

# ── ENVIRONMENT + LOGGING ──────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── CONSTANTS ──────────────────────────────────────────────────────────────────
MODEL_NAME        = "gpt-5.4-mini"
DATASET_NAME      = "gstore-alert-classification-v2"
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "gstore-ai-dev")

# ── LLM INSTANCES ─────────────────────────────────────────────────────────────
# Two separate instances — one for the classifier, one for the judge.
# Keeping them separate means you can swap the judge model independently
# of the model under evaluation. This matters for Q5 in the challenge questions.
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# DOMAIN KNOWLEDGE: Why temperature=0 for the judge?
# The classification judge should be deterministic — given the same output,
# it should always return the same score. Temperature=0 maximises this.
# The classifier uses 0.1 (allows slight variation for better JSON formatting).
judge_llm = ChatOpenAI(
    model=MODEL_NAME,      # same model as classifier — Challenge Q5 asks about this
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ── STATE SCHEMA ───────────────────────────────────────────────────────────────
class AlertState(TypedDict):
    alert_text:     str
    store_id:       str
    client:         str
    classification: str
    severity:       str
    recommendation: str


# ── PROMPT CONSTANTS ──────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Why extract prompts to module-level constants?
# The whole point of evaluation is to measure the effect of prompt changes.
# If the prompt is buried inside a function, you have to hunt for it.
# As a constant at the top of the file, you edit it in one place, re-run,
# and compare scores. This is the eval-driven development workflow made concrete.
#
# Experiment A uses CLASSIFICATION_SYSTEM_PROMPT_V1 (the baseline).
# Experiment B uses CLASSIFICATION_SYSTEM_PROMPT_V2 (improved for false positives).
# In a real project you'd have one prompt and edit it between runs — we define
# both here so you can run both experiments in a single script execution.

CLASSIFICATION_SYSTEM_PROMPT_V1 = (
    "You are a gStore replenishment QA engineer. "
    "Classify the incoming alert.\n"
    "Return ONLY valid JSON — no markdown, no explanation:\n"
    '{{"classification": "data_quality|ops_failure|config_error|false_positive", '
    '"severity": "low|medium|high|critical", '
    '"reason": "<one concise sentence>"}}'
)

# V2: Adds explicit instruction to look for config/seasonal flags.
# Hypothesis: the model is misclassifying config_error and false_positive
# alerts as data_quality because the base prompt gives no guidance on those
# categories. Adding examples of what signals each category should improve
# those two labels without hurting the other two.
CLASSIFICATION_SYSTEM_PROMPT_V2 = (
    "You are a gStore replenishment QA engineer. "
    "Classify the incoming alert.\n\n"
    "Classification guide:\n"
    "  data_quality   — SKU count mismatch, ingestion delay, pipeline truncation\n"
    "  ops_failure    — store offline, job crashed, file never arrived\n"
    "  config_error   — wrong threshold, missing config, onboarding gap\n"
    "  false_positive — alert fired correctly but situation is expected "
    "(seasonal_mode, planned maintenance, known profile)\n\n"
    "Return ONLY valid JSON — no markdown, no explanation:\n"
    '{{"classification": "data_quality|ops_failure|config_error|false_positive", '
    '"severity": "low|medium|high|critical", '
    '"reason": "<one concise sentence>"}}'
)


# ── GRAPH BUILDER ─────────────────────────────────────────────────────────────
# Accepts the system prompt as a parameter so we can build two versions
# of the classifier — one per experiment — without duplicating node logic.
# PYTHON CONCEPT: Default argument
# system_prompt=CLASSIFICATION_SYSTEM_PROMPT_V1 means if you call
# build_classifier() with no arguments, you get the V1 (baseline) graph.
# Pass a different prompt to get a different graph.
def build_classifier(system_prompt: str = CLASSIFICATION_SYSTEM_PROMPT_V1) -> StateGraph:
    """Build and compile the alert classifier graph with the given prompt."""

    def classify_alert(state: AlertState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Alert: {alert_text}"),
        ])
        chain = prompt | llm | StrOutputParser()
        try:
            raw    = chain.invoke({"alert_text": state["alert_text"]})
            result = json.loads(raw.strip())
        except json.JSONDecodeError:
            logger.error(f"classify | non-JSON response: {raw[:200]}")
            result = {"classification": "data_quality", "severity": "medium", "reason": "parse error"}

        return {
            "classification": result.get("classification", "data_quality"),
            "severity":       result.get("severity", "medium"),
        }

    def generate_recommendation(state: AlertState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a gStore ops lead. Given a classified alert, produce a concise "
                    "action recommendation in 2–3 sentences. "
                    "Name the specific team (data engineering, store ops, QA), "
                    "system (BigQuery ingestion job, store config table, GCP bucket), "
                    "or tool to contact. Do not use bullet points."
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

    builder = StateGraph(AlertState)
    builder.add_node("classify", classify_alert)
    builder.add_node("recommend", generate_recommendation)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "recommend")
    builder.add_edge("recommend", END)
    return builder.compile()


# Build both versions up front
classifier_v1 = build_classifier(CLASSIFICATION_SYSTEM_PROMPT_V1)
classifier_v2 = build_classifier(CLASSIFICATION_SYSTEM_PROMPT_V2)


# ── DATASET ────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: What makes a good evaluation dataset?
#
# Coverage: every failure mode you care about should have at least one example.
#   Below: all 4 classification types × 3 severity levels are represented.
#
# Difficulty distribution: include easy examples (clear signals) AND hard ones
#   (ambiguous signals, edge cases). A dataset of only easy examples gives you
#   100% accuracy on things that were never going to fail.
#
# Size: start small (6–20), validate the pipeline works, then grow.
#   6 examples is not statistically robust but is enough to catch obvious regressions.
#   In production: aim for 50+ examples before making deployment decisions.
#
# Source: ideally curated from real production traces + human-verified labels.
#   Don't use the same traces you evaluated for initial feedback (contamination —
#   see Challenge Q4). 

EVAL_EXAMPLES = [
    # ── data_quality / medium ─────────────────────────────────────────────────
    {
        "inputs": {
            "alert_text": (
                "ALERT — Store PET-NYC-001 | Client: PetSmart\n"
                "SKU count mismatch in 14:30 replenishment run. "
                "Actual: 1,234. Expected: 1,450. Delta: 216 SKUs. "
                "Ingestion job completed without errors. "
                "Possible cause: GCP bucket sync delay."
            ),
            "store_id": "PET-NYC-001",
            "client":   "PetSmart",
        },
        "outputs": {"classification": "data_quality", "severity": "medium"},
    },
    # ── ops_failure / critical ────────────────────────────────────────────────
    {
        "inputs": {
            "alert_text": (
                "CRITICAL — store_id=HM-LON-042 has not received a replenishment file "
                "for 3 consecutive cycles (15 minutes). Store is open and active. "
                "All other HM London stores are receiving files normally. Isolated failure."
            ),
            "store_id": "HM-LON-042",
            "client":   "HMGroup",
        },
        "outputs": {"classification": "ops_failure", "severity": "critical"},
    },
    # ── false_positive / low ──────────────────────────────────────────────────
    {
        "inputs": {
            "alert_text": (
                "ALERT — FL-CHI-007 SKU count below threshold. "
                "Actual: 980. Threshold: 1,000. "
                "seasonal_mode=true is active for this store. "
                "Reduced inventory profile is expected — threshold config not updated."
            ),
            "store_id": "FL-CHI-007",
            "client":   "FootLocker",
        },
        "outputs": {"classification": "false_positive", "severity": "low"},
    },
    # ── ops_failure / high ────────────────────────────────────────────────────
    {
        "inputs": {
            "alert_text": (
                "ERROR — BigQuery ingestion job for Fabletics failed at 16:00 UTC. "
                "No replenishment data loaded for the 16:00 cycle. "
                "Job error: quota exceeded on dataset fabletics_replen_prod. "
                "4 stores affected: FAB-NYC-001 through FAB-NYC-004."
            ),
            "store_id": "FAB-NYC-001",
            "client":   "Fabletics",
        },
        "outputs": {"classification": "ops_failure", "severity": "high"},
    },
    # ── config_error / low ────────────────────────────────────────────────────
    {
        "inputs": {
            "alert_text": (
                "ALERT — New store DEC-LA-009 (Deckers, LA) opened 6 days ago. "
                "SKU threshold is set to 500 but store carries 1,200 SKUs. "
                "Alert firing every cycle because threshold was never updated "
                "during store onboarding. No actual replenishment failure."
            ),
            "store_id": "DEC-LA-009",
            "client":   "Deckers",
        },
        "outputs": {"classification": "config_error", "severity": "low"},
    },
    # ── data_quality / high ───────────────────────────────────────────────────
    {
        "inputs": {
            "alert_text": (
                "HIGH — PetSmart PET-CHI-017: multi-category SKU discrepancy. "
                "Pet food: actual=890, expected=1,100. "
                "Toys: actual=340, expected=450. "
                "Electronics: actual=210, expected=290. "
                "Total delta: 400 SKUs across 3 categories. "
                "Pattern suggests upstream data truncation in the ingestion pipeline."
            ),
            "store_id": "PET-CHI-017",
            "client":   "PetSmart",
        },
        "outputs": {"classification": "data_quality", "severity": "high"},
    },
]


def get_or_create_dataset(client: LangSmithClient) -> str:
    """
    Return the dataset name if it already exists, otherwise create and populate it.

    DOMAIN KNOWLEDGE: Idempotency pattern
    Safe to call on every script run — if the dataset exists, this no-ops.
    Only creates the dataset on the first run.
    In production: add to your CI pipeline startup so the dataset always exists
    before eval runs, without creating duplicates.
    """
    existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]
    if existing:
        logger.info(
            f"Dataset '{DATASET_NAME}' already exists "
            f"— skipping creation"
        )
        return DATASET_NAME

    logger.info(f"Creating dataset '{DATASET_NAME}' with {len(EVAL_EXAMPLES)} examples...")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "gStore alert classification ground truth. "
            "6 labeled examples: all 4 alert types, 3 severity levels."
        ),
    )

    # PYTHON CONCEPT: list comprehension
    # [expression for item in list] creates a new list by applying expression
    # to each item. Here we split EVAL_EXAMPLES into two parallel lists:
    # one of inputs, one of expected outputs.
    client.create_examples(
        inputs=[ex["inputs"] for ex in EVAL_EXAMPLES],
        outputs=[ex["outputs"] for ex in EVAL_EXAMPLES],
        dataset_id=dataset.id,
    )

    logger.info(f"✅ Dataset created: {len(EVAL_EXAMPLES)} examples")
    return DATASET_NAME


# ── TARGET FUNCTION FACTORY ───────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: The target function
# The target is your application. evaluate() calls it once per dataset example,
# passing the example's `inputs` dict. The return value is the "actual output"
# that evaluators will score against the "expected output" in the dataset.
#
# Requirements for the target function:
#   - Accepts a single dict argument (the inputs from the dataset example)
#   - Returns a dict (the actual outputs your evaluators will score)
#   - Should be deterministic given the same input (temperature=0.1 introduces
#     slight variance — acceptable for a dev demo, lower in production eval)
#
# PYTHON CONCEPT: Factory function
# make_target(classifier) returns a function (not a result).
# We use this pattern so we can create two versions of the target — one for
# each classifier version — without duplicating the wrapper logic.
# When you call make_target(classifier_v1), you get back a function called
# target that, when called, runs classifier_v1.
def make_target(classifier: StateGraph, experiment_label: str):
    """Factory: returns a target function that wraps the given classifier graph."""

    def target(inputs: dict) -> dict:
        initial_state: AlertState = {
            "alert_text":     inputs["alert_text"],
            "store_id":       inputs["store_id"],
            "client":         inputs["client"],
            "classification": "",
            "severity":       "",
            "recommendation": "",
        }

        result: AlertState = classifier.invoke(
            initial_state,
            config={
                "run_name": f"{experiment_label}-{inputs['store_id']}",
                "metadata": {
                    "store_id":    inputs["store_id"],
                    "client":      inputs["client"],
                    "environment": "eval",    # tag as eval, not prod
                    "experiment":  experiment_label,
                },
                "tags": ["gstore", "evaluation", experiment_label],
            },
        )

        return {
            "classification": result["classification"],
            "severity":       result["severity"],
            "recommendation": result["recommendation"],
        }

    # Name the function dynamically so LangSmith shows a useful label
    target.__name__ = f"target_{experiment_label}"
    return target


# ── EVALUATORS ────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Evaluator function signatures
#
# LangSmith injects arguments by parameter NAME, not position.
# Use these parameter names exactly:
#
#   outputs           — what your target function returned
#   reference_outputs — the expected output from the dataset
#   inputs            — the input from the dataset example
#
# Rule-based evaluators typically use (outputs, reference_outputs).
# LLM-as-judge evaluators typically use (inputs, outputs) — they don't
# compare against a reference; they judge quality against criteria.
#
# Return format: {"key": "metric_name", "score": float, "comment": "optional"}
# score: 0.0 (wrong/poor) to 1.0 (correct/excellent)

def classification_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    """
    Rule-based: did the classifier get the alert type right?
    Exact string match — classification is categorical.
    """
    predicted = outputs.get("classification", "").lower().strip()
    expected  = reference_outputs.get("classification", "").lower().strip()

    return {
        "key":     "classification_accuracy",
        "score":   1.0 if predicted == expected else 0.0,
        "comment": f"predicted='{predicted}' | expected='{expected}'",
    }


def severity_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    """
    Rule-based: did the classifier assign the right severity level?
    Exact match — severity is a defined categorical scale.
    """
    predicted = outputs.get("severity", "").lower().strip()
    expected  = reference_outputs.get("severity", "").lower().strip()

    return {
        "key":     "severity_accuracy",
        "score":   1.0 if predicted == expected else 0.0,
        "comment": f"predicted='{predicted}' | expected='{expected}'",
    }


def recommendation_quality(inputs: dict, outputs: dict) -> dict:
    """
    LLM-as-judge: is the recommendation specific and actionable?

    We use an LLM here because there is no single correct recommendation string.
    A good recommendation can be worded many ways — any version that names a team,
    system, or action step is correct.
    The judge scores against criteria (specificity + actionability), not against
    a reference string.

    KNOWN LIMITATION: we're using the same model (gpt-5.4-mini) to generate and
    to judge. Challenge Q5 asks you to reason through why this is a problem
    and how you'd fix it in production.
    """
    judge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are evaluating a gStore ops recommendation. "
                "Score it 1 if ALL of these are true:\n"
                "  - Names a specific team (data engineering, store ops, QA) OR system\n"
                "  - Describes a concrete action (check job X, update config Y, escalate to Z)\n"
                "  - Is appropriate for the alert described\n"
                "Score it 0 if the recommendation is not crystel clear "
                "('investigate further', 'contact the team' without specifying which).\n"
                "Return ONLY valid JSON: "
                '{{"score": 0 or 1, "reason": "<one sentence>"}}'
            ),
        ),
        (
            "human",
            "Alert: {alert_text}\nRecommendation: {recommendation}",
        ),
    ])

    chain = judge_prompt | judge_llm | StrOutputParser()

    try:
        raw    = chain.invoke({
            "alert_text":     inputs.get("alert_text", ""),
            "recommendation": outputs.get("recommendation", ""),
        })
        result = json.loads(raw.strip())
        return {
            "key":     "recommendation_quality",
            "score":   float(result.get("score", 0.0)),
            "comment": result.get("reason", ""),
        }
    except Exception as e:
        logger.error(f"recommendation_quality judge failed: {e}")
        return {
            "key":     "recommendation_quality",
            "score":   0.0,
            "comment": f"Judge error: {str(e)[:80]}",
        }


# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
def print_eval_summary(results, experiment_name: str) -> None:
    """Print evaluation scores to console in a readable format."""
    print(f"\n{'═' * 65}")
    print(f"EXPERIMENT: {experiment_name}")
    print("═" * 65)

    try:
        df = results.to_pandas()

        # Feedback columns are named "feedback.metric_name" in the DataFrame
        score_cols = [c for c in df.columns if c.startswith("feedback.")]

        # Per-example breakdown
        id_col = next(
            (c for c in df.columns if "store_id" in c.lower()),
            None
        )
        print("\nPer-example scores:")
        show_cols = ([id_col] if id_col else []) + score_cols
        print(df[show_cols].to_string(index=False))

        # Aggregate summary
        print("\nAggregate scores (mean across all examples):")
        for col in score_cols:
            metric_name = col.replace("feedback.", "")
            mean_score  = df[col].mean()
            bar         = "█" * int(mean_score * 10) + "░" * (10 - int(mean_score * 10))
            print(f"  {metric_name:<30} {bar}  {mean_score:.0%}")

    except Exception as e:
        logger.warning(f"Could not render DataFrame: {e}")
        for r in results:
            print(r)

    print(f"\nView in LangSmith → Datasets & Experiments → {experiment_name}")
    print("═" * 65)


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    langsmith_client = LangSmithClient()

    # ── STEP 1: DATASET ────────────────────────────────────────────────────────
    # Idempotent — creates once, no-ops on subsequent runs.
    dataset_name = get_or_create_dataset(langsmith_client)
    print(f"\n✅ Dataset ready: '{dataset_name}' ({len(EVAL_EXAMPLES)} examples)")

    evaluators = [
        classification_accuracy,
        severity_accuracy,
        recommendation_quality,  # LLM-as-judge — adds ~2 LLM calls per example
    ]

    # ── STEP 2: EXPERIMENT A — BASELINE ────────────────────────────────────────
    # DOMAIN KNOWLEDGE: experiment_prefix
    # This is the label shown in LangSmith's "Datasets & Experiments" view.
    # It becomes "baseline-gpt5_4_mini-<timestamp>" in the UI.
    # Include model name so you can see at a glance what each experiment used.
    # Include a semantic label (baseline, v2-false-positive-fix) so the diff
    # between experiments is immediately obvious without clicking in.
    print(f"\n{'─' * 65}")
    print(f"Running Experiment A — BASELINE (V1 prompt, {MODEL_NAME})")
    print(f"  {len(EVAL_EXAMPLES)} examples × 2 LLM calls (classify + recommend)")
    print(f"  + {len(EVAL_EXAMPLES)} LLM judge calls = ~{len(EVAL_EXAMPLES) * 3} total calls")
    print(f"  max_concurrency=1 (sequential) — ~60–90s estimated")
    print("─" * 65)

    results_v1 = evaluate(
        make_target(classifier_v1, "baseline-v1"),
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"baseline-{MODEL_NAME.replace('.', '_')}",
        max_concurrency=1,   # sequential — avoids rate limits on free tier
    )

    print_eval_summary(results_v1, f"baseline-{MODEL_NAME}")
    wait_for_all_tracers()

    # ── STEP 3: EXPERIMENT B — IMPROVED PROMPT ─────────────────────────────────
    # DOMAIN KNOWLEDGE: What changed?
    # V2 prompt adds per-category guidance: explicit signals for config_error and
    # false_positive. Hypothesis: the baseline prompt's lack of category guidance
    # causes the model to default to data_quality for ambiguous alerts.
    # We measure whether the hypothesis is correct by comparing scores.
    # If classification_accuracy improves and the other metrics hold, ship V2.
    print(f"\n{'─' * 65}")
    print("Running Experiment B — V2 PROMPT (added per-category guidance)")
    print("─" * 65)

    results_v2 = evaluate(
        make_target(classifier_v2, "v2-category-guidance"),
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"v2-category-guidance-{MODEL_NAME.replace('.', '_')}",
        max_concurrency=1,
    )

    print_eval_summary(results_v2, "v2-category-guidance")
    wait_for_all_tracers()

    # ── STEP 4: COMPARE ────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print("COMPARISON: Baseline vs V2")
    print("═" * 65)

    try:
        import pandas as pd

        df_v1 = results_v1.to_pandas()
        df_v2 = results_v2.to_pandas()

        score_cols = [c for c in df_v1.columns if c.startswith("feedback.")]

        for col in score_cols:
            metric    = col.replace("feedback.", "")
            score_v1  = df_v1[col].mean()
            score_v2  = df_v2[col].mean()
            delta     = score_v2 - score_v1
            direction = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
            print(
                f"  {metric:<30} "
                f"A={score_v1:.0%}  →  B={score_v2:.0%}  "
                f"{direction} {abs(delta):.0%}"
            )

    except Exception as e:
        logger.warning(f"Comparison failed: {e}")

    print(f"\n{'─' * 65}")
    print("In LangSmith UI — 'Datasets & Experiments':")
    print("  Click both experiments → 'Compare' button → side-by-side diff")
    print("  Failed examples are highlighted — click to see the exact trace")
    print("═" * 65)
