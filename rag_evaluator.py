# =============================================================================
# rag_evaluator.py
# =============================================================================
#
# WHAT IS RAG EVALUATION? (Domain Knowledge)
# ─────────────────────────────────────────────────────────────────────────────
# Building a RAG system is easy. Knowing if it WORKS is hard.
#
# Without evaluation, your RAG system might:
#   - Retrieve wrong chunks and the LLM confidently answers from them
#   - Miss the correct chunk and say "I don't know" when it should know
#   - Use general knowledge instead of your documents (hallucination)
#
# We measure quality with 3 metrics:
#
#   1. FAITHFULNESS (0.0 to 1.0)
#      "Does the answer actually come from the retrieved context?"
#      1.0 = every claim in the answer is backed by the context
#      0.0 = the answer contradicts or ignores the context
#      Industry equivalent: "groundedness" or "attribution score"
#
#   2. ANSWER RELEVANCE (0.0 to 1.0)
#      "Does the answer actually address what was asked?"
#      1.0 = directly and completely answers the question
#      0.0 = answers a different question entirely
#
#   3. CONTEXT PRECISION (0.0 to 1.0)
#      "Of the retrieved chunks, how many were actually relevant?"
#      1.0 = every retrieved chunk was useful
#      0.0 = all retrieved chunks were irrelevant noise
#
# HOW WE EVALUATE — LLM-as-Judge Pattern:
# We use a second LLM call to grade the output of the first LLM call.
# The "judge" LLM reads the context, question, and answer, then scores it.
# Industry term: "LLM-as-judge" or "model-based evaluation"
# This is how Anthropic, OpenAI, and most AI labs evaluate their models.
#
# IMPORTANT INSIGHT from your last run:
# faith=1.00 | relevance=0.00 | precision=0.00
#
# This is correct behavior for empty retrieval:
#   faithfulness=1.0  → "I don't have that information" IS faithful to empty context
#   relevance=0.0     → but it does NOT answer "how many days of leave"
#   precision=0.0     → 0 chunks retrieved = 0% precision
#
# The evaluator was working. The retrieval was broken (empty collection).
# Fix the retrieval → evaluation scores will reflect actual quality.
# =============================================================================

import json
import time
import logging
from dataclasses import dataclass   # for creating simple data container classes
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted
from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

# ── LOGGING ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── CLIENTS ───────────────────────────────────────────────────────────────────
# Gemini = primary judge, OpenAI = fallback judge
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────
# PYTHON CONCEPT: @dataclass decorator
# A dataclass automatically generates __init__, __repr__, and other boilerplate.
# Without it, you'd need to write: def __init__(self, question, expected_answer...):
# With it, Python generates all that automatically from the field definitions.
# Think of it as a named tuple with type hints and less boilerplate.

@dataclass
class RAGSample:
    """
    One test case for evaluation.

    DOMAIN KNOWLEDGE: Test Set Design
    A good test set should cover:
      - Happy path: questions clearly answered in docs
      - Edge cases: questions spanning multiple documents
      - Out-of-scope: questions NOT in the docs (test hallucination resistance)
      - Ambiguous: questions with similar phrasing to docs but different meaning

    In production, test sets are built from:
      - Real user queries (logged from production)
      - Domain expert-written Q&A pairs
      - Adversarial examples (questions designed to trip up the system)
    """
    question:         str   # the user's question
    expected_answer:  str   # what the correct answer SHOULD be (ground truth)
    retrieved_chunks: list[str]  # what chunks the RAG actually retrieved
    generated_answer: str   # what answer the RAG actually generated


@dataclass
class EvalResult:
    """Scores and pass/fail verdict for one test case."""
    question:          str
    faithfulness:      float  # 0.0 to 1.0
    answer_relevance:  float  # 0.0 to 1.0
    context_precision: float  # 0.0 to 1.0
    passed:            bool   # did all metrics pass their thresholds?
    failure_reason:    str    # human-readable explanation if failed


# ── JUDGE FUNCTIONS ───────────────────────────────────────────────────────────
# Each function asks the LLM judge ONE specific question about quality.
# We ask three separate questions instead of one big question because:
#   - Simpler prompts = more reliable scores from the judge
#   - Easier to debug which specific metric failed
#   - Each metric addresses a different failure mode

def evaluate_faithfulness(
    context:          str,
    generated_answer: str
) -> float:
    """
    Ask the judge: is the answer grounded in the context?
    Returns score 0.0 to 1.0.

    This catches HALLUCINATION — when the LLM ignores the context
    and answers from its own training knowledge instead.
    """
    prompt = f"""
You are an evaluation judge for a RAG (Retrieval Augmented Generation) system.
Your job is to check if the answer is grounded in the provided context.

Score faithfulness from 0.0 to 1.0:
1.0 = every claim in the answer is explicitly supported by the context
0.5 = some claims supported, some add outside information
0.0 = answer contradicts or completely ignores the context

Special case: if the answer is "I don't have that information" and the
context is empty or doesn't contain the answer, score = 1.0 (correct behavior).

Return ONLY valid JSON with no extra text:
{{"score": <float between 0.0 and 1.0>, "reason": "<one sentence explaining the score>"}}

CONTEXT:
{context if context.strip() else "(no context retrieved)"}

ANSWER TO EVALUATE:
{generated_answer}
""".strip()

    return _call_judge(prompt, metric="faithfulness")


def evaluate_answer_relevance(
    question:         str,
    generated_answer: str,
    expected_answer:  str
) -> float:
    """
    Ask the judge: does the answer address the question correctly?
    Returns score 0.0 to 1.0.

    IMPROVEMENT over original: we now pass expected_answer too.
    The judge can compare the generated answer against what we expected.
    This catches cases where the answer is "faithful" but still wrong.
    """
    prompt = f"""
You are an evaluation judge for a RAG system.
Score how well the generated answer addresses the question.

Compare the generated answer against the expected answer for accuracy.

1.0 = directly and completely answers the question, matches expected answer
0.7 = answers the question but misses some detail from expected answer
0.5 = partially answers or is slightly off-topic
0.0 = does not answer the question, or contradicts the expected answer

Special case: if expected_answer is "NOT IN DOCUMENTS" and the generated
answer says "I don't have that information", score = 1.0 (correct behavior).

Return ONLY valid JSON with no extra text:
{{"score": <float 0.0 to 1.0>, "reason": "<one sentence>"}}

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}
""".strip()

    return _call_judge(prompt, metric="answer_relevance")


def evaluate_context_precision(
    question: str,
    chunks:   list[str]
) -> float:
    """
    Ask the judge: of the retrieved chunks, how many were actually relevant?
    Returns score 0.0 to 1.0.

    This catches RETRIEVAL NOISE — when irrelevant chunks are retrieved
    alongside (or instead of) the relevant ones.

    Special case: if no chunks were retrieved (empty list), returns 0.0.
    (This was the case in your failing eval run — collection was empty.)
    """
    # PYTHON CONCEPT: early return
    # If there are no chunks, there's nothing to evaluate — return 0.0 immediately
    if not chunks:
        return 0.0

    # PYTHON CONCEPT: f-string with enumerate inside a join
    # enumerate(chunks) gives (0, chunk0), (1, chunk1), ...
    # We format each as "[Chunk 1]: text\n\n[Chunk 2]: text\n\n..."
    chunks_formatted = "\n\n".join(
        f"[Chunk {i+1}]: {chunk}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""
You are an evaluation judge for a RAG system.
For each retrieved chunk, decide if it is relevant to answering the question.

A chunk is relevant if it contains information that helps answer the question.
A chunk is NOT relevant if it's about a different topic or policy.

Return ONLY valid JSON with no extra text:
{{
    "relevant_chunk_numbers": [<list of chunk numbers 1-based that are relevant>],
    "score": <float 0.0-1.0, where 1.0 = all chunks relevant, 0.0 = none relevant>,
    "reason": "<one sentence>"
}}

QUESTION:
{question}

RETRIEVED CHUNKS:
{chunks_formatted}
""".strip()

    return _call_judge(prompt, metric="context_precision")


# ── JUDGE CALLER ──────────────────────────────────────────────────────────────

def _call_judge(
    prompt: str,
    metric: str,
    delay:  int = 13
) -> float:
    """
    Call LLM as a judge. Try Gemini first, fall back to OpenAI.

    PYTHON CONCEPT: underscore prefix (_call_judge)
    A function starting with _ is a convention meaning "private" or "internal".
    It signals: "this function is for internal use within this module."
    Python doesn't enforce this — it's just a naming convention.

    Returns:
        float between 0.0 and 1.0 (the evaluation score)
        Returns 0.0 on any error (fail safe — don't crash evaluation)
    """
    if delay > 0:
        time.sleep(delay)  # rate limit buffer

    def parse_score(content: str) -> float:
        """
        Parse the JSON response and extract the score.
        Inner function — only used inside _call_judge.
        """
        result = json.loads(content)
        # PYTHON CONCEPT: dict.get() with fallback chain
        # Try "score" first, then "precision_score", then default to 0.0
        raw_score = result.get("score", result.get("precision_score", 0.0))
        # Clamp score between 0.0 and 1.0 (LLM might return out-of-range values)
        return max(0.0, min(1.0, float(raw_score)))

    # Try Gemini (primary judge)
    try:
        response = gemini_client.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = prompt,
            config   = types.GenerateContentConfig(
                response_mime_type = "application/json"  # forces JSON response
            )
        )
        score = parse_score(response.text)
        logger.debug(f"Judge ({metric}) → {score:.2f} [gemini]")
        return score

    except ResourceExhausted:
        logger.warning(f"Gemini rate limit on judge call ({metric}) — trying OpenAI")
    except json.JSONDecodeError as e:
        logger.error(f"Judge (gemini/{metric}) returned invalid JSON: {str(e)[:80]}")
    except Exception as e:
        logger.warning(f"Gemini judge failed ({metric}): {str(e)[:80]} — trying OpenAI")

    # Try OpenAI as fallback judge
    try:
        response = openai_client.chat.completions.create(
            model    = "gpt-4.1-mini",  # ← gpt-5 replaced with confirmed model
            messages = [
                {
                    "role":    "system",
                    "content": "You are an evaluation judge. Return ONLY valid JSON."
                },
                {
                    "role":    "user",
                    "content": prompt
                }
            ],
            response_format = {"type": "json_object"},  # forces JSON output
            temperature     = 0.1  # very low temperature for consistent scoring
        )
        score = parse_score(response.choices[0].message.content)
        logger.debug(f"Judge ({metric}) → {score:.2f} [openai]")
        return score

    except json.JSONDecodeError as e:
        logger.error(f"Judge (openai/{metric}) returned invalid JSON: {str(e)[:80]}")
        return 0.0

    except Exception as e:
        logger.error(f"All judge providers failed ({metric}): {str(e)[:100]}")
        return 0.0  # fail safe — return 0 rather than crashing evaluation


# ── SAMPLE EVALUATOR ──────────────────────────────────────────────────────────

def evaluate_sample(sample: RAGSample) -> EvalResult:
    """
    Run all 3 evaluation metrics on one test case.

    DOMAIN KNOWLEDGE: Why 3 Separate Metrics?
    Each metric catches a different failure mode:
      - faithfulness catches: LLM hallucinating from its training data
      - answer_relevance catches: LLM answering the wrong question
      - context_precision catches: retrieval returning noise chunks

    If your system has a specific weakness (e.g. always hallucinating),
    one metric will be systematically low — pointing you to the fix.
    """
    logger.info(f"Evaluating: '{sample.question[:55]}'")

    # Join retrieved chunks into one context string for the faithfulness judge
    context = "\n\n".join(sample.retrieved_chunks)

    # Run all 3 metric evaluations
    # Note: each involves an LLM call with a delay → evaluating 10 samples
    # takes ~10 * 3 * 13 seconds = ~6.5 minutes. This is normal.
    faithfulness      = evaluate_faithfulness(context, sample.generated_answer)
    answer_relevance  = evaluate_answer_relevance(
                            sample.question,
                            sample.generated_answer,
                            sample.expected_answer      # ← improved: includes expected
                        )
    context_precision = evaluate_context_precision(sample.question, sample.retrieved_chunks)

    # Pass/fail thresholds — tune these based on your use case and risk tolerance
    # Higher stakes (medical, legal) → raise thresholds
    # Lower stakes (internal FAQ bot) → can lower thresholds
    FAITHFULNESS_THRESHOLD      = 0.7  # 70% of claims must be grounded in context
    ANSWER_RELEVANCE_THRESHOLD  = 0.7  # answer must 70% address the question
    CONTEXT_PRECISION_THRESHOLD = 0.5  # at least 50% of retrieved chunks relevant

    failure_reasons = []
    if faithfulness      < FAITHFULNESS_THRESHOLD:
        failure_reasons.append(f"faithfulness={faithfulness:.2f} (threshold {FAITHFULNESS_THRESHOLD})")
    if answer_relevance  < ANSWER_RELEVANCE_THRESHOLD:
        failure_reasons.append(f"answer_relevance={answer_relevance:.2f} (threshold {ANSWER_RELEVANCE_THRESHOLD})")
    if context_precision < CONTEXT_PRECISION_THRESHOLD:
        failure_reasons.append(f"context_precision={context_precision:.2f} (threshold {CONTEXT_PRECISION_THRESHOLD})")

    passed = len(failure_reasons) == 0

    return EvalResult(
        question          = sample.question,
        faithfulness      = faithfulness,
        answer_relevance  = answer_relevance,
        context_precision = context_precision,
        passed            = passed,
        failure_reason    = " | ".join(failure_reasons) if failure_reasons else "all checks passed"
    )


# ── BATCH EVALUATOR ───────────────────────────────────────────────────────────

def run_evaluation(samples: list[RAGSample]) -> dict:
    """
    Evaluate a full test set and return an aggregated report.

    DOMAIN KNOWLEDGE: Evaluation Report
    This is your RAG system's "report card". Key numbers to watch:
      - pass_rate > 0.8 (80%) = good for most use cases
      - avg_faithfulness > 0.8 = LLM is grounded in your documents
      - avg_answer_relevance > 0.75 = questions are being answered correctly
      - avg_context_precision > 0.6 = retrieval is finding relevant chunks

    Run this after EVERY significant change to your system:
      - After changing chunking strategy
      - After updating prompts
      - After adding new documents
      - After changing embedding model
    Industry term: "regression testing" — making sure changes don't break things.
    """
    results = []
    for sample in samples:
        result = evaluate_sample(sample)
        results.append(result)

        # Print result immediately so you can see progress
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(
            f"{status} | "
            f"faith={result.faithfulness:.2f} | "
            f"relevance={result.answer_relevance:.2f} | "
            f"precision={result.context_precision:.2f} | "
            f"{result.question[:50]}"
        )

    # PYTHON CONCEPT: list comprehension with sum()
    # sum(1 for r in results if r.passed) counts how many results have passed=True
    # It's equivalent to: count = 0; for r in results: if r.passed: count += 1
    total  = len(results)
    passed = sum(1 for r in results if r.passed)

    report = {
        "total_samples":        total,
        "passed":               passed,
        "failed":               total - passed,
        "pass_rate":            round(passed / total, 3),
        "avg_faithfulness":     round(sum(r.faithfulness     for r in results) / total, 3),
        "avg_answer_relevance": round(sum(r.answer_relevance for r in results) / total, 3),
        "avg_context_precision":round(sum(r.context_precision for r in results) / total, 3),
        "failures": [
            {
                "question": r.question,
                "reason":   r.failure_reason,
                "scores": {
                    "faithfulness":      r.faithfulness,
                    "answer_relevance":  r.answer_relevance,
                    "context_precision": r.context_precision
                }
            }
            for r in results if not r.passed
        ]
    }

    return report