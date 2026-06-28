# =============================================================================
# eval_test_set.py
# =============================================================================
#
# WHAT THIS FILE DOES:
# Builds a test set of question/answer pairs and runs the RAG evaluator
# against your actual system to get quality metrics.
#
# WHY YOUR LAST RUN GOT 0 CHUNKS:
# ─────────────────────────────────────────────────────────────────────────────
# The previous version did:
#   from rag_with_metadata import search, ask
#
# Python IMPORTS run module-level code but do NOT run
# the "if __name__ == '__main__':" block.
#
# That means index_documents() was never called.
# The ChromaDB collection existed but was empty.
# Every search returned 0 chunks.
# Every answer was "I don't have that information."
# The evaluator scored faithfulness=1.0 (correct for empty context)
# but relevance=0.0 (because "I don't know" ≠ "24 days of leave").
#
# THE FIX:
# Call ensure_indexed() at the TOP of this file — before any search runs.
# ensure_indexed() checks if the collection is empty and indexes if needed.
# ─────────────────────────────────────────────────────────────────────────────

from rag_with_metadata import (
    search,             # function to retrieve chunks from ChromaDB
    ask,                # function to run the full RAG pipeline
    ensure_indexed,     # function to guarantee data exists before searching
    raw_documents,      # the list of documents to index (if needed)
)
from rag_evaluator import RAGSample, run_evaluation


def build_test_set() -> list[RAGSample]:
    """
    Build a list of RAGSample test cases.

    DOMAIN KNOWLEDGE: What Makes a Good Test Set?
    ─────────────────────────────────────────────────────────────────────────
    A test set should cover 4 categories:

    1. HAPPY PATH (easy, direct answer in docs)
       The question uses similar words to the document.
       Retrieval should easily find the right chunk.

    2. SEMANTIC DISTANCE (answer is there but phrased differently)
       "How many days off do I get?" → should find "24 days of annual leave"
       Tests whether embeddings capture meaning, not just keywords.

    3. OUT-OF-SCOPE (answer is NOT in documents)
       Critical for testing hallucination resistance.
       System should say "I don't know", not make something up.
       Expected answer = "NOT IN DOCUMENTS" (our special marker)

    4. MULTI-DOCUMENT (answer spans two policies)
       "Can probationary employees WFH?" → needs Leave Policy + WFH Policy
       Tests whether retrieval finds relevant chunks across documents.
    ─────────────────────────────────────────────────────────────────────────

    In production, this test set would come from:
      - Real user queries logged from production traffic
      - Domain expert (HR team) writing Q&A pairs
      - Manual annotation of edge cases
    Industry term: "golden dataset" or "ground truth test set"
    """
    test_questions = [
        # ── HAPPY PATH ────────────────────────────────────────────────────────
        {
            "question":       "How many days of annual leave do employees get?",
            "expected_answer": "24 days of paid annual leave per year"
        },
        {
            "question":       "What is the meal allowance for client visits?",
            "expected_answer": "Rs 1500 per meal for client visits"
        },

        # ── SEMANTIC DISTANCE ─────────────────────────────────────────────────
        {
            "question":       "Can I work from home in my first week?",
            "expected_answer": "No, WFH is not permitted during the first 90 days"
        },
        {
            "question":       "What happens if the company rescinds my job offer?",
            "expected_answer": "One month's offered salary as compensation"
            # Note: "rescinds" ≠ "withdraws" — tests semantic search quality
        },

        # ── MULTI-DOCUMENT ────────────────────────────────────────────────────
        {
            "question":       "Can a probationary employee work from home?",
            "expected_answer": "No, WFH is not allowed during the first 90 days (probation period)"
            # Answer requires connecting WFH Policy + Probation context in Leave Policy
        },

        # ── OUT-OF-SCOPE (hallucination tests) ───────────────────────────────
        {
            "question":       "What is the salary structure?",
            "expected_answer": "NOT IN DOCUMENTS"
            # If system answers this, it's hallucinating
        },
        {
            "question":       "What is the company's stock option policy?",
            "expected_answer": "NOT IN DOCUMENTS"
            # Another topic not in our knowledge base
        },
    ]

    samples = []

    print(f"\nBuilding {len(test_questions)} test samples...")
    print("(Each sample runs a real search + ask against your RAG system)\n")

    for i, item in enumerate(test_questions, 1):
        print(f"  [{i}/{len(test_questions)}] {item['question'][:60]}...")

        # ── RETRIEVE ──────────────────────────────────────────────────────────
        # We call search() directly to capture the raw chunks
        # This lets the evaluator judge context_precision separately
        chunks = search(
            query       = item["question"],
            top_k       = 3,
            active_only = True   # always True in production
        )

        # ── GENERATE ──────────────────────────────────────────────────────────
        # We call ask() to get the actual generated answer
        answer = ask(item["question"])

        # Extract just the text from each chunk (evaluator needs plain strings)
        context_texts = [c["text"] for c in chunks]

        # PYTHON CONCEPT: appending to a list
        # .append() adds one item to the end of the list
        samples.append(RAGSample(
            question          = item["question"],
            expected_answer   = item["expected_answer"],
            retrieved_chunks  = context_texts,
            generated_answer  = answer
        ))

    return samples


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CRITICAL FIX ──────────────────────────────────────────────────────────
    # ensure_indexed() checks if ChromaDB has data.
    # If the collection is empty (e.g. first run, or fresh chroma_db_v2 folder),
    # it automatically calls index_documents() to populate it.
    #
    # WITHOUT THIS: every search returns 0 chunks → all answers are
    # "I don't have that information" → evaluation scores are meaningless.
    #
    # WITH THIS: data is guaranteed to exist before any search runs.
    print("Checking knowledge base...")
    ensure_indexed()

    # ── BUILD TEST SAMPLES ────────────────────────────────────────────────────
    print("\nBuilding test samples...")
    samples = build_test_set()

    # ── RUN EVALUATION ────────────────────────────────────────────────────────
    # Each sample requires 3 LLM judge calls (one per metric)
    # With 7 samples × 3 calls × 13 second delay = ~4.5 minutes
    # This is normal — evaluation is intentionally thorough
    print(f"\nRunning evaluation on {len(samples)} samples...")
    print("(This will take a few minutes — each sample needs 3 judge calls)\n")
    print("─" * 70)

    report = run_evaluation(samples)

    # ── PRINT REPORT ──────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("EVALUATION REPORT")
    print("═" * 70)

    # PYTHON CONCEPT: f-string with format spec
    # {value * 100:.1f} means: multiply by 100, format with 1 decimal place
    print(f"Pass rate            : {report['pass_rate'] * 100:.1f}%  ({report['passed']}/{report['total_samples']} passed)")
    print(f"Avg faithfulness     : {report['avg_faithfulness']:.3f}  (target > 0.70)")
    print(f"Avg answer relevance : {report['avg_answer_relevance']:.3f}  (target > 0.70)")
    print(f"Avg context precision: {report['avg_context_precision']:.3f}  (target > 0.50)")

    # ── HOW TO INTERPRET RESULTS ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("HOW TO READ THESE RESULTS:")
    print("─" * 70)
    print("faithfulness LOW     → LLM is hallucinating from training data")
    print("                       Fix: strengthen 'only use context' in prompt")
    print("answer_relevance LOW → Retrieval is finding wrong documents")
    print("                       Fix: improve chunking or embedding model")
    print("context_precision LOW→ Too much noise being retrieved")
    print("                       Fix: raise similarity threshold or add metadata filters")

    if report["failures"]:
        print(f"\nFailed samples ({report['failed']}):")
        for f in report["failures"]:
            print(f"\n  ❌ '{f['question'][:55]}'")
            print(f"     Reason: {f['reason']}")
            # Show individual scores for failed cases
            s = f["scores"]
            print(
                f"     Scores: faith={s['faithfulness']:.2f} | "
                f"relevance={s['answer_relevance']:.2f} | "
                f"precision={s['context_precision']:.2f}"
            )
    else:
        print("\n✅ All samples passed!")