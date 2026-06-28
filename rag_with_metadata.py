# =============================================================================
# rag_with_metadata.py
# =============================================================================
#
# WHAT THIS FILE DOES:
# This file is the core RAG engine. It handles:
#   1. Setting up ChromaDB with the correct distance metric
#   2. Indexing documents (chunking + embedding + storing)
#   3. Searching with metadata filters
#   4. Generating answers using Gemini (fallback to OpenAI)
#
# IMPORTANT — PYTHON IMPORT CONCEPT:
# This file is designed to be IMPORTED by other scripts (like eval_test_set.py).
# The if __name__ == "__main__": block at the bottom only runs when you execute
# this file directly (python3 rag_with_metadata.py).
# It does NOT run when another file does: from rag_with_metadata import search
#
# This means indexing must be called EXPLICITLY before searching.
# We handle this with the ensure_indexed() function below.
# =============================================================================

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted
from openai import OpenAI

import os
import time
import logging
from dotenv import load_dotenv

# ── ENVIRONMENT SETUP ─────────────────────────────────────────────────────────
# load_dotenv() reads your .env file and makes variables available via os.getenv()
# ALWAYS do this before reading any API keys
load_dotenv()

# ── LOGGING ───────────────────────────────────────────────────────────────────
# PYTHON CONCEPT: logging vs print()
# logging adds timestamp + severity level automatically
# In production, these logs go to monitoring systems (Datadog, CloudWatch etc.)
# Industry term: "structured logging" or "observability"
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
# __name__ = the name of this module = "rag_with_metadata"
# This means log lines will show which file generated them

# ── CLIENTS ───────────────────────────────────────────────────────────────────
# PYTHON CONCEPT: Module-level variables
# These lines run ONCE when the module is first imported or run.
# Creating clients at module level = they're shared across all function calls.
# This is more efficient than creating a new client per function call.

# Free local embedding model — converts text to numbers (vectors)
# Downloads ~90MB on first run, then cached locally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini — our primary LLM (default for all generation tasks)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# OpenAI — our fallback LLM (used when Gemini fails or rate-limits)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB — local vector database, data persists in ./chroma_db_v2 folder
# PersistentClient = data survives between runs (unlike in-memory client)
chroma_client = chromadb.PersistentClient(path="./chroma_db_v2")

# ── COLLECTION SETUP ──────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: What is a Vector Database Collection?
# Think of a collection like a table in a SQL database, but instead of rows,
# it stores vectors (embeddings). Each entry has:
#   - An ID (unique identifier)
#   - A document (the original text)
#   - An embedding (list of numbers representing the text's meaning)
#   - Metadata (key-value pairs like department, is_active, version)
#
# CRITICAL: hnsw:space = "cosine"
# ChromaDB defaults to L2 (Euclidean) distance, which breaks our
# similarity formula (1 - distance gives negative scores for L2).
# Cosine distance is bounded [0, 2], so:
#   1 - 0.1 = 0.9  (strong match)
#   1 - 1.0 = 0.0  (unrelated)
#   1 - 1.8 = -0.8 (opposite meaning)
# Industry term: HNSW = Hierarchical Navigable Small World (the indexing algorithm)

def get_or_recreate_collection(name: str) -> chromadb.Collection:
    """
    Load an existing collection OR create a new one with cosine distance.

    PYTHON CONCEPT: try/except
    We 'try' to load the collection. If it fails (doesn't exist) or has
    the wrong settings, we 'except' (catch) the error and create a fresh one.

    Why this function exists:
    If you ran an older version of this code, your collection was created with
    the wrong distance metric (L2). This function detects that and recreates it.
    """
    try:
        col = chroma_client.get_collection(name)

        # PYTHON CONCEPT: dict.get() with a default
        # col.metadata.get("hnsw:space", "l2") means:
        # "get the value of hnsw:space key, or return 'l2' if it doesn't exist"
        if col.metadata.get("hnsw:space") != "cosine":
            logger.warning(
                f"Collection '{name}' has wrong metric "
                f"({col.metadata.get('hnsw:space', 'l2')}). "
                f"Recreating with cosine."
            )
            chroma_client.delete_collection(name)
            raise ValueError("wrong metric")  # fall through to create

        logger.info(f"Loaded collection '{name}' (cosine distance) ✅")
        return col

    except Exception:
        logger.info(f"Creating collection '{name}' with cosine distance")
        return chroma_client.create_collection(
            name     = name,
            metadata = {
                "description": "Company HR and Finance policies",
                "hnsw:space":  "cosine"   # ← the critical setting
            }
        )


# Create the collection at module load time
collection = get_or_recreate_collection("company_knowledge")

# ── DOCUMENTS ─────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Knowledge Base Design
# In production, these would come from:
#   - PDFs parsed with PyMuPDF or pdfplumber
#   - Confluence or Notion pages via API
#   - SharePoint documents
#   - Database queries (SELECT policy_text FROM policies WHERE status='active')
#
# Each document has:
#   - id: unique identifier (used as prefix for chunk IDs)
#   - title: human-readable name (prepended to each chunk for context)
#   - text: the actual policy content
#   - metadata: key-value labels used for filtering (department, version, is_active)
#
# IMPORTANT: is_active is your primary guardrail against outdated information.
# Archived documents (is_active=False) are excluded from search by default.
# This prevents old policies from contradicting current ones.

raw_documents = [
    {
        "id":    "hr_leave_v2",
        "title": "Leave Policy",
        "text":  """
Full-time employees receive 24 days of paid annual leave per year.
Leave must be requested at least 3 days in advance.
Emergency medical leave is exempt from this advance notice rule.
Unused leave carries forward up to 12 days maximum.
Leave beyond 12 days is forfeited at year end with no encashment.

Probationary Period: Employees in their first 90 days receive
12 pro-rated leave days and cannot carry any forward.

Sick Leave: 12 days per year, separate from annual leave.
A medical certificate is required for absences over 2 consecutive days.
Sick leave cannot be carried forward or encashed.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "Leave Policy",
            "is_active":      True,
            "last_updated":   "2024-03-01",
            "audience":       "all_employees",
            "version":        "2.1"
        }
    },
    {
        "id":    "hr_wfh_v2",
        "title": "Work From Home Policy",
        "text":  """
Eligible employees may work from home up to 2 days per week.
Manager approval is required and must be requested by Thursday
for the following week.

Core hours apply: employees must be reachable from 10am to 5pm IST.

Restrictions: WFH is not permitted in the first 90 days.
Employees on a performance improvement plan need explicit HR approval.

Equipment: A one-time Rs 5000 home office allowance is available,
claimed through the standard expense process.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "Work From Home Policy",
            "is_active":      True,
            "last_updated":   "2024-01-15",
            "audience":       "all_employees",
            "version":        "1.3"
        }
    },
    {
        "id":    "finance_expense_v2",
        "title": "Expense and Reimbursement Policy",
        "text":  """
Travel expenses are reimbursed within 7 working days of receipt submission.
All expenses above Rs 500 require a physical or digital receipt.
Cab expenses for client visits are fully reimbursed without a cap.
Personal travel is not reimbursable, even when combined with business travel.

Meal allowance during client visits: up to Rs 1500 per meal.
International travel requires pre-approval from the Finance team
and the employee's department head.

Advance requests for travel above Rs 10000 are available and
must be settled within 5 working days of return.
        """.strip(),
        "metadata": {
            "department":     "finance",
            "document_type":  "policy",
            "document_title": "Expense and Reimbursement Policy",
            "is_active":      True,
            "last_updated":   "2024-02-10",
            "audience":       "all_employees",
            "version":        "3.0"
        }
    },
    {
        "id":    "hr_leave_old",
        "title": "Leave Policy (2022 — Archived)",
        "text":  """
Full-time employees receive 18 days of paid annual leave per year.
Leave must be requested at least 5 days in advance.
No carry forward of unused leave is permitted.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "Leave Policy (Archived)",
            "is_active":      False,   # ← archived: excluded from search by default
            "last_updated":   "2022-01-01",
            "audience":       "all_employees",
            "version":        "1.0"
        }
    },
]


# ── CHUNKING ──────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Why We Chunk Documents
# LLMs charge per token (roughly per word). A full HR document could be
# 10,000 tokens. Sending it on every query = very expensive.
# Solution: split into small focused pieces, only send the relevant ones.
#
# Overlap: we repeat the last N characters of chunk 1 at the start of chunk 2.
# This prevents answers from being cut off at chunk boundaries.
# Think of it like a Venn diagram — adjacent chunks share some content.
#
# Industry term: "sliding window chunking" when overlap is used.

def recursive_chunk(
    text:       str,
    chunk_size: int = 400,
    overlap:    int = 50
) -> list[str]:
    """
    Split text into chunks, trying to break at natural language boundaries.

    PYTHON CONCEPT: Nested functions
    The inner 'split' function is defined INSIDE recursive_chunk.
    It can only be called from within recursive_chunk.
    We use a nested function here because 'split' is a helper that
    doesn't need to be visible outside this function.

    Args:
        text       : full document text to chunk
        chunk_size : max characters per chunk (default 400)
        overlap    : characters repeated between adjacent chunks (default 50)

    Returns:
        list[str] : list of text chunks
    """
    # Try splitting on these separators in order — coarsest to finest
    # We prefer paragraph breaks over line breaks over sentence breaks etc.
    separators = ["\n\n", "\n", ". ", " "]

    def split(text: str, seps: list[str]) -> list[str]:
        """Inner recursive function that tries each separator in order."""
        if not seps:
            return [text]  # no more separators to try — return as-is

        sep    = seps[0]              # try this separator first
        parts  = text.split(sep) if sep else list(text)
        chunks = []
        current = ""

        for part in parts:
            piece = part + sep if sep else part

            if len(current) + len(piece) <= chunk_size:
                current += piece      # still fits — keep building current chunk
            else:
                if current:
                    chunks.append(current.strip())   # save the full chunk

                # If this single piece is STILL too big, go deeper
                if len(piece) > chunk_size and len(seps) > 1:
                    # Recurse with the next finer separator
                    chunks.extend(split(piece, seps[1:]))
                    current = ""
                else:
                    current = piece   # start a new chunk

        if current.strip():
            chunks.append(current.strip())   # don't forget the last chunk

        return chunks

    raw = split(text, separators)

    # Add overlap between adjacent chunks
    if overlap == 0 or len(raw) <= 1:
        return raw

    overlapped = [raw[0]]  # first chunk is unchanged
    for i in range(1, len(raw)):
        # raw[i-1][-overlap:] = last 'overlap' characters of previous chunk
        tail = raw[i - 1][-overlap:]
        overlapped.append(tail + raw[i])  # prepend tail to current chunk

    return overlapped


# ── INDEXING ──────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: The Indexing Pipeline (Phase 1 of RAG)
# This runs ONCE when you set up the system (or when documents change).
# Steps:
#   1. Chunk each document into smaller pieces
#   2. Embed all chunks using the embedding model
#   3. Store chunks + embeddings + metadata in ChromaDB
#
# After this runs, your knowledge base is ready for queries.
# ChromaDB persists everything to disk — data survives between Python runs.
# Industry term: "ingestion pipeline" for this document → vector DB process.

def index_documents(docs: list[dict]) -> int:
    """
    Chunk, embed, and store all documents in ChromaDB with metadata.

    Returns: total number of chunks created (useful for verification)
    """
    # PYTHON CONCEPT: Type annotated lists
    # list[str] and list[dict] are just hints — Python doesn't enforce them
    # They help you and your IDE understand what these lists contain
    all_ids       : list[str]  = []
    all_texts     : list[str]  = []
    all_metadatas : list[dict] = []

    logger.info(f"Starting indexing for {len(docs)} documents")

    for doc in docs:
        chunks = recursive_chunk(doc["text"], chunk_size=400, overlap=50)
        logger.info(f"  '{doc['title']}' → {len(chunks)} chunk(s)")

        for i, chunk_text in enumerate(chunks):
            # e.g. "hr_leave_v2_chunk_0", "hr_leave_v2_chunk_1"
            chunk_id = f"{doc['id']}_chunk_{i}"

            # PYTHON CONCEPT: Dictionary unpacking with **
            # {**doc["metadata"], "chunk_index": i} creates a NEW dict
            # containing EVERYTHING from doc["metadata"] PLUS chunk_index and chunk_total
            # The ** operator "spreads" the dict's key-value pairs into the new dict
            chunk_metadata = {
                **doc["metadata"],         # inherits all document-level metadata
                "chunk_index": i,          # which chunk within this document (0-based)
                "chunk_total": len(chunks),# total chunks in this document
                "section":     doc["title"],
            }

            all_ids.append(chunk_id)
            # Prepend title so every chunk carries its document identity
            # Even if metadata is stripped, the chunk still says what doc it's from
            all_texts.append(f"{doc['title']}\n\n{chunk_text}")
            all_metadatas.append(chunk_metadata)

    # Embed all chunks in ONE batch call — much faster than encoding one by one
    # Industry term: "batch inference" — processing multiple inputs together
    logger.info(f"Embedding {len(all_texts)} chunks in batch...")
    embeddings = embedding_model.encode(all_texts).tolist()
    # .tolist() converts numpy array → Python list (ChromaDB requires Python lists)

    # upsert = "update if exists, insert if new"
    # Safe to call multiple times — won't create duplicates
    collection.upsert(
        ids        = all_ids,
        documents  = all_texts,
        embeddings = embeddings,
        metadatas  = all_metadatas
    )

    total = len(all_ids)
    logger.info(f"✅ Indexed {total} total chunks")
    return total


def ensure_indexed() -> None:
    """
    Check if the collection has data. If empty, index all documents.

    DOMAIN KNOWLEDGE: Idempotent Operations
    An "idempotent" operation produces the same result whether you run it
    once or ten times. This function is idempotent — it only indexes if needed.
    Industry pattern: "lazy initialization" — don't do work until it's needed.

    This function is the key fix for the eval bug:
    eval_test_set.py imports search() and ask() from this module.
    Without calling index_documents(), the collection stays empty.
    ensure_indexed() is called at the top of eval_test_set.py to guarantee
    data exists before any search runs.
    """
    count = collection.count()  # how many chunks are currently stored?

    if count == 0:
        logger.warning(
            "Collection is empty — running index_documents() automatically."
        )
        index_documents(raw_documents)
    else:
        logger.info(f"Collection already has {count} chunks — skipping indexing.")


# ── SEARCH ────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Semantic Search with Metadata Filtering
# This is Phase 2a of RAG — the "R" in RAG (Retrieval).
#
# What makes this "semantic" (vs keyword):
#   Keyword: finds exact word matches ("leave" finds "leave" not "vacation")
#   Semantic: finds meaning matches ("days off" finds "annual leave" policy)
#
# Metadata filtering is like SQL's WHERE clause — it scopes the search
# to only the documents you care about. Without it, every search scans
# ALL documents even when you only need HR docs.
#
# Industry term: "hybrid search" when you combine semantic + keyword + metadata.

def search(
    query:       str,
    top_k:       int         = 3,
    filters:     dict | None = None,
    active_only: bool        = True
) -> list[dict]:
    """
    Find the most semantically similar chunks for a query.

    PYTHON CONCEPT: dict | None type hint
    This means the parameter accepts either a dict OR None.
    None = "no value provided" (equivalent to NULL in SQL).
    We use it when filters are optional.

    Args:
        query       : the user's question in natural language
        top_k       : how many chunks to return (default 3)
        filters     : optional dict to narrow search, e.g. {"department": "hr"}
        active_only : if True, skip documents where is_active=False

    Returns:
        list of dicts, each containing: text, metadata, similarity score
    """
    # Guard: if collection is empty, warn and return nothing
    # This prevents confusing ChromaDB errors when no data exists
    if collection.count() == 0:
        logger.warning("Collection is empty. Call ensure_indexed() first.")
        return []

    # Step 1: Convert the question to an embedding (vector)
    # MUST use the same model that was used during indexing
    # Different models produce incompatible embeddings (different dimensions)
    query_embedding = embedding_model.encode(query).tolist()

    # Step 2: Build the metadata filter (ChromaDB's WHERE clause)
    # PYTHON CONCEPT: building a list of conditions dynamically
    # We start with an empty list and add conditions based on parameters
    where_conditions = []

    if active_only:
        # This is the most important guardrail — skip all archived documents
        # Without this, old incorrect policies could pollute your answers
        where_conditions.append({"is_active": {"$eq": True}})

    if filters:
        # PYTHON CONCEPT: dict.items() returns (key, value) pairs
        # e.g. {"department": "hr"}.items() = [("department", "hr")]
        for key, value in filters.items():
            if isinstance(value, list):
                # {"department": ["hr", "finance"]} → match any in list
                where_conditions.append({key: {"$in": value}})
            else:
                # {"department": "hr"} → exact match
                where_conditions.append({key: {"$eq": value}})

    # Combine all conditions into ChromaDB filter format
    if len(where_conditions) == 0:
        where = None                       # no filter — search everything
    elif len(where_conditions) == 1:
        where = where_conditions[0]        # single condition — use directly
    else:
        where = {"$and": where_conditions} # multiple conditions — AND logic

    logger.info(f"Searching | query='{query[:50]}' | filter={where}")

    # Step 3: Ask ChromaDB to find the most similar chunks
    # ChromaDB compares query_embedding against all stored embeddings
    # Returns the top_k closest ones
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        where            = where,
        include          = ["documents", "metadatas", "distances"]
        # "distances" = how far each chunk is in vector space from the query
    )

    # Step 4: Convert raw results to readable dicts
    # PYTHON CONCEPT: results["documents"][0]
    # ChromaDB supports multiple queries at once, hence the nested list.
    # [0] gets the results for our first (and only) query.
    chunks = []
    for i in range(len(results["documents"][0])):
        # Convert cosine distance → cosine similarity
        # distance=0.0 → similarity=1.0 (identical)
        # distance=1.0 → similarity=0.0 (unrelated)
        # This formula ONLY works correctly with cosine distance metric
        # (that's why hnsw:space="cosine" in the collection setup matters)
        similarity = round(1 - results["distances"][0][i], 3)

        chunks.append({
            "text":       results["documents"][0][i],
            "metadata":   results["metadatas"][0][i],
            "similarity": similarity
        })

    return chunks


# ── LLM CALLERS ───────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Provider Fallback Pattern
# No LLM provider has 100% uptime. Rate limits (too many requests per minute)
# are common on free tiers. We handle this with a "fallback chain":
#   1. Try Gemini (primary — cost-effective, fast)
#   2. If Gemini fails → try OpenAI (fallback)
#   3. If both fail → return a safe error message
#
# Industry terms:
#   "Provider redundancy" — using multiple LLM providers for reliability
#   "Circuit breaker"     — detecting failures and routing around them
#   "Graceful degradation" — failing safely instead of crashing

def call_gemini(prompt: str) -> str:
    """Call Gemini 2.5 Flash. Returns response as plain string."""
    response = gemini_client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = prompt,
        config   = types.GenerateContentConfig(
            response_mime_type = "text/plain"
        )
    )
    return response.text


def call_openai(prompt: str) -> str:
    """Call GPT-4.1-mini. Returns response as plain string."""
    response = openai_client.chat.completions.create(
        model       = "gpt-4.1-mini",  # ← confirmed available model
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.2  # lower = more focused, less creative. Good for factual Q&A.
    )
    return response.choices[0].message.content


def call_llm_with_fallback(prompt: str, delay: int = 13) -> tuple[str, str]:
    """
    Try Gemini. Fall back to OpenAI if Gemini fails.

    PYTHON CONCEPT: Returning multiple values
    Python functions can return multiple values as a tuple.
    The caller unpacks them: answer, provider = call_llm_with_fallback(prompt)

    Returns:
        tuple[str, str]: (answer_text, name_of_provider_that_was_used)
    """
    # Gemini free tier = 5 requests/minute
    # 13 second sleep = ~4.6 requests/minute (safely under limit)
    # Industry term: "throttling" — intentionally slowing down API calls
    if delay > 0:
        logger.info(f"Waiting {delay}s (rate limit buffer)...")
        time.sleep(delay)

    # Try Gemini first
    try:
        return call_gemini(prompt), "gemini"
    except ResourceExhausted:
        # ResourceExhausted = HTTP 429 (rate limit hit)
        logger.warning("Gemini rate limit — falling back to OpenAI")
    except Exception as e:
        logger.warning(f"Gemini error: {str(e)[:80]} — falling back to OpenAI")

    # Try OpenAI as fallback
    try:
        return call_openai(prompt), "openai"
    except Exception as e:
        logger.error(f"All providers failed: {str(e)[:100]}")
        return "[All LLM providers failed — check API keys and logs]", "none"


# ── ASK ───────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: The Full RAG Pipeline (Phase 2)
# This function is the "AG" in RAG — Augmented Generation.
# Steps:
#   1. Retrieve relevant chunks (using search() above)
#   2. Guard: if nothing relevant, return "I don't know" WITHOUT calling LLM
#   3. Augment: inject retrieved chunks into the prompt as context
#   4. Generate: call LLM — it MUST answer only from the context
#
# The "guard" step is critical for two reasons:
#   a) It saves API tokens (no LLM call needed)
#   b) It prevents hallucination from empty or irrelevant context
# Industry term: "no-retrieval guard" or "early exit pattern"

def ask(
    question:    str,
    filters:     dict | None = None,
    active_only: bool        = True,
    delay:       int         = 13
) -> str:
    """
    Answer a question using the RAG pipeline.

    Args:
        question    : the user's natural language question
        filters     : optional metadata filter (e.g. {"department": "finance"})
        active_only : skip archived documents (always True in production)
        delay       : seconds to wait before LLM call (for rate limiting)

    Returns:
        str : the answer, citing source documents
    """
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    if filters:
        print(f"   Filters : {filters}")
    print(f"{'='*60}")

    # ── STEP 1: RETRIEVE ──────────────────────────────────────────────────────
    chunks = search(
        query       = question,
        top_k       = 3,
        filters     = filters,
        active_only = active_only
    )

    print(f"\n📚 Retrieved {len(chunks)} chunk(s):")
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        print(
            f"   [{i}] {meta['document_title']} v{meta['version']} | "
            f"sim={chunk['similarity']} | "
            f"{chunk['text'][:65]}..."
        )

    # ── STEP 2: GUARD — EARLY EXIT IF NOTHING RETRIEVED ──────────────────────
    # PYTHON CONCEPT: falsy check on a list
    # "if not chunks:" is True when chunks is an empty list []
    # It's the pythonic way to check "is this list empty?"
    # Same as: if len(chunks) == 0:
    if not chunks:
        answer = "I don't have that information in the available documents."
        print(f"\n🤷 {answer}")
        print("─" * 60)
        return answer   # ← exit here, no LLM call needed

    # ── STEP 3: AUGMENT — BUILD CONTEXT-ENRICHED PROMPT ──────────────────────
    # We label each chunk with its source document so the LLM can cite it
    context_parts = []
    for chunk in chunks:
        source  = chunk["metadata"]["document_title"]
        version = chunk["metadata"]["version"]
        context_parts.append(f"[Source: {source} v{version}]\n{chunk['text']}")

    # Join all chunks with a separator so the LLM can distinguish between them
    context = "\n\n---\n\n".join(context_parts)

    # DOMAIN KNOWLEDGE: RAG Prompt Design
    # The three key instructions that prevent hallucination:
    #   "ONLY the information in the context" → anchors to retrieved docs
    #   "cite the source document"            → traceability / auditability
    #   "I don't have that information"       → defines exact fallback wording
    # These are called "grounding instructions" in the prompt.
    prompt = f"""
You are a company HR and Finance knowledge assistant.

Answer the question using ONLY the information in the CONTEXT below.
Always cite the source document (e.g. "According to the Leave Policy v2.1...").

If the answer is not in the CONTEXT, say exactly:
"I don't have that information in the available documents."

Do NOT use general knowledge. Do NOT make up information.
Only use what is explicitly written in the CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    # ── STEP 4: GENERATE ──────────────────────────────────────────────────────
    answer, provider = call_llm_with_fallback(prompt, delay=delay)

    print(f"\n💡 Answer [{provider}]:\n   {answer.strip()}")
    print("─" * 60)
    return answer


# ── MAIN ──────────────────────────────────────────────────────────────────────
# PYTHON CONCEPT: if __name__ == "__main__"
# ─────────────────────────────────────────────────────────────────────────────
# This is one of the most important Python patterns to understand.
#
# When you RUN this file directly:
#   python3 rag_with_metadata.py
#   → Python sets __name__ = "__main__"
#   → This block RUNS
#   → Documents get indexed, tests run
#
# When another file IMPORTS from this file:
#   from rag_with_metadata import search, ask
#   → Python sets __name__ = "rag_with_metadata" (the module name)
#   → This block does NOT run
#   → index_documents() is never called automatically
#   → The collection stays empty unless ensure_indexed() is called elsewhere
#
# This is WHY eval_test_set.py was getting 0 chunks — it imported search()
# without ever triggering indexing. The fix: call ensure_indexed() explicitly
# in eval_test_set.py before running any searches.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Index documents — run every time you change documents
    # Safe to re-run (upsert won't create duplicates)
    print("Indexing documents...")
    total = index_documents(raw_documents)
    print(f"✅ {total} chunks indexed\n")
    print("=" * 60)

    # Test 1: No filter — searches all active documents
    ask("How many days leave do I get?")

    # Test 2: Department filter — only searches HR documents
    ask(
        "Can I work from home in my first month?",
        filters={"department": "hr"}
    )

    # Test 3: Finance filter — correct document for this question
    ask(
        "What is the meal allowance for client visits?",
        filters={"department": "finance"}
    )

    # Test 4: active_only=True (default) — archived 2022 policy excluded
    ask(
        "How many days of annual leave do employees get?",
        active_only=True
    )

    # Test 5: active_only=False — archived 2022 policy may appear (dangerous!)
    ask(
        "How many days of annual leave do employees get?",
        active_only=False
    )