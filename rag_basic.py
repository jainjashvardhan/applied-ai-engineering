# =============================================================================
# rag_basic.py — Production-close RAG System
# =============================================================================
#
# WHAT IS RAG? (Domain Knowledge)
# ─────────────────────────────────────────────────────────────────────────────
# RAG = Retrieval Augmented Generation
#
# Problem it solves: LLMs (like GPT, Gemini, Claude) are trained on public
# internet data. They know nothing about YOUR company's internal documents,
# policies, or proprietary knowledge.
#
# How RAG fixes this:
#   Step 1 — INDEX  : Store your documents as searchable "embeddings" (numbers)
#   Step 2 — RETRIEVE: When a user asks a question, find the most relevant docs
#   Step 3 — GENERATE: Pass those docs to the LLM, ask it to answer from them
#
# The LLM is now grounded in YOUR data, not hallucinating from general knowledge.
# "Grounding" is the industry term for this — keeping the LLM anchored to facts.
#
# ARCHITECTURE OVERVIEW:
#   User Question
#       ↓
#   Embedding Model (converts text → numbers)
#       ↓
#   ChromaDB (finds most similar stored chunks)
#       ↓
#   Prompt (question + retrieved context)
#       ↓
#   Gemini / OpenAI (generates the final answer)
#       ↓
#   Answer (grounded in your documents)
#
# INDUSTRY BUZZWORDS USED IN THIS FILE:
#   - Embeddings      : numerical representations of text meaning
#   - Vector DB       : database optimised for similarity search on embeddings
#   - Chunking        : splitting documents into smaller pieces
#   - Semantic search : searching by meaning, not exact keywords
#   - Grounding       : keeping LLM output anchored to real documents
#   - Hallucination   : when LLM makes up facts not in its context
#   - Data poisoning  : malicious/wrong content in the knowledge base
#   - Guardrails      : rules that prevent the AI from going off-rails
#   - Metadata        : data about data (e.g. who wrote it, is it active)
#   - Retrieval       : the act of finding relevant documents
#   - Upsert          : insert if new, update if already exists
# =============================================================================

# ── IMPORTS ───────────────────────────────────────────────────────────────────
# PYTHON CONCEPT: imports bring external libraries into your script
# Each library was installed via: pip install <library-name>

import chromadb                                    # local vector database
from sentence_transformers import SentenceTransformer, CrossEncoder  # free embedding model, Cross-encoder — reads question AND chunk together, Much more accurate than embedding similarity, But too slow to run on entire database — use after retrieval
from google import genai                           # Gemini AI client (new SDK)
from google.genai import types                     # Gemini config types
from google.api_core.exceptions import ResourceExhausted  # Gemini rate limit error
from openai import OpenAI                          # OpenAI client (GPT models)

import os        # reading environment variables (API keys)
import time      # adding delays between API calls
import logging   # structured logging (better than print statements)
from dotenv import load_dotenv  # loads .env file into environment

# ── ENVIRONMENT & LOGGING ─────────────────────────────────────────────────────
# SECURITY CONCEPT: Never hardcode API keys in code.
# Always load them from a .env file which is in your .gitignore.
# If you accidentally commit a key, revoke it IMMEDIATELY on the provider's dashboard.
# Industry term: "secret leakage" — one of the most common security mistakes.

load_dotenv()  # reads .env file, makes keys available via os.getenv()

# PYTHON CONCEPT: logging vs print()
# print() works but gives you no context (no timestamp, no severity level)
# logging gives you: timestamp | level (INFO/WARNING/ERROR) | message
# In production systems, logs are collected, searched, and alerted on.
# "Observability" is the industry term for understanding what your system is doing.
logging.basicConfig(
    level  = logging.INFO,                              # show INFO and above
    format = "%(asctime)s | %(levelname)s | %(message)s"  # timestamp | level | msg
)
logger = logging.getLogger(__name__)  # creates a logger named after this file

# ── CLIENTS ───────────────────────────────────────────────────────────────────
# PYTHON CONCEPT: We create clients once at the top of the file.
# Creating a client = opening a connection to the external service.
# Creating it once and reusing is much faster than creating per request.

# Embedding model — runs LOCALLY on your machine (no API key, no cost)
# all-MiniLM-L6-v2 = a small but capable model, ~90MB download on first run
# It converts any text into 384 numbers that represent the text's meaning
# Industry term: "bi-encoder" — encodes text independently into vectors
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Gemini client — Google's LLM (our default for generation)
# We use the new google.genai SDK (the old google.generativeai is deprecated)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# OpenAI client — GPT models (our fallback if Gemini fails)
# gpt-4.1-mini is the most cost-effective capable model for production use
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── CHROMADB ──────────────────────────────────────────────────────────────────
# ChromaDB is our vector database — it stores:
#   1. The original text of each chunk
#   2. The embedding (list of numbers) for each chunk
#   3. Metadata (department, is_active, version, etc.)
#
# PersistentClient = data is saved to disk in ./chroma_db folder
# It survives between runs — you only index documents once (or when they change)
# Industry alternatives: Pinecone, Weaviate, Qdrant, pgvector (PostgreSQL extension)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# IMPORTANT: hnsw:space = "cosine" is critical
# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB's default distance metric is L2 (Euclidean distance)
# L2 distance is NOT bounded — can be any positive number
# Our similarity formula (1 - distance) breaks when distance > 1 → gives negatives
#
# Cosine distance is bounded [0, 2]:
#   0.0 = identical meaning    → similarity = 1.0
#   1.0 = unrelated            → similarity = 0.0
#   2.0 = opposite meaning     → similarity = -1.0
#
# This is why we must specify cosine explicitly. Default L2 caused the
# negative similarity scores we debugged in the previous session.
# ─────────────────────────────────────────────────────────────────────────────
def get_or_recreate_collection(name: str) -> chromadb.Collection:
    """
    Get existing collection or create a new one with cosine distance.

    PYTHON CONCEPT: Functions with return types
    The '-> chromadb.Collection' tells you what type this function returns.
    This is called a "type hint" — it helps you and your IDE understand the code.

    Why this function exists: if you ran an older version of this code,
    your collection was created with L2 distance (wrong). This function
    detects that and recreates it correctly.
    """
    try:
        # Try to load the existing collection
        col = chroma_client.get_collection(name)

        # Check if the distance metric is correct
        # col.metadata is a dict — we check the hnsw:space key
        if col.metadata.get("hnsw:space") != "cosine":
            logger.warning(
                f"Collection '{name}' uses wrong distance metric "
                f"({col.metadata.get('hnsw:space', 'l2')}). "
                f"Deleting and recreating with cosine."
            )
            chroma_client.delete_collection(name)
            # Fall through to create_collection below
            raise ValueError("wrong metric — recreating")

        logger.info(f"Loaded existing collection '{name}' with cosine distance ✅")
        return col

    except Exception:
        # Collection doesn't exist OR had wrong metric — create fresh
        logger.info(f"Creating new collection '{name}' with cosine distance")
        return chroma_client.create_collection(
            name     = name,
            metadata = {
                "description": "Company HR and Finance policy documents",
                "hnsw:space":  "cosine"   # ← THE critical setting
            }
        )

collection = get_or_recreate_collection("hr_policies")

# ── DOCUMENTS ─────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Knowledge Base Design
# ─────────────────────────────────────────────────────────────────────────────
# In production, documents come from:
#   - SharePoint / Google Drive (company wikis)
#   - Confluence (engineering documentation)
#   - PDFs parsed with libraries like PyMuPDF or pdfplumber
#   - Database queries (e.g. "SELECT policy_text FROM policies WHERE active=true")
#
# For learning, we use hardcoded strings — same concept, simpler setup.
#
# IMPORTANT — DATA POISONING RISK:
# Notice policy_006 and policy_007 below — this is a real example of
# "data poisoning": incorrect or malicious content in the knowledge base.
#
# policy_006 (correct POSH policy) is marked is_active=False (archived)
# policy_007 (wrong/harmful content) is marked is_active=True (active)
#
# If someone queries about harassment, the WRONG policy gets retrieved.
# This is why production RAG systems need a content review step before indexing.
# Industry term: "human-in-the-loop" review for sensitive documents.
#
# We have CORRECTED the flags below (v1=True, v2=False) to demonstrate
# proper knowledge base hygiene.
# ─────────────────────────────────────────────────────────────────────────────

documents = [
    {
        "id":    "policy_001",
        "title": "Leave Policy",
        "text":  """
Leave Policy

All full-time employees are entitled to 24 days of paid annual leave per calendar year.
Leave must be applied for at least 3 days in advance except in medical emergencies.
Unused leave can be carried forward up to a maximum of 12 days.
Leave beyond 12 days is forfeited at year end with no encashment.

Probationary Period Leave: Employees in their first 90 days receive
12 days of pro-rated leave and cannot carry any forward.

Sick Leave: 12 days per year, separate from annual leave.
A medical certificate is required for absences over 2 consecutive days.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "Leave Policy",
            "is_active":      True,    # ← current, valid version
            "version":        "1.0"
        }
    },
    {
        "id":    "policy_002",
        "title": "Offer Withdrawal Policy",
        "text":  """
Offer Withdrawal Policy

Candidates who accept an offer and then withdraw before their joining date
are not eligible for any reimbursement of relocation advances.

However, if the company withdraws the offer after the candidate has resigned
from their previous employer, a compensation of one month's offered salary will be paid.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "Offer Withdrawal Policy",
            "is_active":      True,
            "version":        "1.0"
        }
    },
    {
        "id":    "policy_003",
        "title": "Work From Home Policy",
        "text":  """
Work From Home Policy

Employees may work from home up to 2 days per week with manager approval.
Requests must be submitted by Thursday for the following week.
Employees must be reachable on Slack and email during core hours (10am to 5pm IST).

Restrictions:
WFH is not permitted during the first 90 days of employment (probation period).
Employees on a performance improvement plan require explicit HR approval for WFH.

Equipment Allowance:
The company provides a one-time Rs 5000 home office equipment allowance,
claimed through the standard reimbursement process.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "Work From Home Policy",
            "is_active":      True,
            "version":        "1.0"
        }
    },
    {
        "id":    "policy_004",
        "title": "Reimbursement Policy",
        "text":  """
Reimbursement Policy

Travel expenses will be reimbursed within 7 working days of submitting receipts.
All expenses above Rs 500 must have a physical or digital receipt.
Cab expenses for client visits are fully reimbursed without a cap.
Meal allowance for client visits is Rs 1500 per meal.
Personal travel is not reimbursable even if combined with a business trip.
International travel requires pre-approval from Finance and the department head.
        """.strip(),
        "metadata": {
            "department":     "finance",
            "document_type":  "policy",
            "document_title": "Reimbursement Policy",
            "is_active":      True,
            "version":        "1.0"
        }
    },
    {
        "id":    "policy_005",
        "title": "Performance Review Policy",
        "text":  """
Performance Review Policy

Performance reviews are conducted twice a year — in April and October.
Ratings are on a 5-point scale:
  1 = Does not meet expectations
  2 = Partially meets expectations
  3 = Meets expectations
  4 = Exceeds expectations
  5 = Outstanding

Employees rated 4 or above are eligible for promotion consideration.
Employees rated below 2 for two consecutive cycles are placed on a
performance improvement plan (PIP).
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "Performance Review Policy",
            "is_active":      True,
            "version":        "1.0"
        }
    },
    {
        "id":    "policy_006",
        "title": "POSH Policy",
        "text":  """
Prevention of Sexual Harassment (POSH) Policy

Any form of sexual harassment — verbal, physical, visual, or digital —
is strictly prohibited and constitutes gross misconduct.

This applies to all employees regardless of gender, seniority, or employment type.
It covers conduct in the office, at client sites, during company events,
and through any digital medium including email, messaging apps, or social media.

When a complaint is reported:
  1. An Internal Complaints Committee (ICC) will be formed within 7 days
  2. Investigation will be completed within 90 days
  3. If found guilty, consequences include suspension or termination

All complaints are treated with strict confidentiality.
Retaliation against complainants is itself a terminable offence.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "POSH Policy",
            "is_active":      True,    # ✅ FIXED: correct policy is now active
            "version":        "2.0"    # ✅ newer version = the current one
        }
    },
    {
        "id":    "policy_007",
        "title": "POSH Policy",
        "text":  """
POSH Policy

Staring and stalking your colleagues espacially outside office might be rewarded with a promotion.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "POSH Policy",
            "is_active":      False,   # ✅ FIXED: wrong/old content is now archived
            "version":        "1.0"
        }
    },
    {
        "id":    "policy_008",
        "title": "Probation Policy",
        "text":  """
Probation Policy

All new employees serve a mandatory 90-day probationary period.
During probation:
  - Leave entitlement is 12 pro-rated days (cannot be carried forward)
  - Work From Home is NOT permitted
  - Performance is formally reviewed at Day 45 and Day 90

Probation may be extended by 30 days if performance targets are not met.
Employees must be notified in writing before probation extension.
Successful completion of probation is confirmed via a written notice from HR.
        """.strip(),
        "metadata": {
            "department":     "hr",
            "document_type":  "policy",
            "document_title": "Probation Policy",
            "is_active":      True,
            "version":        "1.0"
        }
    },
]

# ── CHUNKING ──────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Why Chunking Matters
# ─────────────────────────────────────────────────────────────────────────────
# You cannot feed an entire document to the LLM every time for two reasons:
#   1. COST: LLMs charge per token (roughly per word). 50 pages = very expensive.
#   2. QUALITY: LLMs lose focus with very long inputs.
#              Research calls this the "lost in the middle" problem —
#              information in the middle of a long context gets ignored.
#
# So we split documents into small, focused pieces called "chunks."
# Each chunk is separately embedded and stored.
# At query time, only the 2-3 most relevant chunks are retrieved.
#
# Chunk size trade-off:
#   Too small (<100 chars) → chunk lacks context, retrieval misses answers
#   Too large (>2000 chars) → defeats the purpose, LLM loses focus
#   Sweet spot: 300-800 characters for most document types
#
# Overlap: we repeat the last N characters of one chunk at the start of the next.
# This prevents answers from being cut off at chunk boundaries.
# ─────────────────────────────────────────────────────────────────────────────
def recursive_chunk(
    text:       str,
    chunk_size: int = 700,
    overlap:    int = 100
) -> list[str]:
    """
    Split text into chunks, respecting natural language boundaries.

    PYTHON CONCEPT: Default parameter values
    chunk_size=400 means if you call recursive_chunk("some text"),
    chunk_size automatically equals 400. You only need to pass it
    if you want a different value.

    Args:
        text       : the full document text to split
        chunk_size : max characters per chunk (default 400)
        overlap    : characters to repeat between chunks (default 50)

    Returns:
        list[str] : a list of text strings, one per chunk
    """
    # Try splitting on these separators in order — coarse to fine
    # We try paragraph breaks first, then line breaks, then sentences, etc.
    # This preserves meaning better than splitting every N characters blindly
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    def split_with_separator(text: str, separator: str) -> list[str]:
        # PYTHON CONCEPT: if/else as expression
        # "return list(text) if separator=="" else text.split(separator)"
        # list("abc") = ["a","b","c"] — splits into individual characters
        if separator == "":
            return list(text)
        return text.split(separator)

    def chunk_recursive(text: str, separators: list[str]) -> list[str]:
        # PYTHON CONCEPT: nested functions
        # This function is defined inside recursive_chunk.
        # It can only be called from within recursive_chunk.
        # We use it to recurse (call itself) with a smaller separator.

        chunks    = []         # list to collect finished chunks
        separator = separators[0]            # try the first separator
        remaining = separators[1:]           # save the rest for later
        splits    = split_with_separator(text, separator)
        current   = ""                       # chunk being built up

        for split in splits:
            # Re-attach the separator we split on (splitting removes it)
            piece = split + separator if separator != "" else split

            if len(current) + len(piece) <= chunk_size:
                # Piece fits in current chunk — keep building
                current += piece
            else:
                # Current chunk is full — save it and start a new one
                if current:
                    chunks.append(current.strip())

                # If this single piece is STILL too big, go deeper
                # Try splitting on the next (finer) separator
                if len(piece) > chunk_size and remaining:
                    chunks.extend(chunk_recursive(piece, remaining))
                else:
                    current = piece  # start a new chunk with this piece

        # Don't forget the last chunk that was being built
        if current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = chunk_recursive(text, separators)

    # ── ADD OVERLAP ───────────────────────────────────────────────────────────
    # Overlap example:
    #   Chunk 1: "...employees must request leave 3 days in advance"
    #   Chunk 2: "3 days in advance. Unused leave can be carried forward..."
    #             ↑ repeated from end of chunk 1
    #
    # Why: if the answer spans two chunks, overlap ensures the context
    # from the previous chunk is still visible in the next one.
    # Without overlap, answers at boundaries get cut off.
    # ─────────────────────────────────────────────────────────────────────────
    if overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks   # no overlap needed for single chunks

    # PYTHON CONCEPT: list indexing
    # raw_chunks[i - 1] = the previous chunk
    # [-overlap:]       = the last `overlap` characters of that chunk
    # We prepend (add to the front of) the current chunk with this tail
    overlapped = [raw_chunks[0]]  # first chunk stays as-is
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap:]      # last N chars of previous
        overlapped.append(prev_tail + raw_chunks[i])  # prepend to current
    return overlapped


# ── INDEXING ──────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: The Indexing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Indexing = the process of preparing documents for search.
# This is Phase 1 of RAG. It runs ONCE (or when documents change).
#
# The pipeline:
#   Document text
#       ↓ recursive_chunk()
#   List of smaller text chunks
#       ↓ embedding_model.encode()
#   List of embeddings (each chunk = 384 numbers)
#       ↓ collection.upsert()
#   Stored in ChromaDB (text + embedding + metadata)
#
# "Upsert" = INSERT if new, UPDATE if already exists.
# Safe to re-run — won't create duplicates.
# ─────────────────────────────────────────────────────────────────────────────
def index_documents(docs: list[dict]) -> int:
    """
    Chunk all documents, embed them, and store in ChromaDB with metadata.

    PYTHON CONCEPT: return type int
    This function returns how many chunks were created in total.
    Useful for verifying the indexing worked correctly.
    """
    # PYTHON CONCEPT: parallel lists
    # We build three lists together — one entry per chunk across all docs
    # all_ids[i], all_texts[i], all_metadatas[i] all describe the same chunk
    all_ids       : list[str]  = []
    all_texts     : list[str]  = []
    all_metadatas : list[dict] = []

    print(f"\nIndexing {len(docs)} documents...")

    for doc in docs:
        # Split this document into chunks
        chunks = recursive_chunk(doc["text"], chunk_size=400, overlap=50)

        logger.info(f"  '{doc['title']}' → {len(chunks)} chunk(s)")

        for i, chunk_text in enumerate(chunks):
            # Create a unique ID for each chunk
            # e.g. "policy_001_chunk_0", "policy_001_chunk_1", etc.
            chunk_id = f"{doc['id']}_chunk_{i}"

            # PYTHON CONCEPT: dictionary unpacking with **
            # {**doc["metadata"], "chunk_index": i} means:
            # "take everything from doc['metadata'] AND add chunk_index and chunk_total"
            # This creates a new dict combining both — original metadata is unchanged
            chunk_metadata = {
                **doc["metadata"],       # inherits: department, is_active, version, etc.
                "chunk_index": i,        # which chunk within this document (0-based)
                "chunk_total": len(chunks),  # total chunks in this document
                "section":     doc["title"],  # which document this chunk came from
            }

            all_ids.append(chunk_id)
            # Prepend the document title to every chunk
            # This means every chunk knows which document it came from,
            # even without looking at metadata
            all_texts.append(f"{doc['title']}\n\n{chunk_text}")
            all_metadatas.append(chunk_metadata)

    # Embed ALL chunks at once — much faster than calling encode() per chunk
    # embedding_model.encode() accepts a list and returns a numpy array
    # .tolist() converts numpy array → Python list (ChromaDB needs Python lists)
    # Industry term: "batch encoding" — processing multiple items in one call
    print(f"  Embedding {len(all_texts)} chunks (this may take a moment)...")
    embeddings = embedding_model.encode(all_texts).tolist()

    # Store everything in ChromaDB
    # upsert = "update or insert" — safe to run multiple times
    collection.upsert(
        ids        = all_ids,
        documents  = all_texts,
        embeddings = embeddings,
        metadatas  = all_metadatas
    )

    total = len(all_ids)
    logger.info(f"Indexed {total} total chunks across {len(docs)} documents")
    print(f"✅ Indexed {total} chunks\n")
    return total


# ── SEARCH ────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Semantic Search vs Keyword Search
# ─────────────────────────────────────────────────────────────────────────────
# Traditional search (Google pre-2019, CTRL+F) = keyword matching
# "leave policy" only finds documents containing those exact words
#
# Semantic search = meaning matching
# "how many days off do I get" finds the Leave Policy
# even though it contains none of those exact words
#
# How: both the query and documents are converted to embeddings (vectors).
# Similar meanings → similar vectors → small cosine distance → high similarity.
#
# Similarity score guide (after cosine distance fix):
#   > 0.7 = strong match (almost certainly relevant)
#   0.4-0.7 = moderate match (probably relevant)
#   < 0.4 = weak match (likely noise — filter out)
#   < 0.0 = opposite meaning (definitely filter out)
# ─────────────────────────────────────────────────────────────────────────────
def search(
    query:          str,
    top_k:          int         = 3,
    min_similarity: float       = 0.4,
    filters:        dict | None = None,
    active_only:    bool        = True
) -> list[dict]:
    """
    Find the most relevant document chunks for a given query.

    PYTHON CONCEPT: dict | None
    This means the parameter accepts either a dict OR None (nothing).
    None is Python's way of saying "no value provided" (like NULL in SQL).

    Args:
        query          : the user's question
        top_k          : max chunks to return (default 3)
        min_similarity : drop chunks below this score (default 0.4)
        filters        : optional dict to narrow search by metadata
                         e.g. {"department": "finance"}
        active_only    : if True, skip archived (is_active=False) documents

    Returns:
        list of dicts, each with: text, metadata, similarity score
    """
    # Step 1: Convert the question to an embedding
    # MUST use the same model used during indexing — otherwise scores are meaningless
    query_embedding = embedding_model.encode(query).tolist()

    # Step 2: Build the "where clause" (like SQL WHERE) for metadata filtering
    # DOMAIN KNOWLEDGE: metadata filtering is how you scope your search
    # Without it, you search ALL documents even when you only need Finance docs
    where_conditions = []

    if active_only:
        # Only retrieve documents marked as current/active
        # This is your guardrail against data poisoning from old/wrong documents
        where_conditions.append({"is_active": {"$eq": True}})

    if filters:
        # PYTHON CONCEPT: .items() on a dict returns (key, value) pairs
        # e.g. {"department": "hr"}.items() = [("department", "hr")]
        for key, value in filters.items():
            if isinstance(value, list):
                # e.g. {"department": ["hr", "finance"]} → match either
                where_conditions.append({key: {"$in": value}})
            else:
                # e.g. {"department": "hr"} → match exactly
                where_conditions.append({key: {"$eq": value}})

    # Combine conditions into ChromaDB's filter format
    # PYTHON CONCEPT: len() returns the number of items in a list
    if len(where_conditions) == 0:
        where = None                          # no filters — search everything
    elif len(where_conditions) == 1:
        where = where_conditions[0]           # single condition — use directly
    else:
        where = {"$and": where_conditions}    # multiple conditions — AND them

    logger.info(f"Searching | query='{query[:50]}' | filter={where}")

    # Step 3: Query ChromaDB — find top_k*2 candidates, then filter by threshold
    # We fetch more than needed because some will be dropped by min_similarity
    safe_n = min(top_k * 2, collection.count())  # can't request more than exists
    if safe_n == 0:
        logger.warning("Collection is empty — run index_documents() first")
        return []

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = safe_n,
        where            = where,
        include          = ["documents", "metadatas", "distances"]
        # "distances" = how far each chunk is from the query in vector space
        # We convert distance → similarity below
    )

    # Step 4: Convert distances to similarities, filter by threshold
    chunks = []
    for i in range(len(results["documents"][0])):
        # With cosine distance metric: similarity = 1 - distance
        # distance=0.0 → similarity=1.0 (identical)
        # distance=1.0 → similarity=0.0 (unrelated)
        # distance=2.0 → similarity=-1.0 (opposite meaning)
        similarity = round(1 - results["distances"][0][i], 3)

        if similarity < min_similarity:
            logger.info(f"  Dropped chunk (similarity {similarity} < {min_similarity})")
            continue  # PYTHON CONCEPT: 'continue' skips to the next loop iteration

        chunks.append({
            "text":       results["documents"][0][i],
            "metadata":   results["metadatas"][0][i],
            "similarity": similarity
        })

    # Sort by similarity descending — best matches first
    # PYTHON CONCEPT: sorted() with key= parameter
    # key=lambda x: x["similarity"] means "sort by the 'similarity' value in each dict"
    # reverse=True means highest first
    chunks = sorted(chunks, key=lambda x: x["similarity"], reverse=True)

    return chunks[:top_k]  # return only the top_k results







#Enhanced Search with reranking

def search_and_rerank(
    query: str,
    top_k: int = 3,
    retrieve_k: int = 10,
    min_score: float = 1.5,
    filters: dict | None = None,
    active_only: bool = True
) -> list[dict]:
    """
    Two-stage retrieval

    Stage 1:
        Embedding search retrieves candidate chunks

    Stage 2:
        Cross-encoder reranks candidates

    Returns both:
        embedding_similarity
        rerank_score
    """

    # ── BUILD METADATA FILTER ─────────────────────────────

    where_conditions = []

    if active_only:
        where_conditions.append({
            "is_active": {"$eq": True}
        })

    if filters:
        for key, value in filters.items():

            if isinstance(value, list):
                where_conditions.append({
                    key: {"$in": value}
                })
            else:
                where_conditions.append({
                    key: {"$eq": value}
                })

    if len(where_conditions) == 0:
        where = None

    elif len(where_conditions) == 1:
        where = where_conditions[0]

    else:
        where = {
            "$and": where_conditions
        }

    # ── EMBEDDING SEARCH ─────────────────────────────────

    query_embedding = embedding_model.encode(query).tolist()

    safe_n = min(retrieve_k, collection.count())

    if safe_n == 0:
        logger.warning("Collection is empty")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=safe_n,
        where=where,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    candidates = []

    for i in range(len(results["documents"][0])):

        distance = results["distances"][0][i]

        candidates.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],

            # cosine similarity
            "embedding_similarity": round(
                1 - distance,
                3
            )
        })

    if not candidates:
        return []

    # ── RERANK ───────────────────────────────────────────

    pairs = [
        [query, c["text"]]
        for c in candidates
    ]

    scores = reranker.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    # ── DEBUG OUTPUT ─────────────────────────────────────

    print("\n🔍 Retrieval Debug")

    for candidate in candidates:

        print(
            f"embed={candidate['embedding_similarity']:.3f} | "
            f"rerank={candidate['rerank_score']:.3f} | "
            f"{candidate['metadata']['document_title']}"
        )

    # ── SORT BY RERANK SCORE ─────────────────────────────

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # ── THRESHOLD FILTER ────────────────────────────────

    if min_score is not None:

        candidates = [
            c
            for c in candidates
            if c["rerank_score"] >= min_score
        ]

    return candidates[:top_k]

# ── LLM CALLERS ───────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Why Gemini Default + OpenAI Fallback?
# ─────────────────────────────────────────────────────────────────────────────
# Different providers have different:
#   - Rate limits (how many calls per minute)
#   - Costs (price per 1000 tokens)
#   - Strengths (Gemini is good at structured output, GPT-4 at reasoning)
#   - Uptime (no provider has 100% availability)
#
# Using multiple providers with fallback logic is called "provider redundancy"
# or "multi-LLM architecture". It ensures your product stays up even if one
# provider has an outage or rate-limits you.
#
# Industry pattern: some companies route different task types to different providers
# (e.g. simple tasks → cheaper model, complex reasoning → better model)
# This is called "model routing" or "LLM routing".
# ─────────────────────────────────────────────────────────────────────────────
def call_gemini(prompt: str) -> str:
    """
    Call Gemini 2.5 Flash — our default LLM.
    Returns the model's response as a plain string.
    """
    response = gemini_client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = prompt,
        config   = types.GenerateContentConfig(
            response_mime_type = "text/plain"  # we want a plain text answer
        )
    )
    return response.text   # extract just the text string from the response object


def call_openai(prompt: str) -> str:
    """
    Call GPT-4.1-mini — our fallback LLM.
    Returns the model's response as a plain string.

    IMPORTANT: gpt-5 does not exist (as of mid-2025).
    Use gpt-4.1-mini for cost-effective production use.
    """
    response = openai_client.chat.completions.create(
        model       = "gpt-5",   # ← correct model name
        messages    = [{"role": "user", "content": prompt}],
        # temperature = 0.2  # lower temperature = more focused, less creative
        # Temperature range: 0.0 (deterministic) to 2.0 (very random)
        # For factual Q&A, we want low temperature — no creativity needed
    )
    return response.choices[0].message.content


def call_llm_with_fallback(prompt: str, delay: int = 13) -> tuple[str, str]:
    """
    Try Gemini first. Automatically fall back to OpenAI if Gemini fails.

    PYTHON CONCEPT: tuple return
    This function returns TWO values at once: (answer_text, provider_used)
    The caller can unpack them: answer, provider = call_llm_with_fallback(prompt)

    delay: seconds to wait before calling (respects Gemini free tier: 5 RPM limit)
    Set delay=0 if you have a paid Gemini plan.

    Returns:
        tuple[str, str]: (answer_text, provider_that_was_used)
    """
    # Rate limiting buffer — free tier allows 5 requests/minute
    # 13 seconds between calls = ~4.6 requests/minute (safely under limit)
    # Industry term: "throttling" — intentionally slowing down to respect limits
    if delay > 0:
        logger.info(f"Waiting {delay}s (rate limit buffer)...")
        time.sleep(delay)

    # Try Gemini (primary provider)
    try:
        answer = call_gemini(prompt)
        return answer, "gemini"   # success — return immediately with provider name
    except ResourceExhausted:
        # ResourceExhausted = HTTP 429 (Too Many Requests) from Gemini
        logger.warning("Gemini rate limit hit — falling back to OpenAI")
    except Exception as e:
        # Any other Gemini error (network issue, API outage, etc.)
        logger.warning(f"Gemini failed: {str(e)[:80]} — falling back to OpenAI")

    # Try OpenAI (fallback provider)
    try:
        answer = call_openai(prompt)
        return answer, "openai"   # success with fallback provider
    except Exception as e:
        # Both providers failed — return a safe error message
        # NEVER let an LLM error crash the whole application
        logger.error(f"All LLM providers failed. Last error: {str(e)[:100]}")
        return "[All LLM providers failed — check logs and API keys]", "none"


# ── ASK ───────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: The Complete RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# This function brings everything together:
#
#   1. RETRIEVE  : search ChromaDB for relevant chunks
#   2. GUARD     : if nothing relevant found, return "I don't know" without LLM call
#   3. AUGMENT   : build a prompt that includes the retrieved context
#   4. GENERATE  : call LLM to answer based ONLY on that context
#   5. CITE      : include source document names in the answer
#
# The "GUARD" step is important — it saves tokens and prevents hallucination.
# If we called the LLM with empty context, it might make something up.
# Returning "I don't know" directly is cheaper and more honest.
#
# Industry term: "RAG pipeline" for this entire sequence.
# Industry term: "guardrails" for the rules that prevent bad outputs.
# ─────────────────────────────────────────────────────────────────────────────
def ask(
    question:       str,
    filters:        dict | None = None,
    # active_only:    bool        = True,
    # min_similarity: float       = 0.4,
    delay:          int         = 13
) -> str:
    """
    Answer a question using the RAG pipeline.

    Args:
        question       : the user's question
        filters        : optional metadata filter (e.g. {"department": "finance"})
        active_only    : skip archived documents (default True — recommended)
        min_similarity : minimum relevance threshold (default 0.4)
        delay          : seconds to wait before LLM call (for rate limiting)
    """
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    if filters:
        print(f"   Filters:  {filters}")
    print(f"{'='*60}")

    # ── STEP 1: RETRIEVE ──────────────────────────────────────────────────────
    # chunks = search(
    #     query          = question,
    #     top_k          = 3,
    #     min_similarity = min_similarity,
    #     filters        = filters,
    #     active_only    = active_only
    # )search_and_rerank

    chunks = search_and_rerank(
        query          = question,
        top_k          = 5,
        retrieve_k     = 10, 
        filters        = filters,
        min_score      = -15,
        active_only=True
    )


    print(f"\n📚 Retrieved {len(chunks)} relevant chunk(s):")
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        print(
            f"   [{i}] {meta['document_title']} v{meta['version']} | "
            f"embed={chunk['embedding_similarity']:.3f} | "
            f"rerank={chunk['rerank_score']:.3f} | "
            f"{chunk['text'][:100]}..."
        )

    # ── STEP 2: GUARD — NO RESULTS ────────────────────────────────────────────
    # If no relevant chunks found above the threshold:
    # → Return "I don't know" directly WITHOUT calling the LLM
    # → This saves API costs and prevents hallucination
    # → Industry term: "early exit" or "no-retrieval guard"
    if not chunks:
        answer = "I don't have that information in the available documents."
        print(f"\n🤷 {answer}")
        print("─" * 60)
        return answer

    # ── STEP 3: AUGMENT — BUILD CONTEXT-ENRICHED PROMPT ──────────────────────
    # Each retrieved chunk gets labelled with its source document
    # The LLM can then cite sources in its answer
    context_parts = []
    for chunk in chunks:
        source  = chunk["metadata"]["document_title"]
        version = chunk["metadata"]["version"]
        # Format: [Source: Leave Policy v1.0]\n<chunk text>
        context_parts.append(f"[Source: {source} v{version}]\n{chunk['text']}")

    # Join chunks with a separator so the LLM can distinguish between them
    context = "\n\n---\n\n".join(context_parts)

    # DOMAIN KNOWLEDGE: Prompt Engineering for RAG
    # ─────────────────────────────────────────────────────────────────────────
    # Notice the explicit instructions:
    #   "ONLY the information in the context" → prevents hallucination
    #   "cite the source document"            → improves traceability
    #   "I don't have that information"       → defines exact fallback wording
    # These constraints are called "guardrails" in the industry.
    # ─────────────────────────────────────────────────────────────────────────
    prompt = f"""
You are a company HR and Finance knowledge assistant.
Answer the question using ONLY the information provided in the CONTEXT below.
Always cite the source document name (e.g. "According to the Leave Policy...").

If the answer is not in the context, say exactly:
"I don't have that information in the available documents."

Do NOT use any general knowledge. Do NOT make up information.
Only answer from what is explicitly stated in the CONTEXT.

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
# This block only runs when you execute THIS file directly (python3 rag_basic.py)
# It does NOT run when another file imports functions from this file
# This is how Python prevents code from running unintentionally on import
if __name__ == "__main__":

    # STEP 1: Index documents (builds the knowledge base)
    # Safe to run every time — upsert won't create duplicates
    # In production: you'd only re-index when documents change
    # index_documents(documents)

    # print("\n" + "="*60)
    # print("TEST SET 1 — Questions answered in the documents")
    # print("="*60)

    # # These should all return correct, sourced answers
    # ask("How many days of annual leave do I get?")
    # ask("Can I work from home during my probation period?")
    # ask("What happens if the company cancels my offer after I resign?")
    # ask("What is the meal allowance for client visits?",
    #     filters={"department": "finance"})  # scoped to Finance docs only

    print("\n" + "="*60)
    print("TEST SET 2 — Sensitive question (tests POSH guardrail)")
    print("="*60)

    # This SHOULD now return the correct POSH policy (v2, is_active=True)
    # Previously: wrong policy (v1 archived, v2 harmful) would have been returned
    # Fixed by: correcting the is_active flags in the documents above
    # ask("What will happen if I harass or stare at a colleague repeatedly and stalk her around everywhere?"#, filters={"is_active": True}
        # )

    ask("What does my policy policy_007 say?", {"document_title" : "POSH Policy"})


    # print("\n" + "="*60)
    # print("TEST SET 3 — Questions NOT in the documents (tests hallucination guard)")
    # print("="*60)

    # # These should return "I don't have that information" — NOT made-up answers
    # ask("What is the salary structure?")
    # ask("What is the company's maternity leave duration?")
    # ask("My manager keeps asking for sexual favours, what can I do?")

    # print("\n" + "="*60)
    # print("TEST SET 4 — Ambigious and multi chunk")
    # print("="*60)

    # These should cross reference multiple chunks then return right answeres
    # ask("What are the diffrent ways and of getting a promotion here? Which is the fastest and which one do you recommend"
        # ,
        # filters={"is_active": True}
        # )
    # ask("I visted a client and want to reimburse my cab expense but I got a 2 in ratings. Can I still get the full reimbursement? How is the reimbursement affected by ratings")
    # ask("Before joining the company for my work from home steup I bought a chair but the company revoked my offer, will I still get the reimbursement?")