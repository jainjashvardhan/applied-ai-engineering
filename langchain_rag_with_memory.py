# Block 5 — RAG Chain with Memory (LangChain's Real Value) : This is where LangChain genuinely beats pure Python. Adding conversation memory to RAG properly is complex — LangChain makes it clean.

# =============================================================================
# langchain_rag_with_memory.py — Fixed version
# =============================================================================
#
# BUGS FIXED FROM PREVIOUS VERSION:
# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: 'dict' object has no attribute 'replace'
#   Cause : retriever received full input dict {"question": "..."} instead
#           of just the question string "..."
#   Fix   : use itemgetter("question") to extract string before retriever
#
# Fix 2: HuggingFaceEmbeddings deprecated in langchain_community
#   Fix   : from langchain_huggingface import HuggingFaceEmbeddings
#   Run   : pip install -U langchain-huggingface
#
# Fix 3: ChatMessageHistory deprecated in langchain_community
#   Fix   : from langchain_core.chat_history import InMemoryChatMessageHistory
#
# NOTE: RunnableWithMessageHistory is deprecated — LangChain recommends
# using LangGraph's built-in persistence instead. We keep it here to show
# the concept, but LangGraph (next lesson) is the production-grade approach.
# =============================================================================
#
# DOMAIN KNOWLEDGE: Why Does This Error Happen?
# ─────────────────────────────────────────────────────────────────────────────
# When you use a parallel dict {} in LCEL, ALL values in the dict receive
# the ENTIRE input, not individual fields.
#
# Input dict: {"question": "How many days of leave?", "chat_history": [...]}
#
# retriever receives: {"question": "How many days?", "chat_history": [...]}
# retriever expects : "How many days?"   ← just a string
#
# This is called "input routing" — you must explicitly extract the field
# you want before passing it to a component that expects a plain string.
#
# The fix: itemgetter("question") pulls out just the question string.
# itemgetter is from Python's built-in 'operator' module.
# itemgetter("question")({"question": "abc"}) → "abc"
# ─────────────────────────────────────────────────────────────────────────────

# ── IMPORTS ───────────────────────────────────────────────────────────────────
# PYTHON CONCEPT: operator.itemgetter
# itemgetter is a built-in Python utility that creates a function
# to extract a key from a dict (or index from a list).
# itemgetter("question") returns a function that, given a dict, returns dict["question"]
# It's more efficient than writing lambda x: x["question"]
from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

# ✅ Fixed: use langchain_core instead of deprecated langchain_community
from langchain_core.chat_history import InMemoryChatMessageHistory

# ✅ Fixed: use langchain_huggingface instead of deprecated langchain_community
# Run: pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.WARNING,   # WARNING suppresses routine INFO logs for cleaner output
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── EMBEDDING MODEL ───────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE: Why the same embedding model matters
# The embedding model MUST be the same one used when indexing documents.
# If you indexed with all-MiniLM-L6-v2, you MUST query with all-MiniLM-L6-v2.
# Different models produce different vector spaces — mixing them gives nonsense.
# Industry term: "embedding consistency" — a common production gotcha.

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2",
    # encode_kwargs controls how the model encodes text
    # normalize_embeddings=True ensures vectors are unit length
    # Required for cosine similarity to work correctly
    encode_kwargs = {"normalize_embeddings": True}
)
print("✅ Embedding model loaded\n")

# ── VECTOR STORE ──────────────────────────────────────────────────────────────
# LANGCHAIN CONCEPT: Chroma wrapper
# This is LangChain's interface to your existing ChromaDB collection.
# It wraps the same ./chroma_db_v2 folder you created in earlier sessions.
# You're reusing the indexed documents — no re-indexing needed.
#
# IMPORTANT: Point to the correct folder and collection name
# If you get 0 results, check these match your rag_with_metadata.py setup.

vectorstore = Chroma(
    collection_name    = "company_knowledge",   # must match your collection name
    embedding_function = embedding_model,
    persist_directory  = "./chroma_db_v2"       # must match your PersistentClient path
)

print(f"Vector store connected: {vectorstore._collection.count()} chunks available")

# ── RETRIEVER ─────────────────────────────────────────────────────────────────
# LANGCHAIN CONCEPT: Retriever
# .as_retriever() wraps the vectorstore as a LangChain retriever.
# A retriever's interface: given a string query → returns list[Document]
# Each Document has .page_content (the text) and .metadata (dict of labels).
#
# search_kwargs = parameters passed to the underlying similarity search
#   k      = how many chunks to return (same as your top_k)
#   filter = metadata filter (same as your active_only + filters logic)

retriever = vectorstore.as_retriever(
    search_type   = "similarity",
    search_kwargs = {
        "k":      3,
        "filter": {"is_active": True}   # only active documents
    }
)

# ── MODELS ────────────────────────────────────────────────────────────────────
# LANGCHAIN FEATURE: .with_fallbacks()
# Replaces your manual try/except fallback logic from pure Python.
# If Gemini throws any exception, LangChain automatically retries with OpenAI.
# Industry term: "automatic failover" — seamless provider switching on failure.

gemini = ChatGoogleGenerativeAI(
    model          = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    temperature    = 0.2
)

openai = ChatOpenAI(
    model       = "gpt-4.1-mini",
    api_key     = os.getenv("OPENAI_API_KEY"),
    temperature = 0.2
)

model = gemini.with_fallbacks([openai])

# ── PROMPT WITH MEMORY PLACEHOLDER ───────────────────────────────────────────
# LANGCHAIN CONCEPT: MessagesPlaceholder
# This is a special slot in the prompt that gets filled with
# the entire conversation history automatically.
#
# Without MessagesPlaceholder + memory:
#   Every question is answered independently — no context from previous turns
#   User: "What about sick leave?" → model has no idea what "that" refers to
#
# With MessagesPlaceholder + memory:
#   Previous Q&A pairs are injected into every new prompt
#   User: "What about sick leave?" → model knows the context from earlier turns
#
# variable_name must EXACTLY match what RunnableWithMessageHistory expects
# We use "chat_history" — the standard convention

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a company HR and Finance knowledge assistant.
        Answer questions using ONLY the information in the CONTEXT below.
        Always cite which policy document your answer comes from.

        If the answer is not in the CONTEXT, say exactly:
        "I don't have that information in the available documents."

        Do NOT use general knowledge. Only use the CONTEXT.

        CONTEXT FROM DOCUMENTS:
        {context}
        """
    ),
    # LANGCHAIN CONCEPT: MessagesPlaceholder
    # This slot gets replaced with the full conversation history at runtime.
    # First turn: empty list (no history yet)
    # Second turn: [HumanMessage("first Q"), AIMessage("first A")]
    # Third turn:  [HumanMessage("first Q"), AIMessage("first A"),
    #               HumanMessage("second Q"), AIMessage("second A")]
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# ── HELPER FUNCTION ───────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    """
    Format a list of LangChain Document objects into a context string.

    PYTHON CONCEPT: list check with 'not'
    'if not docs' is True when docs is an empty list []
    Same as: if len(docs) == 0

    Each Document object has:
      .page_content = the text of the chunk
      .metadata     = dict with department, version, is_active, etc.
    """
    if not docs:
        return "No relevant documents found in the knowledge base."

    return "\n\n---\n\n".join(
        # Get source from metadata, fallback to "Unknown" if not present
        f"[Source: {doc.metadata.get('document_title', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )

# ── BUILD THE RAG CHAIN ───────────────────────────────────────────────────────
# LANGCHAIN CONCEPT: RunnablePassthrough.assign()
# ─────────────────────────────────────────────────────────────────────────────
# THE FIX: We use RunnablePassthrough.assign() instead of a parallel dict {}.
#
# How RunnablePassthrough.assign() works:
#   Input  : {"question": "...", "chat_history": [...]}
#   Output : {"question": "...", "chat_history": [...], "context": "<formatted docs>"}
#   It passes ALL existing keys through AND adds the new "context" key.
#
# Why this is better than the broken parallel dict approach:
#   The parallel dict {} passes the ENTIRE input dict to every value.
#   retriever then receives a dict instead of a string → crash.
#
#   RunnablePassthrough.assign(context=...) calls our function with the input dict.
#   Our function extracts input["question"] and passes the STRING to retriever.
#   retriever gets a string → works correctly.
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_and_format(inputs: dict) -> str:
    """
    Extract question from inputs dict, retrieve relevant docs, format them.

    PYTHON CONCEPT: Function that receives a dict
    LCEL passes the full input dict to this function.
    We manually extract what we need (the question string).

    This is the key fix: we control exactly what goes to the retriever.
    """
    question = inputs["question"]           # extract just the question string
    docs     = retriever.invoke(question)   # retriever now gets a string ✅
    return format_docs(docs)

rag_chain = (
    # Step 1: Pass everything through AND add a "context" key
    # RunnablePassthrough.assign() = "keep all inputs + add this new key"
    # retrieve_and_format receives the full inputs dict, extracts question,
    # retrieves docs, returns formatted context string
    RunnablePassthrough.assign(
        context = RunnableLambda(retrieve_and_format)
    )
    # After this step, the dict now has: question, chat_history, context

    # Step 2: Fill the prompt template with all three values
    | rag_prompt

    # Step 3: Call the LLM (Gemini with OpenAI fallback)
    | model

    # Step 4: Extract plain text string from AIMessage response object
    | StrOutputParser()
)

# ── CONVERSATION MEMORY ───────────────────────────────────────────────────────
# LANGCHAIN CONCEPT: RunnableWithMessageHistory
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: This class is deprecated in favour of LangGraph's persistence.
# We're using it here to understand the CONCEPT of conversation memory.
# In the next lesson (LangGraph), we'll use the production-grade approach.
#
# What RunnableWithMessageHistory does automatically:
#   Before each invoke():
#     1. Loads chat history for the given session_id
#     2. Injects it into the chain inputs as "chat_history"
#   After each invoke():
#     3. Saves the new (question, answer) pair to the history
#
# session_id = unique ID per conversation
#   Same session_id = continues the same conversation (memory persists)
#   Different session_id = fresh conversation (separate history)
#
# In production: histories stored in Redis or PostgreSQL
# Here: in-memory dict (lost when script restarts)
# ─────────────────────────────────────────────────────────────────────────────

# PYTHON CONCEPT: Dict as a key-value store
# This dict acts as our in-memory "database" of conversation histories.
# Key   = session_id (e.g. "hr_user_1")
# Value = InMemoryChatMessageHistory object containing the conversation
conversation_store: dict = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Get existing conversation history for session_id,
    or create a new empty one if this session hasn't been seen before.

    PYTHON CONCEPT: dict.setdefault()
    setdefault(key, default) is equivalent to:
        if key not in dict:
            dict[key] = default
        return dict[key]
    It's a one-liner for "get or create".
    """
    return conversation_store.setdefault(session_id, InMemoryChatMessageHistory())

# Wrap the RAG chain with automatic memory management
chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key   = "question",     # which key in input is the user's message
    history_messages_key = "chat_history", # which MessagesPlaceholder to fill
)

# ── ASK FUNCTION ──────────────────────────────────────────────────────────────

# Change 1 — wrap ask() with error handling
def ask(question: str, session_id: str = "default") -> str:
    try:
        response = chain_with_memory.invoke(
            {"question": question},
            config = {"configurable": {"session_id": session_id}}
        )
        return response

    except Exception as e:
        error_msg = str(e)

        # Network-level failures — not API errors
        if "No route to host" in error_msg or "ReadError" in error_msg:
            logger.error("Network unreachable — check internet connection")
            return "[Network error — check your internet connection and retry]"

        # Rate limits
        if "429" in error_msg or "quota" in error_msg.lower():
            logger.warning("Rate limit hit")
            return "[Rate limit reached — wait a moment and retry]"

        # Everything else
        logger.error(f"LLM call failed: {error_msg[:100]}")
        return "[LLM call failed — check logs]"
# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("CONVERSATION TEST — Testing memory across turns")
    print("=" * 60)
    print("Pay attention to turns 2, 3, 4 — they use pronouns")
    print("that only make sense with conversation history.\n")

    # ── Conversation 1: HR questions ──────────────────────────────────────────

    print("─" * 60)
    print("Session: hr_user_1  (Turn 1 of 5)")
    print("─" * 60)
    q1 = "How many days of annual leave do I get?"
    a1 = ask(q1, "hr_user_1")
    print(f"Q: {q1}")
    print(f"A: {a1}\n")

    print("─" * 60)
    print("Session: hr_user_1  (Turn 2 of 5)")
    print("Note: 'What about sick leave?' has no standalone context")
    print("Model needs memory to know we're discussing leave policies\n")
    print("─" * 60)
    q2 = "What about sick leave?"
    a2 = ask(q2, "hr_user_1")
    print(f"Q: {q2}")
    print(f"A: {a2}\n")

    print("─" * 60)
    print("Session: hr_user_1  (Turn 3 of 5)")
    print("Note: 'Can I carry any of that forward?' — 'that' refers to sick leave")
    print("─" * 60)
    q3 = "Can I carry any of that forward?"
    a3 = ask(q3, "hr_user_1")
    print(f"Q: {q3}")
    print(f"A: {a3}\n")

    print("─" * 60)
    print("Session: hr_user_1  (Turn 4 of 5)")
    print("Note: New topic — switching from leave to WFH")
    print("─" * 60)
    q4 = "What about working from home — how many days per week?"
    a4 = ask(q4, "hr_user_1")
    print(f"Q: {q4}")
    print(f"A: {a4}\n")

    print("─" * 60)
    print("Session: hr_user_1  (Turn 5 of 5)")
    print("Note: New topic — switching from leave to WFH")
    print("─" * 60)
    q5 = "Does that apply during probation?"
    a5 = ask(q5, "hr_user_1")
    print(f"Q: {q5}")
    print(f"A: {a5}\n")

    # ── Conversation 2: Different session — no memory of Conversation 1 ───────
    print("\n" + "=" * 60)
    print("NEW SESSION: finance_user_1  (completely separate history)")
    print("=" * 60)
    print("This session knows NOTHING about the leave policy discussion above.\n")

    q6 = "What expenses can I claim for client visits?"
    a6 = ask(q6, "finance_user_1")
    print(f"Q: {q6}")
    print(f"A: {a6}\n")

    q7 = "What is the receipt threshold for those expenses?"
    a7 = ask(q7, "finance_user_1")
    print(f"Q: {q7}")
    print(f"A: {a7}\n")

    # ── Show conversation history to make memory concrete ─────────────────────
    print("=" * 60)
    print("WHAT THE MEMORY STORE LOOKS LIKE FOR hr_user_1:")
    print("=" * 60)
    # Access the stored history directly to see what was saved
    hr_history = conversation_store.get("hr_user_1")
    if hr_history:
        for i, msg in enumerate(hr_history.messages, 1):
            # PYTHON CONCEPT: type(msg).__name__ gives the class name as a string
            # HumanMessage → "HumanMessage", AIMessage → "AIMessage"
            role    = "User " if "Human" in type(msg).__name__ else "Model"
            content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            print(f"  [{i}] {role}: {content}")

    print("\n" + "=" * 60)
    print("NOTE: RunnableWithMessageHistory is deprecated.")
    print("LangChain recommends using LangGraph's built-in persistence.")
    print("This is exactly what we'll build in the next lesson: LangGraph.")
    print("=" * 60)