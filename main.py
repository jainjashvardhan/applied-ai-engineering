# ─────────────────────────────────────────────────────
# CV Analyser API — v2.0.0
# Multi-provider: Claude, OpenAI, Gemini
# Auto-fallback to Gemini if primary provider fails
#
# Endpoints:
#   GET  /health
#   POST /analyse-cv
#   POST /compare-candidates
# ─────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator       # ← field_validator, not validator
from typing import Optional

import anthropic
from openai import OpenAI
from google import genai
from google.genai import types
import chromadb
from sentence_transformers import SentenceTransformer


import json
import os
import logging
import time
from dotenv import load_dotenv




# ─────────────────────────────────────────────────────
# ENV + LOGGING
# ─────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# CLIENTS
# Initialised once at startup — reused across all requests
# ─────────────────────────────────────────────────────

anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

gemini_client   = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



# ─────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client   = chromadb.PersistentClient(path="./chroma_db_v2")

collection = chroma_client.get_or_create_collection(
    name     = "company_knowledge",
    metadata = {"description": "Company policies and documents"}
)



# ─────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────

VALID_PROVIDERS: list[str] = [
    "openai",
    "gemini",
    "claude"
]

FALLBACK_ORDER: list[str] = [
    "openai",
    "gemini",
    "claude"
]

# ─────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────

app = FastAPI(
    title="CV Analyser API",
    description="AI-powered CV analysis with multi-provider support and auto-fallback",
    version="2.0.0"
)


# ─────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────

class CVRequest(BaseModel):
    cv_text:            str
    role:               str
    experience_level:   str           = "mid"
    additional_context: Optional[str] = None
    provider:           str           = "gemini"

    @field_validator("cv_text")
    @classmethod
    def cv_text_must_be_valid(cls, v: str) -> str:
        if len(v.strip()) < 50:
            raise ValueError("CV text too short — minimum 50 characters")
        if len(v) > 10000:
            raise ValueError("CV text too long — maximum 10,000 characters")
        return v.strip()

    @field_validator("experience_level")
    @classmethod
    def valid_experience_level(cls, v: str) -> str:
        allowed = ["junior", "mid", "senior", "lead"]
        if v.lower() not in allowed:
            raise ValueError(f"experience_level must be one of: {allowed}")
        return v.lower()

    @field_validator("provider")
    @classmethod
    def valid_provider(cls, v: str) -> str:
        if v.lower() not in VALID_PROVIDERS:
            raise ValueError(f"provider must be one of: {VALID_PROVIDERS}")
        return v.lower()


class CVResponse(BaseModel):
    verdict:             str
    confidence:          int
    strengths:           list[str]
    concerns:            list[str]
    interview_questions: list[str]
    processing_time_ms:  Optional[int] = None
    provider_used:       Optional[str] = None
    fallback_used:       bool          = False


# Reused inside CompareResponse — one definition, two uses
class CandidateVerdict(BaseModel):
    verdict:             str
    confidence:          int
    strengths:           list[str]
    concerns:            list[str]
    interview_questions: list[str]


class CompareRequest(BaseModel):
    cv_one:           str
    cv_two:           str
    role:             str
    experience_level: str = "mid"
    provider:         str = "gemini"

    @field_validator("cv_one", "cv_two")       # one validator covers both fields
    @classmethod
    def cv_must_be_valid(cls, v: str) -> str:
        if len(v.strip()) < 50:
            raise ValueError("CV text too short — minimum 50 characters")
        if len(v) > 10000:
            raise ValueError("CV text too long — maximum 10,000 characters")
        return v.strip()

    @field_validator("experience_level")
    @classmethod
    def valid_experience_level(cls, v: str) -> str:
        allowed = ["junior", "mid", "senior", "lead"]
        if v.lower() not in allowed:
            raise ValueError(f"experience_level must be one of: {allowed}")
        return v.lower()

    @field_validator("provider")
    @classmethod
    def valid_provider(cls, v: str) -> str:
        if v.lower() not in VALID_PROVIDERS:
            raise ValueError(f"provider must be one of: {VALID_PROVIDERS}")
        return v.lower()


class CompareResponse(BaseModel):
    winner:                       str
    winner_rationale:             str
    key_differentiator:           str
    recommended_interview_order:  str
    candidate_one:                CandidateVerdict
    candidate_two:                CandidateVerdict
    processing_time_ms:           Optional[int] = None
    provider_used:                Optional[str] = None
    fallback_used:                bool          = False



class KnowledgeRequest(BaseModel):
    question:    str
    department:  Optional[str]  = None
    active_only: bool           = True
    provider:    str            = "gemini"

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Question too short")
        return v.strip()


class KnowledgeResponse(BaseModel):
    answer:          str
    sources_used:    list[str]   # document titles cited
    chunks_retrieved: int
    filters_applied: dict | None
    provider_used:   Optional[str] = None

# ─────────────────────────────────────────────────────
# PROMPT BUILDERS
# Kept separate from endpoints — easy to test and iterate
# ─────────────────────────────────────────────────────

def build_cv_prompt(request: CVRequest) -> str:
    context_line = ""
    if request.additional_context:
        context_line = f"\nAdditional context: {request.additional_context}"

    # Note: {{ and }} in f-strings produce literal { and }
    # {request.role} still interpolates normally
    return f"""
You are a Senior Talent Acquisition Lead with 10 years of experience
hiring for technical roles at Indian product companies (Dream11, Razorpay, Zerodha).

DOMAIN: B2C product company, Series A.
AUDIENCE: Hiring manager making a shortlist decision today.
GOAL: Decide whether to move this candidate to interview.
CONSTRAINTS: Return ONLY valid JSON. Never assume anything not in the CV.

FEW-SHOT EXAMPLES:

[EXAMPLE — Strong candidate]
CV signals: 4 yrs relevant exp, domain match, measurable impact
("reduced query time by 60%"), no employment gaps
→ verdict: "recommend", confidence: 88

[EXAMPLE — Overqualified edge case]
CV signals: 12 yrs exp for a mid-level role, last 3 roles were senior,
no reason given for stepping down
→ verdict: "consider", confidence: 52,
  concern: "Overqualified — likely to leave within 6 months"

ROLE BEING EVALUATED: {request.role}
EXPECTED LEVEL: {request.experience_level}{context_line}

Return ONLY valid JSON matching this exact schema.
No preamble. No markdown. No explanation. Raw JSON only.

{{
    "verdict": "recommend | consider | reject",
    "confidence": <integer 0-100>,
    "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "concerns": ["<concern 1>", "<concern 2>"],
    "interview_questions": ["<question 1>", "<question 2>", "<question 3>"]
}}

Rules:
- strengths: exactly 3 items
- concerns: 1-3 items. If none, return []
- interview_questions: exactly 3, specific to this candidate's CV
- confidence: how certain you are of the verdict, not candidate quality
- If input looks like a job description not a CV:
  verdict "reject", confidence 99, concerns: ["Input is a job description, not a CV"]

CV:
{request.cv_text}
""".strip()


def build_compare_prompt(request: CompareRequest) -> str:
    return f"""
You are a Senior Talent Acquisition Lead comparing two candidates for the same role.
Be decisive — hiring managers need a clear recommendation, not a balanced essay.

ROLE: {request.role}
EXPECTED LEVEL: {request.experience_level}

Return ONLY valid JSON matching this exact schema.
No preamble. No markdown. No explanation. Raw JSON only.

{{
    "winner": "candidate_one | candidate_two | tie",
    "winner_rationale": "<one decisive sentence explaining the winner>",
    "key_differentiator": "<the single factor that tipped the decision>",
    "recommended_interview_order": "candidate_one first | candidate_two first",
    "candidate_one": {{
        "verdict": "recommend | consider | reject",
        "confidence": <integer 0-100>,
        "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
        "concerns": ["<concern 1>"],
        "interview_questions": ["<question 1>", "<question 2>", "<question 3>"]
    }},
    "candidate_two": {{
        "verdict": "recommend | consider | reject",
        "confidence": <integer 0-100>,
        "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
        "concerns": ["<concern 1>"],
        "interview_questions": ["<question 1>", "<question 2>", "<question 3>"]
    }}
}}

Rules:
- strengths: exactly 3 items per candidate
- interview_questions: exactly 3 per candidate, tailored to that individual
- winner_rationale and key_differentiator must be different sentences
- Use "tie" only when candidates are genuinely indistinguishable — be decisive

CANDIDATE ONE CV:
{request.cv_one}

CANDIDATE TWO CV:
{request.cv_two}
""".strip()




# ── SEARCH WITH FILTERING ─────────────────────────────
def search(
    query:      str,
    top_k:      int           = 3,
    filters:    dict | None   = None,
    active_only: bool         = True    # default: skip archived docs
) -> list[dict]:
    """
    Search with optional metadata filters.

    Returns list of dicts with text AND metadata —
    not just text strings. Metadata is useful for
    citations and debugging.
    """
    query_embedding = embedding_model.encode(query).tolist()

    # Build the where clause
    where_conditions = []

    if active_only:
        where_conditions.append({"is_active": {"$eq": True}})

    if filters:
        for key, value in filters.items():
            if isinstance(value, list):
                where_conditions.append({key: {"$in": value}})
            else:
                where_conditions.append({key: {"$eq": value}})

    # Combine all conditions
    if len(where_conditions) == 0:
        where = None
    elif len(where_conditions) == 1:
        where = where_conditions[0]
    else:
        where = {"$and": where_conditions}

    logger.info(f"Searching with filter: {where}")

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        where            = where,
        include          = ["documents", "metadatas", "distances"]
    )

    # Package results with metadata
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":       results["documents"][0][i],
            "metadata":   results["metadatas"][0][i],
            "similarity": round(1 - results["distances"][0][i], 3)
        })

    return chunks








# ─────────────────────────────────────────────────────
# LLM CALLERS
# Each returns a plain str — the raw text from the model
# ─────────────────────────────────────────────────────

def call_claude(prompt: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text                    # ← returns str


def call_openai(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        # response_format={"type": "json_object"}        # forces JSON output
    )
    return response.choices[0].message.content         # ← returns str


def call_gemini(prompt: str) -> str:

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",

        contents=prompt,

        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    return response.text.strip()

# ─────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────

def generate_response(provider: str, prompt: str) -> str:
    """
    Route to the correct LLM caller.
    Provider is already validated by Pydantic before this runs.
    """
    if provider == "claude":
        return call_claude(prompt)
    elif provider == "openai":
        return call_openai(prompt)
    elif provider == "gemini":
        return call_gemini(prompt)

    # Safety net — should never reach here after Pydantic validation
    raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


# ─────────────────────────────────────────────────────
# FALLBACK LOGIC
# ─────────────────────────────────────────────────────

def generate_response_with_fallback(
    provider: str,
    prompt: str
) -> tuple[str, str]:

    providers_to_try = [provider]

    for fallback_provider in FALLBACK_ORDER:
        if fallback_provider not in providers_to_try:
            providers_to_try.append(fallback_provider)

    logger.info(
        f"Providers to try: {providers_to_try}"
    )

    errors = {}

    for current_provider in providers_to_try:

        try:

            if current_provider != provider:
                logger.warning(
                    f"Primary provider '{provider}' failed. "
                    f"Trying '{current_provider}'"
                )

            response = generate_response(
                current_provider,
                prompt
            )

            return response, current_provider

        except Exception as e:

            errors[current_provider] = str(e)

            logger.error(
                f"Provider '{current_provider}' failed: {str(e)}"
            )

    raise HTTPException(
    status_code=502,
    detail=errors
    )

# ─────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status":             "ok",
        "supported_providers": VALID_PROVIDERS,
        "fallback_provider":  FALLBACK_ORDER,
        "version":            "2.1.0"
    }

@app.get("/debug-openai")
def debug_openai():

    try:

        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Say hello"
                }
            ]
        )

        return {
            "success": True,
            "response": response.choices[0].message.content
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

@app.post("/analyse-cv")
async def analyse_cv(request: CVRequest) -> CVResponse:

    logger.info(
        f"analyse-cv | role={request.role} | "
        f"level={request.experience_level} | "
        f"provider={request.provider}"
    )

    start_time = time.time()
    prompt     = build_cv_prompt(request)

    # ── LLM CALL WITH FALLBACK ────────────────────────
    try:
        raw_text, provider_used = generate_response_with_fallback(
            request.provider, prompt
        )
    except HTTPException:
        raise                                          # pass through cleanly
    except Exception as e:
        logger.error(f"Unexpected error in analyse-cv: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # ── PARSE JSON ────────────────────────────────────
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error(f"JSON parse failed | raw: {raw_text[:300]}")
        raise HTTPException(
            status_code=500,
            detail="LLM returned malformed JSON — retry"
        )

    # ── BUILD RESPONSE ────────────────────────────────
    processing_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"analyse-cv done | verdict={result.get('verdict')} | "
        f"confidence={result.get('confidence')} | "
        f"provider={provider_used} | {processing_ms}ms"
    )

    result["processing_time_ms"] = processing_ms
    result["provider_used"]      = provider_used
    result["fallback_used"]      = (provider_used != request.provider)

    return CVResponse(**result)


@app.post("/compare-candidates")
async def compare_candidates(request: CompareRequest) -> CompareResponse:

    logger.info(
        f"compare-candidates | role={request.role} | "
        f"level={request.experience_level} | "
        f"provider={request.provider}"
    )

    start_time = time.time()
    prompt     = build_compare_prompt(request)

    # ── LLM CALL WITH FALLBACK ────────────────────────
    try:
        raw_text, provider_used = generate_response_with_fallback(
            request.provider, prompt
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in compare-candidates: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # ── PARSE JSON ────────────────────────────────────
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error(f"JSON parse failed | raw: {raw_text[:300]}")
        raise HTTPException(
            status_code=500,
            detail="LLM returned malformed JSON — retry"
        )

    # ── BUILD RESPONSE ────────────────────────────────
    processing_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"compare-candidates done | winner={result.get('winner')} | "
        f"provider={provider_used} | {processing_ms}ms"
    )

    result["processing_time_ms"] = processing_ms
    result["provider_used"]      = provider_used
    result["fallback_used"]      = (provider_used != request.provider)

    return CompareResponse(**result)


@app.post("/ask-knowledge-base")
async def ask_knowledge_base(request: KnowledgeRequest) -> KnowledgeResponse:

    filters = {}
    if request.department:
        filters["department"] = request.department

    # Get chunks with metadata
    chunks = search(
        query       = request.question,
        top_k       = 3,
        filters     = filters if filters else None,
        active_only = request.active_only
    )

    if not chunks:
        return KnowledgeResponse(
            answer           = "No relevant documents found for your question.",
            sources_used     = [],
            chunks_retrieved = 0,
            filters_applied  = filters or None
        )

    # Build context
    context_parts = [
        f"[Source: {c['metadata']['document_title']}]\n{c['text']}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a company knowledge assistant.
Answer using ONLY the context below. Cite the source document.
If not found, say: "I don't have that information."

CONTEXT:
{context}

QUESTION: {request.question}
""".strip()

    raw_text, provider_used = generate_response_with_fallback(
        request.provider, prompt
    )

    # Extract source titles for the response
    sources = list({
        c["metadata"]["document_title"]
        for c in chunks
    })

    return KnowledgeResponse(
        answer           = raw_text,
        sources_used     = sources,
        chunks_retrieved = len(chunks),
        filters_applied  = filters or None,
        provider_used    = provider_used
    )