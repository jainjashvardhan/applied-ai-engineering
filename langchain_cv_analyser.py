# Block 4 — Building Your First Real Chain : Let's rebuild your CV analyser using LangChain. Compare it to your pure Python version:



# langchain_cv_analyser.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableWithFallbacks
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# ── MODELS ────────────────────────────────────────────────────────────────────
# LANGCHAIN FEATURE: .with_fallbacks()
# This is LangChain's built-in fallback handling.
# Your pure Python version needed manual try/except blocks.
# LangChain handles it with one method call.
#
# Industry term: "chain-level fallback" vs "manual retry logic"

gemini = ChatGoogleGenerativeAI(
    model          = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    temperature    = 0.1
)

openai_fallback = ChatOpenAI(
    model       = "gpt-4.1-mini",
    api_key     = os.getenv("OPENAI_API_KEY"),
    temperature = 0.1
)

# If Gemini fails for any reason, automatically try OpenAI
# This replaces your entire call_llm_with_fallback() function
model_with_fallback = openai_fallback.with_fallbacks([gemini])

# ── OUTPUT SCHEMA ─────────────────────────────────────────────────────────────
# LANGCHAIN FEATURE: Pydantic + JsonOutputParser
# Instead of writing json.loads() and hoping the model returns valid JSON,
# you define a Pydantic model and LangChain validates the structure for you.
# If the model returns invalid JSON, it raises a clear error.

class CVVerdict(BaseModel):
    """
    Pydantic model defining the expected output structure.
    PYTHON CONCEPT: Pydantic BaseModel for validation
    This is the same Pydantic you used in FastAPI for request validation.
    Here we use it to validate LLM output structure.
    """
    verdict:             str        # "recommend" | "consider" | "reject"
    confidence:          int        # 0-100
    strengths:           list[str]  # exactly 3
    concerns:            list[str]  # 1-3 items
    interview_questions: list[str]  # exactly 3

# JsonOutputParser uses the schema to validate and parse the LLM's JSON response
parser = JsonOutputParser(pydantic_object=CVVerdict)

# ── PROMPT WITH FORMAT INSTRUCTIONS ──────────────────────────────────────────
# LANGCHAIN FEATURE: parser.get_format_instructions()
# This automatically generates the JSON schema description for the prompt.
# You don't have to manually write the schema in your prompt anymore.

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Senior Talent Acquisition Lead with 10 years experience
        hiring for technical roles at Indian product companies.

        ROLE: {role}
        LEVEL: {experience_level}

        {format_instructions}

        Rules:
        - strengths: exactly 3 items
        - concerns: 1-3 items, empty list if none
        - interview_questions: exactly 3, specific to this candidate
        """
    ),
    ("human", "Analyse this CV:\n\n{cv_text}")
])

# ── BUILD THE CHAIN ───────────────────────────────────────────────────────────
# This is the complete CV analysis pipeline in one line.
# Read it left to right: fill prompt → call model → parse JSON

cv_chain = prompt | model_with_fallback | parser

# ── INVOKE ────────────────────────────────────────────────────────────────────
def analyse_cv(cv_text: str, role: str, experience_level: str = "mid") -> dict:
    """
    Analyse a CV using the LangChain chain.
    Returns a validated dict matching the CVVerdict schema.
    """
    result = cv_chain.invoke({
        "cv_text":            cv_text,
        "role":               role,
        "experience_level":   experience_level,
        "format_instructions": parser.get_format_instructions()
    })
    return result

# ── TEST IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = analyse_cv(
        cv_text = """
        Sarah Chen. Senior Data Engineer at Razorpay (3 years).
        Built Kafka pipeline processing 5M events/day.
        Python, Spark, Airflow, dbt. Led team of 3.
        IIT Bombay B.Tech 2019.
        """,
        role             = "Senior Data Engineer",
        experience_level = "senior"
    )

    print(f"Verdict    : {result['verdict']}")
    print(f"Confidence : {result['confidence']}")
    print(f"Strengths  : {result['strengths']}")
    print(f"Concerns   : {result['concerns']}")

# Compare this to your pure Python version — the chain definition is 1 line instead of 30. The trade-off is that it's harder to debug when something goes wrong.
