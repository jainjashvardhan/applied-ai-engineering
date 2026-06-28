# langchain_basics.py — run this file to follow along

# Block 1 — ChatModel : The LLM wrapper. Swap providers by changing one line.

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# ── CHATMODELS — wrapping LLM providers ───────────────────────────────────────
# These are LangChain's wrappers around each provider's SDK.
# Instead of: anthropic_client.messages.create(model=..., messages=[...])
# You write:  model.invoke(messages)
#
# The interface is IDENTICAL for all providers.
# Swap Claude for Gemini by just changing this one line.
# Industry term: "provider abstraction" — hiding provider-specific details

claude  = ChatAnthropic(
    model       = "claude-sonnet-4-20250514",
    api_key     = os.getenv("ANTHROPIC_API_KEY"),
    temperature = 0.2
)

gemini  = ChatGoogleGenerativeAI(
    model       = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    temperature = 0.2
)

openai  = ChatOpenAI(
    model       = "gpt-4.1-mini",
    api_key     = os.getenv("OPENAI_API_KEY"),
    temperature = 0.2
)

# Pick your default — we'll use Gemini throughout
model = openai

# ── INVOKING A MODEL ──────────────────────────────────────────────────────────
# PYTHON CONCEPT: List of message objects
# LangChain uses message objects instead of raw dicts.
# HumanMessage  = user's message   (role: "user")
# SystemMessage = system prompt    (role: "system")
# AIMessage     = model's response (role: "assistant")

messages = [
    SystemMessage(content="You are a stupid HR assistant. Answer in one sentence."),
    HumanMessage(content="How many days of leave do employees get?")
]

# .invoke() sends the messages and returns an AIMessage object
response = model.invoke(messages)

print(response.content)    # → the text of the model's reply
print(type(response))      # → <class 'langchain_core.messages.ai.AIMessage'>



# Block 2 — PromptTemplate : Replaces your build_prompt() functions with reusable templates.

from langchain_core.prompts import ChatPromptTemplate

# ── CHATPROMPTTEMPLATE ────────────────────────────────────────────────────────
# This is LangChain's version of your f-string prompt builders.
# The difference: variables are defined with {curly_braces}
# and filled in when you call .format_messages() or invoke the chain.
#
# PYTHON CONCEPT: Triple-quoted strings
# The """ ... """ syntax allows multi-line strings.
# It's the same as your f-strings but the variables aren't filled in yet.
# LangChain fills them in later when it has the actual values.

cv_analysis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Senior Talent Acquisition Lead with 10 years experience.
        Analyse the CV for the {role} position at {experience_level} level.
        Return a JSON verdict: recommend, consider, or reject."""
    ),
    (
        "human",
        "CV:\n{cv_text}"
    )
])

# ── FILLING IN VARIABLES ──────────────────────────────────────────────────────
# .format_messages() fills in the {variables} and returns a list of messages
# This is what gets sent to the model

filled_messages = cv_analysis_prompt.format_messages(
    role             = "Senior Data Engineer",
    experience_level = "senior",
    cv_text          = "John Doe. 5 years at Flipkart. Python, Spark, Airflow..."
)

print(filled_messages[0].content)  # → the filled system message
print(filled_messages[1].content)  # → the filled human message



# messages = [
#     SystemMessage(content=filled_messages[0].content),
#     HumanMessage(content=filled_messages[1].content)
# ]

# # .invoke() sends the messages and returns an AIMessage object
# response = model.invoke(messages)

# print(response.content)    # → the text of the model's reply
# print(type(response))      # → <class 'langchain_core.messages.ai.AIMessage'>




#Block 3 — LCEL and the Pipe Operator : This is the heart of LangChain. The | operator chains components together.

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# ── THE PIPE OPERATOR | ───────────────────────────────────────────────────────
# PYTHON CONCEPT: The | operator
# In Python, | is normally the bitwise OR operator.
# LangChain overrides it to mean "chain these components together".
# This is called "operator overloading" — redefining what an operator does.
#
# Reading a chain left to right:
#   prompt | model | parser
#   = "fill the prompt, then send to model, then parse the output"
#
# This is similar to Unix pipes:
#   cat file.txt | grep "error" | sort | uniq
# Each step passes its output to the next step as input.
#
# Industry term: "pipeline" or "chain" for this connected sequence.

# ── SIMPLE CHAIN: prompt → model → string output ──────────────────────────────

simple_chain = cv_analysis_prompt | model | StrOutputParser()
# StrOutputParser() extracts just the text content from the AIMessage object
# Without it, you'd get the full AIMessage object, not just the text

result = simple_chain.invoke({
    "role":             "Senior Data Engineer",
    "experience_level": "senior",
    "cv_text":          "Sarah. 5 yrs Razorpay. Python, Kafka, Spark."
})

print(result)  # → plain string response

# ── WHAT THE PIPE OPERATOR ACTUALLY DOES ─────────────────────────────────────
# These two are IDENTICAL:
#
# Using pipe (clean, readable):
#   chain = prompt | model | parser
#   result = chain.invoke(inputs)
#
# Without pipe (verbose, what LangChain does internally):
#   filled = prompt.invoke(inputs)
#   response = model.invoke(filled)
#   result = parser.invoke(response)
#
# The pipe version is just cleaner syntax for the same thing.

