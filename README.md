# Applied AI Engineering — Progress & Portfolio

**Owner:** Jash Vardhan Jain (JJ)
**Last updated:** 2026-06-29
**Target role:** Applied AI Engineer / LLM-Agent Engineer
**Repo:** `github.com/jainjashvardhan/applied-ai-engineering`

> This document is both a learning tracker and the backbone for the repo README.
> It replaces the older migration document, which predated Sessions 5A–7,
> the career plan, and the 22-week roadmap.

---

## 1. Who I Am (Starting Point)

Senior Data Analyst (~5 years) transitioning into production AI engineering.

**Genuine strengths (the moat):**
- Elite SQL (CTEs, window functions), BigQuery (expert), bash + bq CLI
- Real production data systems — architected gStore analytics powering 1,000+ retail stores
- Strong business framing: ROI, experimentation, edge-case thinking, stakeholder translation
- Already shipped 2 features to production (deploy intuition is real, not theoretical)

**The real gap (the interview gatekeeper):**
- Software engineering fundamentals — classes, modules, packaging, typing, testing, Git workflows
- Can read and tweak Python; not yet writing it fluently from scratch
- Can run/modify a Dockerfile; cannot yet author one from scratch

**The strategic read:** This is not "becoming an AI engineer from zero." It's a strong data
professional closing a *software engineering* gap while shipping a real agent. That profile —
Applied AI Engineer — is well-paid and under-supplied.

---

## 2. Role Target & Why

Targeting **Applied AI Engineer** and **LLM/Agent Engineer** roles. Deliberately *not* the
ML Engineer path (no deep PyTorch / distributed training / model compression) — that plays
against the analytics strength and would burn months on the wrong skills.

Benchmarked against real India job descriptions (June 2026). Consistent JD signals:
- Build agentic pipelines (LangGraph / multi-agent), RAG, prompt engineering, context/memory
- Strong backend + Python depth (the screen most candidates fail)
- Evaluation and guardrails on model outputs
- MCP (Model Context Protocol) — rising fast, high signal, few capable people
- A **deployed, public portfolio** — the single biggest hiring signal

Fine-tuning (LoRA/QLoRA): kept **conceptual-only** — know the *decision* (fine-tune vs RAG
vs prompt) cold; don't spend weeks on actual training runs.

---

## 3. Skills Status

### AI Engineering Skills

| Skill | Status | Evidence |
|---|---|---|
| Prompt engineering (6 techniques) | ✅ Strong | CV analyser prompts, structured CoT |
| FastAPI + Pydantic v2 | ✅ Moderate–Strong | CV analyser API, multi-endpoint |
| RAG (chunking, embeddings, metadata) | ✅ Strong | RAG system w/ metadata guardrails |
| Vector DBs (ChromaDB, cosine) | ✅ Strong | Production RAG, eval pipeline |
| LangChain (LCEL, fallbacks) | ✅ Strong | Multi-provider chains w/ memory |
| LangGraph (StateGraph → subgraphs) | ✅ Strong | Sessions 1–5A complete |
| Multi-agent (supervisor/workers) | ✅ Strong | Session 4 + 5A compiled subgraphs |
| HITL (interrupt/Command/resume) | ✅ Complete | Session 5 |
| LangSmith observability | ✅ Complete | Session 6 + challenge debrief |
| LangSmith evaluation | ✅ Complete | Session 7 + challenge debrief |
| AI security / guardrails | ⏳ Conceptual | Planned Wk 13 |
| Advanced retrieval (hybrid/re-rank) | ⏳ Pending | Planned Wk 16–17 if needed |
| Fine-tuning | 📖 Conceptual-only | By design |

### Software Engineering Fundamentals (#1 gap — tracked separately)

| Skill | Status | Notes |
|---|---|---|
| Git workflows | 🔄 In progress | Week 1 — first repo + commits |
| Project structure / modules | ⏳ Next | Week 2 |
| Functions done right | ⏳ Next | Week 2 |
| Classes & objects | ⏳ Pending | Week 3 |
| Type hints (rigorous) | ⏳ Pending | Week 4 |
| Testing (pytest) | ⏳ Pending | Week 5 |
| async / await | ⏳ Pending | Week 9 |
| Docker authoring (from scratch) | ⏳ Pending | Week 7 |
| GCP Cloud Run deploy | ⏳ Pending | Week 8 (GCP knowledge already real) |

---

## 4. Key Concepts Locked In

**Observability (Session 6):**
- Tracing comes from **LangChain wrappers** (`ChatOpenAI`, `ChatAnthropic`,
  `ChatGoogleGenerativeAI`), NOT from being inside LangGraph. A raw SDK call inside a
  node is invisible to LangSmith. *This is the single most important correction.*
- `@traceable` nests plain Python functions into the trace tree via `contextvars`.
- Enriched config (`run_name` / `metadata` / `tags`) makes traces searchable at scale.
- Feedback API turns passive tracing into an active quality signal.

**Evaluation (Session 7):**
- Dataset / evaluator / experiment are the three primitives.
- Rule-based evaluators for categorical outputs; LLM-as-judge for open-ended quality.
- **Statistical significance:** 6 examples is not enough for a deployment decision;
  aim for 50+ so one prediction flip moves the score <5%.
- **Dataset contamination:** examples used to *develop* a prompt must NOT be in the eval set.
  Keep a development pool and a locked, unseen eval pool separate.
- **Same-model judge** has two failure modes: self-serving bias AND cross-model comparison
  blindness. Use a stronger, independent model as judge.

**Architecture principles:**
- AI belongs where deterministic code fails. At 1.5M records / 5 min, comparison is a
  data-engineering problem; LLM value is reasoning about grouped discrepancies.
- "Thick tools" — encode deterministic domain logic in testable functions; let the LLM
  reason, not parse.
- Workflow (code owns the control flow) vs agent (LLM owns the control flow).

---

## 5. Projects (Portfolio)

| Project | What it demonstrates | Status |
|---|---|---|
| CV Analyser API | FastAPI, Pydantic v2, multi-provider LLM + fallback | ✅ Working |
| Production RAG (metadata guardrails) | Chunking, embeddings, cosine, active-only filtering | ✅ Working |
| RAG Evaluation | LLM-as-judge, faithfulness/relevance/precision | ✅ Working |
| LangChain RAG + memory | Multi-turn, session isolation, LCEL | ✅ Working |
| LangGraph curriculum (1–5A) | StateGraph → checkpointing → multi-agent → subgraphs | ✅ Complete |
| LangSmith observability (S6) | Auto-tracing, @traceable, feedback | ✅ Complete |
| LangSmith evaluation (S7) | Datasets, evaluators, experiment comparison | ✅ Complete |
| **gStore Replenishment QA Agent** | First real production agent (separate repo) | 🔜 On scope finalization |

---

## 6. The 22-Week Roadmap

**Principle:** the gStore QA agent is the spine. Build it deployed + public + tested,
and let it force the SWE gaps. One real shipped agent beats ten tutorial repos.

> Reframed from an initial "3-month" target. At ~10 hrs/week with a near-zero SWE base,
> 3 months wasn't realistic — and the switch decision is undecided, so the deadline was
> artificial. 22 weeks builds a genuinely stronger candidate, with an honest go/no-go at Wk 12.

### Phase 1 — Foundations + Agent Core (Wk 1–6)
- Wk 1: Git + project structure → repo live, README, first commits ← **CURRENT**
- Wk 2: Functions & modules → modular codebase
- Wk 3: Classes & objects → core objects modeled
- Wk 4: Type hints & defensive code → typed enrichment node
- Wk 5: Testing (pytest) → first green test suite
- Wk 6: Dev rules-extraction agent → end-to-end local; Phase 1 retro

### Phase 2 — Production-ize (Wk 7–12)
- Wk 7: Docker from scratch → self-authored image
- Wk 8: GCP Cloud Run deploy → **agent live, public URL**
- Wk 9: FastAPI + async → agent callable via API
- Wk 10: Slack → LangSmith feedback loop → working loop
- Wk 11: Eval pipeline (50+ examples, contamination discipline) → eval guarding agent
- Wk 12: MCP integration + **HONEST CHECKPOINT** (hunt now or keep building?)

### Phase 3 — Interview Surface (Wk 13–18)
- Wk 13: AI security & guardrails (prompt injection, PII, output constraints)
- Wk 14: Classical ML conversational fluency (bounded, 1 week)
- Wk 15: System design for AI (lived via the agent)
- Wk 16–17: Advanced retrieval (hybrid search, re-ranking) if agent needs it
- Wk 18: Portfolio consolidation → 2–3 clean repos + writeup

### Phase 4 — Polish & Launch (Wk 19–22)
- Wk 19–20: Mock interviews + resume rewrite (analyst → AI engineer narrative)
- Wk 21–22: Targeted applications from strength

---

## 7. Working Conventions

**Code style:**
- Section headers: `# ── SECTION NAME ────`
- Comments explain *why*, not *what*
- `# DOMAIN KNOWLEDGE:` and `# PYTHON CONCEPT:` prefix blocks
- Module-level constants in `UPPER_SNAKE_CASE`; clients created once at module level
- Type hints everywhere; specific exceptions; logging over print
- `if __name__ == "__main__":` block with tests
- All examples grounded in the gStore domain, never generic

**LLM providers:**
- Primary: OpenAI `gpt-5.4-mini` (`OPENAI_API_KEY`)
- Fallback: Gemini 2.5 Flash free tier (`GEMINI_API_KEY` — note: not `GOOGLE_API_KEY`)
- Available: Claude `claude-sonnet-4-6` (`ANTHROPIC_API_KEY`)

**LangSmith conventions:**
- Project = `gstore-ai-dev`, dataset = `gstore-alert-classification-v1`
- Classifier temp = 0.1, judge temp = 0, `max_concurrency=1` to avoid rate limits

**Security hygiene:**
- API keys never hardcoded — always `os.getenv(...)` from `.env`
- `.env`, `venv/`, `__pycache__/`, `chroma_db*/`, `.DS_Store` always in `.gitignore`
- A committed secret is public forever, even if later deleted — protect before first commit

**Project separation:**
- *This* (mentorship) project: curriculum, skills, the 22-week plan, learning state
- *Separate* project "gStore Replenishment QA Agent": build artifacts, schemas, codebase
- gStore proprietary code lives in the company's private GitHub org, never personal public

---

## 8. Teaching Rhythm (for the mentor)

1. **Why first** — the real problem, in plain language
2. **Mental model / analogy**
3. **Diagram** — how pieces connect, how data flows
4. **Small runnable code** — explained group-of-lines by group-of-lines
5. **Build incrementally** — naive version first, then refactor toward production
6. **Then the depth** — trade-offs, errors, logging, eval, observability, scaling, cost, security

Define every term on first use. Never assume SWE knowledge. Challenge thinking with
questions scaled to level. Stay a direct mentor — name real gaps, frame each as the next
thing to build. Track SWE fundamentals as a separate line from AI skills.
