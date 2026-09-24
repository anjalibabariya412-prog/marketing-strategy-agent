# Marketing Strategy Agent

## 1. Project Overview

This project is a production-oriented AI Marketing Strategy Agent.

The agent collects the information it needs about a business through a
dynamic, conversational flow, and then generates a business-specific
marketing strategy.

The agent does not behave like a fixed questionnaire. It understands the
current business context, identifies what important information is
missing, decides which missing information is relevant to *this specific*
business, asks one question at a time, processes the answer, updates its
understanding, and repeats until it has enough information to produce a
useful strategy.

---

## 2. Core Flow (as implemented)

```text
User sends an initial free-text message describing their business
              ↓
LLM extracts whatever of the 4 basics it can find
(company, product/service, marketing goal, target audience)
              ↓
Requirements library is loaded into a fresh conversation state
              ↓
Gap Analysis: LLM marks requirements that are clearly NOT_RELEVANT
to this business (e.g. "pricing_model" for a free non-profit) —
this happens WITHOUT asking the user
              ↓
Deterministic Sufficiency Check (pure Python, no LLM):
  - Are all must-have requirements KNOWN, UNAVAILABLE, or NOT_RELEVANT?
  - OR has the max question limit (default 6) been reached?
      ↙                                    ↘
     NO                                    YES
      ↓                                     ↓
Select the single highest-priority   Generate the final
missing requirement (LLM) and        MarketingStrategy (LLM),
generate a natural question (LLM)    leaving irrelevant sections
      ↓                               as null rather than filling
Graph PAUSES here (interrupt) and    them with generic content
waits for the user's answer
      ↓
User answers → answer is classified as KNOWN or UNAVAILABLE (LLM),
state is updated, loop returns to Gap Analysis
```

The key design principle: **what to ask is decided dynamically by the
LLM; whether to stop asking is decided deterministically by code.**
These two decisions are intentionally kept separate.

---

## 3. Key Design Decisions

These were deliberately chosen during development and should not be
silently changed:

### 3.1 Sufficiency is deterministic, never an LLM judgment call
The system never asks the LLM "do you have enough information?" Instead,
a plain Python function (`is_sufficient`) checks two conditions:
1. `get_unresolved_must_haves()` returns an empty list, OR
2. `len(qa_history) >= settings.max_questions` (default: 6)

This guarantees the user is never asked more than `max_questions`
questions, regardless of what the LLM "thinks."

### 3.2 The requirements library is industry-agnostic
`REQUIREMENTS_LIBRARY` (in `core/requirements_library.py`) describes
*what* each piece of information is and *why it generally matters*, in
plain, audience-neutral language. It does **not** tag requirements to
specific business types (no "relevant_for: SaaS" style tagging). The LLM
decides relevance for a specific business at runtime, using the
`analyze_relevance()` gap-analysis step. This means new business types
(a bakery, a SaaS company, a non-profit) all use the exact same library
with zero code changes.

Descriptions use audience-neutral language (e.g. "a customer, volunteer,
donor, or member") rather than assuming a commercial "customer"
relationship, since the agent is also used by non-profits and
membership-based organizations.

### 3.3 Four requirement statuses, not three
- `UNKNOWN` — not yet evaluated or asked about
- `KNOWN` — the user provided an actual answer
- `UNAVAILABLE` — applicable, but the user doesn't have/know the answer
  yet (e.g. "we haven't set a budget")
- `NOT_RELEVANT` — does not apply to this business at all (e.g.
  "pricing_model" for a free community program)

`UNAVAILABLE` and `NOT_RELEVANT` are deliberately distinct: the first
means "ask isn't wrong, but there's no answer yet"; the second means
"this question should never have been asked."

### 3.4 No duplicate data stores
Factual information about the business lives in exactly two places:
1. `BusinessContext`'s four fixed fields (the starting basics)
2. Each `InformationRequirement.value` (everything else)

There is intentionally no separate `known_information` dictionary or
similar duplicate store, to avoid data drifting out of sync.

### 3.5 The requirements library is a template, copied per conversation
`REQUIREMENTS_LIBRARY` is a single, static, shared list. Every new
conversation gets its own **deep copy** via `load_requirements_into_state()`
(`req.model_copy(deep=True)` for each item). Without this deep copy,
every conversation would share and corrupt the same in-memory objects.

### 3.6 Conversations pause and resume across separate HTTP calls
Because FastAPI is stateless per request, the agent uses LangGraph's
`interrupt()` combined with a checkpointer (currently `MemorySaver`, an
in-memory, non-persistent checkpointer — real persistence is planned for
Phase 10). Each conversation has a `thread_id`; invoking the graph with
`Command(resume=<user's answer>)` and the same `thread_id` continues
exactly where the conversation paused, rather than starting over.

Order of operations inside the "ask" node matters: on resume, the user's
answer must be processed (`process_answer_for_requirement`) **before**
picking the next question, otherwise the same question gets re-asked
(this was a real bug caught during Phase 8 testing — see Section 9).

### 3.7 Strategy sections are optional, not fixed-mandatory
`MarketingStrategy` (9 defined sections + `additional_sections`) uses
`Optional[str] = None` for every section. If a section is genuinely not
relevant to a business (e.g. "Competitive Positioning" for an
organization with no real competitors), the LLM is instructed to leave
it `None` rather than writing generic filler content. Sections that are
`None` should be hidden/skipped entirely when the strategy is displayed
to the user (frontend concern — the user should only ever see a clean,
complete-looking document, never an empty or "N/A" section).

---

## 4. System Architecture

```text
React Frontend (Phase 9, in progress)
      ↓  HTTP
FastAPI Backend (/start, /reply, /strategy)
      ↓
LangGraph Agent (StateGraph with interrupt/checkpoint)
      ↓
Agent Logic Layer (gap analysis, question selection,
sufficiency check, answer processing, strategy generation)
      ↓
LLM Service (provider-agnostic wrapper)
      ↓
Groq (current provider) — model configurable via .env
```

Responsibilities are kept separate: API layer → agent/orchestration layer
→ LLM service → LLM provider. No business logic lives in `main.py`.

---

## 5. LLM Provider

- **Provider:** Groq (free tier)
- **Model:** `openai/gpt-oss-120b` (configurable via `GROQ_MODEL` in `.env`)
- Structured output is obtained via Groq's JSON mode
  (`response_format={"type": "json_object"}`), validated as needed.
- The model and provider are read from `core/config.py` (Pydantic
  Settings), never hardcoded in business logic, so switching providers
  later should only require changes to `services/llm_service.py` and
  configuration.

---

## 6. Technology Stack

### Backend
- Python, FastAPI
- LangGraph (StateGraph, `interrupt()`, `MemorySaver` checkpointer)
- Pydantic + Pydantic Settings
- python-dotenv
- `groq` Python SDK

### Frontend
- React (via Vite)

### Future Persistence (Phase 10, not yet implemented)
- A real database (PostgreSQL was the original plan) to replace
  `MemorySaver` with persistent, restart-safe conversation storage.

---

## 7. Project Structure (actual)

```text
marketing_strategy_agent/
│
├── .venv/
├── requirements.txt
├── README.md
├── .env / .env.example / .gitignore
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── agent/
│       │   ├── graph.py                 (LangGraph StateGraph + interrupt/resume)
│       │   ├── gap_analysis.py          (analyze_relevance)
│       │   ├── question_selection.py    (select_next_requirement, generate_question,
│       │   │                              prepare_next_question)
│       │   ├── sufficiency_check.py     (is_sufficient — pure logic, no LLM)
│       │   ├── answer_processing.py     (extract_initial_context,
│       │   │                              process_answer_for_requirement)
│       │   └── strategy_generation.py   (generate_strategy)
│       ├── api/
│       │   └── conversation.py          (/start, /reply, /strategy endpoints)
│       ├── schemas/
│       │   └── conversation.py          (request/response Pydantic schemas)
│       ├── core/
│       │   ├── config.py                (Settings: groq_api_key, groq_model,
│       │   │                              max_questions)
│       │   └── requirements_library.py  (REQUIREMENTS_LIBRARY,
│       │                                  load_requirements_into_state)
│       ├── models/
│       │   ├── business_context.py      (BusinessContext — 4 starting basics)
│       │   ├── information_requirement.py (InformationRequirement,
│       │   │                                RequirementStatus enum)
│       │   ├── qa_turn.py               (QATurn — one Q&A exchange)
│       │   ├── agent_state.py           (MarketingAgentState — central state,
│       │   │                              includes final_strategy field)
│       │   └── marketing_strategy.py    (MarketingStrategy — 9 optional
│       │                                  sections + additional_sections)
│       └── services/
│           └── llm_service.py           (get_llm_response, LLMServiceError)
│
└── frontend/            (Phase 9, in progress — Vite + React)
```

`backend/app/main.py` is the FastAPI entry point. Do not create an
alternative root-level `main.py`.

---

## 8. Development Phases — Status

| Phase | Description | Status |
|---|---|---|
| 1 | Backend Foundation (FastAPI, config, health check) | ✅ DONE |
| 2 | Business Context / State (Pydantic models) | ✅ DONE |
| 3 | Information Requirements Framework (14-item library) | ✅ DONE |
| 4 | Dynamic Gap Analysis (relevance filtering) | ✅ DONE |
| 5 | Dynamic Question Generation (selection + phrasing) | ✅ DONE |
| 6 | Sufficiency Check (deterministic) | ✅ DONE |
| 7 | Strategy Generation (structured, optional sections) | ✅ DONE |
| 8 | FastAPI Integration (LangGraph interrupt/resume, endpoints) | ✅ DONE — tested end-to-end |
| 9 | React Frontend | 🔧 IN PROGRESS |
| 10 | Persistence (real database, replacing MemorySaver) | ⏳ PENDING |
| 11 | Testing & Production Hardening | ⏳ PENDING |

Do not start a phase beyond what is currently in progress without
explicit instruction.

---

## 9. Known Issues / Improvements Deferred to Phase 11

These were identified during development and testing, and are
deliberately deferred rather than fixed immediately, since they don't
block frontend or database work:

1. **LLM arithmetic is not reliable.** Generated budget breakdowns have
   been observed to sometimes not sum correctly. No validation layer
   currently checks this. Plan: add a simple Python-side arithmetic
   sanity check after strategy generation, and/or a user-facing
   disclaimer that figures should be verified.
2. **No user-facing AI-disclaimer yet.** The generated strategy does not
   currently tell the user "this is AI-generated, verify important
   figures before acting on them." Should be added in the frontend
   and/or API response.
3. **Limited test coverage across business types.** Testing so far has
   focused heavily on one example business (a non-profit / volunteer
   organization). Should test at least one clearly commercial business
   and one B2B/SaaS business to confirm the dynamic relevance filtering
   and strategy quality generalize well.
4. **A real "some sections are None" test is still pending.** All
   strategy tests so far have resulted in all 9 sections being filled;
   the "leave irrelevant sections null" path has not yet been observed
   in a real test run.
5. **A bug was found and fixed during Phase 8 testing:** the "ask" node
   originally called `prepare_next_question()` before processing the
   resumed answer, causing the same question to be re-asked. Fixed by
   reordering: on resume, `interrupt()`'s return value is processed via
   `process_answer_for_requirement()` *before* selecting the next
   question. If this logic is ever refactored, this ordering constraint
   must be preserved.

---

## 10. Important Rules for Development

### Rule 1 — Work Phase by Phase
Do not implement future phases unless explicitly requested.

### Rule 2 — Preserve Established Architecture
Do not change the agreed project structure or architecture without first
explaining what needs to change, why, and what existing code is affected.
Wait for approval before significant architectural changes.

### Rule 3 — Inspect Existing Code First
Check existing files and structure before creating or modifying anything.
Reuse existing components; avoid duplicate implementations.

### Rule 4 — Do Not Replace Existing Decisions Silently
If a file that was previously marked complete needs to change (e.g.
adding a parameter to `llm_service.py`), say so explicitly rather than
changing it quietly as a side effect of another task.

### Rule 5 — Keep Responsibilities Separate
API layer → agent/orchestration layer → LLM service → LLM provider.
No business logic in `main.py`.

### Rule 6 — Configuration Should Be Centralized
API keys, model names, and tunable values (like `max_questions`) belong
in `core/config.py` / `.env`, never hardcoded inline.

### Rule 7 — Do Not Expose Secrets
API keys must stay on the backend, never in frontend code or committed
to Git (`.env` is gitignored; `.env.example` holds placeholders only).

### Rule 8 — Dynamic Agent Behavior
Never turn the marketing agent into a hardcoded questionnaire. Use
`REQUIREMENTS_LIBRARY` as guidance; let the LLM determine relevance and
priority at runtime. Never tag requirements to specific business types.

### Rule 9 — Sufficiency Logic Stays Deterministic
`is_sufficient()` must remain pure Python logic with no LLM call. The LLM
may decide *what* to ask; only code decides *whether to stop*.

### Rule 10 — Commit Working States
After each task is completed and tested/confirmed, commit the change
with `git add . && git commit -m "..."` before starting the next task,
so any future broken change can be safely reverted with `git checkout .`

---

## 11. Current Status Summary

```text
Phases 1–8:  DONE, tested end-to-end via Swagger UI
             (full conversation → dynamic questions → strategy generation
             flow confirmed working over real HTTP requests)
Phase 9:     React frontend — in progress
Phase 10:    Persistence — not started
Phase 11:    Hardening — not started, see Section 9 for the known
             backlog of issues to address then
```

Do not start Phase 10 or 11 work automatically. Wait for explicit
instruction.