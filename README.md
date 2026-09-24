# Marketing Strategy Agent

## 1. Project Overview

This project is a production-oriented AI Marketing Strategy Agent.

The purpose of the agent is to collect the information it needs about a business and then generate a useful, business-specific marketing strategy.

The agent should not behave like a fixed questionnaire where every business is asked the same list of questions.

Instead, the agent should understand the current business context, identify what important information is missing, decide which missing information is relevant to the current business and marketing goal, ask one useful question, process the user's answer, update its understanding, and then decide what information is needed next.

The agent continues this process until it has enough information to create a useful marketing strategy.

---

# 2. Main Goal

The overall flow is:

```text
User
  ↓
Business Context
  ↓
Analyze Information Gaps
  ↓
Identify Relevant Missing Information
  ↓
Prioritize Missing Information
  ↓
Generate ONE Question
  ↓
User Answers
  ↓
Update Business Context
  ↓
Analyze Information Gaps Again
  ↓
...
  ↓
Is sufficient information available?
      ↓                 ↓
     NO                YES
      ↓                 ↓
 Ask next question   Generate Strategy
```

The important point is that the agent should make decisions dynamically instead of following a predefined questionnaire.

---

# 3. Initial User Information

The user initially provides basic information such as:

* Company / Business
* Product / Service
* Marketing Goal
* Target Audience

These are starting context, not a complete questionnaire.

The agent may identify additional information that is necessary depending on the specific business.

---

# 4. Dynamic Information Collection

The project should NOT use a fixed list of questions.

Instead, we define a base framework of possible information requirements.

Example requirements include:

* Business information
* Product / Service
* Target Audience
* Marketing Goal
* USP / Differentiation
* Pricing
* Competitors
* Customer Pain Points
* Current Marketing Channels
* Previous Marketing Results
* Budget / Resources
* Sales / Conversion Process

This framework is guidance, not a closed list.

The LLM should determine:

1. Which information is relevant to the current business.
2. Which relevant information is already known.
3. Which relevant information is missing.
4. Which missing information is most important.
5. What question should be asked to obtain that information.

The LLM should generate only ONE question at a time.

---

# 5. Dynamic Requirements

The base information framework must not restrict the agent.

Sometimes a business may require information that is not present in the predefined framework.

In that situation, the LLM should be able to identify a new relevant information requirement.

Therefore:

```text
Base Framework
      +
LLM Reasoning
      ↓
Relevant Information Requirements
```

The system should not assume that every possible requirement can be known in advance.

The dynamic behavior applies to both:

* Which information is required
* How the question is asked

---

# 6. Handling Unavailable Information

The agent should understand that some information may not exist or may not be available.

For example:

```text
Agent:
What is your marketing budget?

User:
We don't have a fixed budget yet.
```

The agent should accept this information and continue.

It should not repeatedly ask the same question simply because the information is unavailable.

The agent should work with the information that is realistically available.

---

# 7. When Questioning Should Stop

The agent should not keep asking questions unnecessarily.

After each user answer, the agent should reassess the available context.

If the information is sufficient to create a useful marketing strategy:

```text
Sufficient Information
        ↓
Stop Questioning
        ↓
Generate Marketing Strategy
```

The agent should prioritize useful information collection rather than maximizing the number of questions.

---

# 8. Marketing Strategy Output

The final strategy should be business-specific.

The overall structure can remain consistent while the actual content changes according to the business.

Possible strategy sections include:

* Business Overview
* Marketing Objective
* Target Audience
* Customer Pain Points
* Value Proposition
* Positioning
* Marketing Channels
* Content Strategy
* Campaign Ideas
* Customer Acquisition Strategy
* Budget Considerations
* KPIs
* Action Plan

These sections are a starting structure and should be adapted when necessary based on the business context.

---

# 9. System Architecture

The planned high-level architecture is:

```text
React Frontend
      ↓
FastAPI Backend
      ↓
LangGraph Agent
      ↓
LLM Service
      ↓
Groq (initial LLM provider)
```

Responsibilities should be separated.

### React

Responsible for the user interface.

It should communicate with the backend through APIs.

The LLM API key must never be exposed to the React frontend.

### FastAPI

Responsible for the backend API layer.

It receives requests from the frontend and communicates with the agent/application services.

### LangGraph

Responsible for agent orchestration and stateful workflow.

It will eventually manage the iterative process of:

```text
Analyze
 → Identify missing information
 → Prioritize
 → Ask question
 → Receive answer
 → Update state
 → Analyze again
```

### LLM Service

The LLM should be accessed through a separate service layer rather than tightly coupling the agent logic to a specific provider.

Groq is the initial provider for development.

The architecture should allow the LLM provider/model to be changed later without rewriting the entire agent.

---

# 10. LLM Provider

The initial LLM provider is Groq.

The exact model should be configurable through application configuration/environment variables rather than being hard-coded throughout the project.

The company may provide a different paid model later.

Changing the provider/model should ideally require changing configuration and the LLM service implementation rather than rewriting the core agent architecture.

Any new model must still be tested for:

* Prompt compatibility
* Structured output compatibility
* Reliability
* Response quality
* Tool/function calling if used
* Rate limits
* Error handling

---

# 11. Technology Stack

Current planned technologies:

### Backend

* Python
* FastAPI
* LangGraph
* LangChain
* Pydantic
* Pydantic Settings
* python-dotenv
* Groq

### Frontend

* React

### Future Persistence

* PostgreSQL
* SQLAlchemy

The database should be introduced when persistence and conversation/session storage are implemented.

---

# 12. Project Structure

The agreed backend structure is:

```text
marketing_strategy_agent/
│
├── .venv/
├── requirements.txt
├── README.md
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── agent/
│       ├── api/
│       ├── core/
│       ├── models/
│       └── services/
│
└── frontend/
```

Important:

```text
backend/app/main.py
```

is the FastAPI application entry point.

Do not create an alternative root-level `main.py` unless the architecture is explicitly changed and approved.

The existing project structure should be inspected before creating or moving files.

---

# 13. Current Backend Entry Point

The FastAPI application currently contains a health-check endpoint.

Conceptually:

```text
GET /health
      ↓
{
    "status": "ok"
}
```

The health endpoint has already been tested successfully through FastAPI Swagger documentation.

Therefore the initial FastAPI foundation is complete.

---

# 14. Development Phases

The project should be developed phase by phase.

## Phase 1 — Backend Foundation

Set up:

* FastAPI
* Application entry point
* Basic configuration
* Health endpoint
* Basic backend structure

Status:

```text
DONE
```

---

## Phase 2 — Business Context / State

Define how the agent represents the current understanding of the business.

The state should eventually contain information such as:

```text
Business
Product / Service
Marketing Goal
Target Audience
Known Information
User Answers
Identified Requirements
Missing Information
Current Question
Conversation State
```

The exact implementation should be decided during this phase.

---

## Phase 3 — Information Requirements Framework

Create the base framework of possible marketing information requirements.

The framework should guide the agent but should not restrict it.

---

## Phase 4 — Dynamic Gap Analysis

Implement the logic that determines:

```text
What do we already know?
What is relevant?
What is missing?
What should we ask next?
```

The LLM should participate in this reasoning.

---

## Phase 5 — Dynamic Question Generation

Implement the ability to generate one appropriate question based on the highest-priority missing information.

The question should be based on the current business context rather than a static questionnaire.

---

## Phase 6 — Sufficiency Check

Implement logic for determining whether enough information has been collected to generate a useful marketing strategy.

If information is insufficient:

```text
Ask another question
```

If sufficient:

```text
Stop questioning
```

---

## Phase 7 — Strategy Generation

Generate the final business-specific marketing strategy using the collected context.

---

## Phase 8 — FastAPI Integration

Expose the agent functionality through proper backend APIs.

The API design should be decided based on the actual agent workflow rather than prematurely creating unnecessary endpoints.

---

## Phase 9 — React Frontend

Build the user interface that communicates with the FastAPI backend.

---

## Phase 10 — Persistence

Introduce database persistence for things such as:

* Conversations
* Agent sessions/threads
* Business information
* User answers
* Generated strategies

The exact database schema should be designed when this phase begins.

---

## Phase 11 — Testing and Production Hardening

Add:

* Unit tests
* Integration tests
* Input validation
* Error handling
* Logging
* Observability
* Security considerations
* Configuration management
* Reliability improvements
* Production deployment considerations

---

# 15. Development Principles

This is a production-oriented company project, not a quick demo or academic project.

The implementation should prioritize:

* Maintainability
* Reliability
* Scalability
* Clear separation of responsibilities
* Validation
* Error handling
* Security
* Testability
* Observability
* Configuration management
* Clean architecture

Avoid unnecessary complexity, but do not sacrifice proper architecture just to make the first version faster.

---

# 16. Important Rules for Development

These rules should be followed throughout the project.

### Rule 1 — Work Phase by Phase

Do not implement future phases unless explicitly requested.

If we are working on Phase 2, do not automatically implement Phase 3, Phase 4, database, frontend, etc.

---

### Rule 2 — Preserve Established Architecture

Do not change the agreed project structure or architecture without first explaining:

1. What needs to change.
2. Why it needs to change.
3. What existing code will be affected.

Wait for approval before making a significant architectural change.

---

### Rule 3 — Inspect Existing Code First

Before creating or modifying files:

* Inspect the existing project structure.
* Check existing implementations.
* Reuse existing components when appropriate.
* Avoid creating duplicate files or alternative implementations.

---

### Rule 4 — Do Not Replace Existing Decisions Silently

If a new approach is better than an existing approach, explain the difference first.

Do not silently replace an established design.

---

### Rule 5 — Keep Responsibilities Separate

For example:

```text
API Layer
    ↓
Application / Agent Layer
    ↓
LLM Service
    ↓
LLM Provider
```

Do not put all application logic into `main.py`.

---

### Rule 6 — Configuration Should Be Centralized

API keys, model names, database URLs, and environment-specific settings should not be hard-coded throughout the application.

Use configuration/environment variables.

---

### Rule 7 — Do Not Expose Secrets

API keys and other secrets must remain on the backend and must not be exposed to the React frontend or committed to Git.

---

### Rule 8 — Dynamic Agent Behavior

Do not turn the marketing agent into a hard-coded questionnaire.

The system should use defined information requirements as guidance while allowing the LLM to determine what is relevant and what should be asked next.

---

# 17. Current Project Status

Current status:

```text
Project Context              → Defined
Architecture                 → Defined
Technology Stack             → Defined
Project Structure             → Defined
FastAPI Foundation            → DONE
Health Endpoint               → Tested Successfully

Next Phase:
Business Context / State
```

Do not start the next phase automatically.

Wait for the developer/user instruction to begin the next phase.

---

# 18. Important Instruction for AI Coding Assistant

When working on this project, treat this README as the established project context.

Before making changes, inspect the current codebase and understand which development phase is currently active.

Do not assume that future phases should be implemented.

Do not change the architecture or folder structure without explaining the reason first.

If a requested implementation conflicts with the established architecture, point out the conflict and explain the alternatives before making the change.

The goal is to build the Marketing Strategy Agent incrementally, with a consistent architecture and clear understanding of each component.
