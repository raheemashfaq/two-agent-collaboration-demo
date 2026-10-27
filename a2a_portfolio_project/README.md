# A2A Portfolio Project

A multi-framework agent-to-agent system where an ADK orchestrator
delegates tasks to CrewAI and LangChain host agents.

## Architecture

- **client_adk/** — ADK orchestrator. Decides which host to call.
- **host_crewai/** — CrewAI host on :8001 (Zuckerberg persona).
- **host_langchain/** — LangChain host on :8002.
- **a2a_protocol.py** — shared A2ARequest / A2AResponse schema.

## How to run

Three terminals:

  uvicorn host_crewai.main:app --port 8001
  uvicorn host_langchain.main:app --port 8002
  adk run client_adk

## Flow

User → ADK orchestrator → (A2ARequest) → host → (A2AResponse) → orchestrator → user