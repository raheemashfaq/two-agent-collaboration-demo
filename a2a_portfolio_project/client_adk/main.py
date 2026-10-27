"""ADK host agent that orchestrates CrewAI and LangChain hosts via A2A."""
import uuid
import requests
from a2a_protocol import A2ARequest, A2AResponse

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

HOSTS = {
    "Zuckerberg": "http://127.0.0.1:8001/execute",
    "LangChain":  "http://127.0.0.1:8002/execute",
}


def _call_host(target: str, objective: str, context: str | None = None) -> dict:
    req = A2ARequest(
        task_id=str(uuid.uuid4()),
        sender="ADK_Host",
        target_agent=target,
        objective=objective,
        context=context,
    )
    try:
        r = requests.post(HOSTS[target], json=req.model_dump(), timeout=180)
        r.raise_for_status()
        resp = A2AResponse(**r.json())
        return {"status": resp.status, "result": resp.result}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def ask_zuckerberg(objective: str, context: str = "") -> dict:
    """Ask the CrewAI/Zuckerberg host for product strategy analysis.

    Args:
        objective: The strategic question to answer.
        context: Optional background information.
    """
    return _call_host("Zuckerberg", objective, context or None)


def ask_langchain_agent(objective: str, context: str = "") -> dict:
    """Ask the LangChain host for its analysis.

    Args:
        objective: The question to answer.
        context: Optional background information.
    """
    return _call_host("LangChain", objective, context or None)


root_agent = Agent(
    name="orchestrator",
    model="gemini-3.6-flash",
    description="Host agent that delegates strategy questions to remote A2A agents.",
    instruction=(
        "You are a host agent. You do not answer questions yourself. "
        "You delegate to remote agents using your tools. "
        "If the user asks a product/strategy question, call ask_zuckerberg. "
        "If the user asks for a second opinion or a comparison, also call ask_langchain_agent. "
        "Then summarize the results."
    ),
    tools=[ask_zuckerberg, ask_langchain_agent],
)