"""Mark Zuckerberg agent implemented with CrewAI."""

# host_crewai/main.py
import os
import uuid
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

# CrewAI Imports
from crewai import Agent, Task, Crew, LLM, Process

ENV_FILE = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_FILE)

app = FastAPI(title="Zuckerberg Host Agent (CrewAI)")

# --- A2A Protocol Models (Mirroring root a2a_protocol.py) ---
class A2ARequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    target_agent: str
    objective: str
    context: Optional[str] = None
    expected_output_format: str = "text"

class A2AResponse(BaseModel):
    task_id: str
    status: str
    result: str
    error_message: Optional[str] = None

def create_zuckerberg_agent() -> Agent:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            f"Gemini API key is missing. Add GOOGLE_API_KEY or GEMINI_API_KEY to {ENV_FILE}."
        )

    llm = LLM(
        model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}",
        api_key=api_key,
        temperature=0.3,
    )
    return Agent(
        role="Tech Product Visionary",
        goal="Analyze tech product strategies, scalability, and innovation opportunities.",
        backstory="You are a world-class tech product leader focused on rapid iteration, user growth, and scalable architecture.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

@app.post("/execute")
async def execute_task(request: A2ARequest):
    try:
        zuckerberg_agent = create_zuckerberg_agent()

        # 1. Formulate the specific task for CrewAI
        task_description = f"Objective: {request.objective}"
        if request.context:
            task_description += f"\nContext: {request.context}"
        task_description += f"\nFormat the output as {request.expected_output_format}."

        crew_task = Task(
            description=task_description,
            expected_output=f"A strategic analysis in {request.expected_output_format} format.",
            agent=zuckerberg_agent
        )

        # 2. Execute the Crew
        crew = Crew(
            agents=[zuckerberg_agent],
            tasks=[crew_task],
            process=Process.sequential,
            verbose=False # Set to True if you want to see CrewAI thinking in console
        )

        result = await crew.kickoff_async()

        # 3. Return A2A Compliant Response
        return A2AResponse(
            task_id=request.task_id,
            status="success",
            result=str(result)
        )

    except Exception as e:
        return A2AResponse(
            task_id=request.task_id,
            status="failed",
            result="",
            error_message=str(e)
        )

# Run with: uvicorn host_crewai.main:app --reload --port 8001