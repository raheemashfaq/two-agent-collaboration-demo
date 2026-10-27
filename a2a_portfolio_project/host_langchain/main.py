"""Jeff Bezos agent implemented with LangChain."""

# host_langchain/main.py
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ENV_FILE = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_FILE)

app = FastAPI(title="Bezos Host Agent (LangChain)")

# --- A2A Protocol Models ---
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

# --- LangChain Setup ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an elite operations and e-commerce strategist. Focus on customer obsession, logistical efficiency, and cost optimization."),
    ("human", "Objective: {objective}\nContext: {context}\nOutput Format: {output_format}")
])

def create_chain():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini API key is missing. Add GOOGLE_API_KEY or GEMINI_API_KEY to {ENV_FILE}.",
        )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0.3,
    )
    return prompt | llm | StrOutputParser()

@app.post("/execute")
async def execute_task(request: A2ARequest):
    chain = create_chain()
    try:
        # 1. Execute LangChain
        result = chain.invoke({
            "objective": request.objective,
            "context": request.context or "No additional context provided.",
            "output_format": request.expected_output_format
        })

        # 2. Return A2A Compliant Response
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

# Run with: uvicorn host_langchain.main:app --reload --port 8002