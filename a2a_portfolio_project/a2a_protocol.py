# a2a_protocol.py
from pydantic import BaseModel, Field
from typing import Optional

class A2ARequest(BaseModel):
    task_id: str = Field(..., description="Unique UUID for tracking")
    sender: str = Field(..., description="e.g., 'Elon_Client_Agent'")
    target_agent: str = Field(..., description="e.g., 'Zuckerberg_CrewAI_Host'")
    objective: str = Field(..., description="The specific task to execute")
    context: Optional[str] = Field(None, description="Background info for the task")
    expected_output_format: str = Field("text", description="text or json")

class A2AResponse(BaseModel):
    task_id: str
    status: str  # "success", "failed", "needs_more_info"
    result: str
    error_message: Optional[str] = None