from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class AnalystSchema(BaseModel):
    # technical data for audit
    info: str = Field(..., description = "critical execution data for future analysis and auditing")

class Nuance(BaseModel):
    # extracted data point metadata
    value: str
    source_task: str

class AnalystReport(BaseModel):
    # final audit result schema
    status: Literal["done", "failure"] = Field(..., description="The final status of the execution flow")
    findings_summary: str = Field(..., description="A concise summary of all critical data points, findings, and results recorded during execution")
    failure_reason: Optional[str] = Field(None, description="Detailed reason for failure, or None if status is done")
    nuances: Dict[str, Any] = Field(default_factory=dict, description="Key technical data points extracted (e.g., version: 1.2, ip: 127.0.0.1)")


# possible task states
TaskStatus = Literal["pending", "running", "done", "skipped"]
# possible workflow states
GoalStatus = Literal["running", "success", "failure"]

class Task(BaseModel):
    # individual unit of work
    description: str
    status: TaskStatus = "pending"
    depends_on: List[str] = Field(default_factory=list)
    findings_summary: Optional[str] = None
    failure_reason: Optional[str] = None

class PlanState(BaseModel):
    # global execution context
    goal: str
    status: GoalStatus = "running"
    tasks: Dict[str, Task]
    nuances: Dict[str, Nuance] = Field(default_factory=dict)
    abort_reason: Optional[str] = None


class CreatePlanOutput(BaseModel):
    # initial planning result
    plan_explanation: str = Field(..., description="A brief explanation of how the tasks collectively achieve 100% of the user's goal")
    tasks: Dict[str, Task]

class EditTaskOutput(BaseModel):
    # task refinement result
    replacement_task: Task

class ReplanOutput(BaseModel):
    # workflow restructuring result
    tasks: Dict[str, Task]

class SupervisorRunSchema(BaseModel):
    # orchestrator tool input
    task: str = Field(..., description="MANDATORY: Multi-step local request to execute.")

class MemoryQuerySchema(BaseModel):
    # memory search tool input
    query: str = Field(..., description="Fact or topic to query in memory.")
