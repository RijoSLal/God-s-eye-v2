from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class AnalystSchema(BaseModel):
    # technical data for audit
    info: str = Field(..., description = "Factual data and execution evidence to be stored for auditing.")

class Nuance(BaseModel):
    # extracted data point metadata
    value: str
    source_task: str

class AnalystReport(BaseModel):
    # final audit result schema
    status: Literal["done", "failure"] = Field(..., description="The definitive outcome. Use 'done' only if the target goal is fully verified as achieved. Use 'failure' if any part of the goal remains unfulfilled.")
    findings_summary: str = Field(..., description="A direct synthesis of the verified results and data found during execution.")
    failure_reason: Optional[str] = Field(None, description="The technical blockage or missing condition that prevented success")
    nuances: Dict[str, Any] = Field(default_factory=dict, description="A collection of specific data points and factual values extracted during the process, mapped to unique keys.")


# possible task states
TaskStatus = Literal["pending", "running", "done", "skipped"]
# possible workflow states
GoalStatus = Literal["running", "success", "failure"]

class Task(BaseModel):
    # individual unit of work
    description: str = Field(..., description="A discrete execution instruction. Define one specific action that produces a verifiable outcome.")
    expected_output: str = Field(..., description="The specific data or system state required to verify the successful completion of this task.")
    status: TaskStatus = Field("pending", description="The current execution state. Initialize as 'pending'.")
    depends_on: List[str] = Field(default_factory=list, description="IDs of tasks that must reach 'done' status before this task can start.")
    findings_summary: Optional[str] = Field(None, description="The actual data or results produced after task execution.")
    failure_reason: Optional[str] = Field(None, description="The error or reason for non-completion if the task fails.")

class PlanState(BaseModel):
    # global execution context
    goal: str
    status: GoalStatus = "running"
    tasks: Dict[str, Task]
    nuances: Dict[str, Nuance] = Field(default_factory=dict)
    abort_reason: Optional[str] = None


class CreatePlanOutput(BaseModel):
    # initial planning result
    plan_explanation: str = Field(..., description="A technical rationale for the chosen task sequence and how it fulfills the target goal.")
    tasks: Dict[str, Task]

class EditTaskOutput(BaseModel):
    # task refinement result
    replacement_task: Task

class ReplanOutput(BaseModel):
    # workflow restructuring result
    tasks: Dict[str, Task]

class OrchestratorRunSchema(BaseModel):
    # orchestrator tool input
    task: str = Field(..., description="The objective to be achieved through multi-step execution.")

class MemoryQuerySchema(BaseModel):
    # memory search tool input
    query: str = Field(..., description="The technical keyword or factual query to retrieve from historical execution logs.")
