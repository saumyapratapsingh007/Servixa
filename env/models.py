from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class SupportReward(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0, description="Reward applied for the last action.")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Named reward contributions for deterministic shaping.",
    )
    rationale: str = Field(default="", description="Human-readable explanation of the reward.")


class TicketView(BaseModel):
    ticket_id: str
    customer_name: str
    subject: str
    body: str
    channel: str
    customer_tier: str
    order_value: float
    hours_open: int
    sla_hours_remaining: int
    sentiment: str
    prior_contacts: int
    tags: List[str] = Field(default_factory=list)
    visible_notes: List[str] = Field(default_factory=list)
    allowed_templates: List[str] = Field(default_factory=list)
    current_status: str
    current_category: Optional[str] = None
    current_priority: Optional[str] = None
    current_route: Optional[str] = None
    last_response_template: Optional[str] = None
    resolution: Optional[str] = None
    closed: bool = False


class QueueSummary(BaseModel):
    total_tickets: int
    handled_tickets: int
    pending_tickets: int
    escalated_tickets: int
    closed_tickets: int
    max_steps: int
    step_count: int


class SupportAction(Action):
    action_type: Literal["classify", "respond", "resolve"] = Field(
        ...,
        description="Type of support operation to perform on a ticket.",
    )
    ticket_id: str = Field(..., description="Identifier of the ticket being updated.")
    category: Optional[str] = Field(default=None, description="Target issue category.")
    priority: Optional[str] = Field(default=None, description="Priority level.")
    route_to: Optional[str] = Field(default=None, description="Owning queue.")
    template_key: Optional[str] = Field(default=None, description="Customer response template identifier.")
    resolution: Optional[str] = Field(default=None, description="Resolution label for the ticket.")
    internal_note: Optional[str] = Field(default=None, description="Internal note recorded on the ticket.")
    close_ticket: bool = Field(default=False, description="Whether to close the ticket after resolution.")


class SupportObservation(Observation):
    task_id: str = Field(..., description="Current task identifier.")
    task_title: str = Field(..., description="Human-readable task title.")
    objective: str = Field(..., description="Objective the agent is trying to satisfy.")
    queue_summary: QueueSummary
    tickets: List[TicketView] = Field(default_factory=list)
    last_event: str = Field(default="", description="Summary of what happened on the last step.")
    progress_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_details: SupportReward = Field(default_factory=lambda: SupportReward(score=0.0))
    hints: List[str] = Field(default_factory=list)


class TicketState(BaseModel):
    ticket_id: str
    expected_category: str
    expected_priority: str
    expected_route: str
    expected_template: str
    expected_resolution: str
    must_close: bool
    unsafe_if_closed_early: bool = False
    customer_name: str
    subject: str
    body: str
    channel: str
    customer_tier: str
    order_value: float
    hours_open: int
    sla_hours_remaining: int
    sentiment: str
    prior_contacts: int
    tags: List[str] = Field(default_factory=list)
    allowed_templates: List[str] = Field(default_factory=list)
    visible_notes: List[str] = Field(default_factory=list)
    current_status: str = "open"
    current_category: Optional[str] = None
    current_priority: Optional[str] = None
    current_route: Optional[str] = None
    last_response_template: Optional[str] = None
    resolution: Optional[str] = None
    closed: bool = False
    action_log: List[Dict[str, Any]] = Field(default_factory=list)


class SupportState(State):
    task_id: str = Field(default="")
    task_title: str = Field(default="")
    objective: str = Field(default="")
    max_steps: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    progress_score: float = Field(default=0.0)
    completed: bool = Field(default=False)
    failure_reason: Optional[str] = Field(default=None)
    guidance: List[str] = Field(default_factory=list)
    tickets: List[TicketState] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
