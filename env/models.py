from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class SupportReward(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


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
    action_type: Literal["classify", "respond", "resolve"]
    ticket_id: str
    category: Optional[str] = None
    priority: Optional[str] = None
    route_to: Optional[str] = None
    template_key: Optional[str] = None
    resolution: Optional[str] = None
    internal_note: Optional[str] = None
    close_ticket: bool = False


class SupportObservation(Observation):
    task_id: str
    task_title: str
    objective: str
    queue_summary: QueueSummary
    tickets: List[TicketView] = Field(default_factory=list)
    last_event: str = ""
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
    task_id: str = ""
    task_title: str = ""
    objective: str = ""
    max_steps: int = 0
    total_reward: float = 0.0
    progress_score: float = 0.0
    completed: bool = False
    failure_reason: Optional[str] = None
    guidance: List[str] = Field(default_factory=list)
    tickets: List[TicketState] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
