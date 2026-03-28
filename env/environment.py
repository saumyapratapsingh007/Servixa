from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from .grader import grade_state
from .models import (
    QueueSummary,
    SupportAction,
    SupportObservation,
    SupportReward,
    SupportState,
    TicketState,
    TicketView,
)
from .tasks import TASKS, get_task


ACTION_REQUIRED_FIELDS = {
    "classify": ("category", "priority", "route_to"),
    "respond": ("template_key",),
    "resolve": ("resolution",),
}


class SupportOpsEnvironment(Environment[SupportAction, SupportObservation, SupportState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        self._lock = Lock()
        self._task_index = 0
        self._state = SupportState(episode_id=str(uuid4()))
        self.reset(task_id=TASKS[0]["id"])

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: object,
    ) -> SupportObservation:
        del seed, kwargs
        with self._lock:
            if task_id is None:
                task = TASKS[self._task_index % len(TASKS)]
                self._task_index += 1
            else:
                task = get_task(task_id)

            tickets = [TicketState(**deepcopy(ticket)) for ticket in task["tickets"]]
            self._state = SupportState(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
                task_id=str(task["id"]),
                task_title=str(task["title"]),
                objective=str(task["objective"]),
                max_steps=int(task["max_steps"]),
                guidance=list(task["guidance"]),
                tickets=tickets,
                total_reward=0.0,
                progress_score=0.0,
                completed=False,
                failure_reason=None,
                action_history=[],
            )
            reward = SupportReward(
                score=0.0,
                rationale="Environment reset. Ready to handle tickets.",
            )
            return self._build_observation(reward=reward, last_event="Queue ready.", done=False)

    def step(
        self,
        action: SupportAction,
        timeout_s: Optional[float] = None,
        **kwargs: object,
    ) -> SupportObservation:
        del timeout_s, kwargs
        with self._lock:
            if self._state.completed:
                reward = SupportReward(
                    score=-0.05,
                    components={"late_action_penalty": -0.05},
                    rationale="This episode has already ended. Reset before sending more actions.",
                )
                return self._build_observation(
                    reward=reward,
                    last_event="Action rejected because the episode is already complete.",
                    done=True,
                )

            self._state.step_count += 1
            reward = SupportReward(score=0.0, components={}, rationale="")
            ticket = self._ticket_lookup(action.ticket_id)
            if ticket is None:
                reward.score = -0.20
                reward.components["invalid_ticket_penalty"] = -0.20
                reward.rationale = "The action referenced an unknown ticket."
                self._state.total_reward += reward.score
                self._state.action_history.append({"action": action.model_dump(), "valid": False})
                self._refresh_progress()
                return self._finalize_step(reward=reward, last_event=f"Ticket {action.ticket_id} was not found.")

            missing_field = self._missing_required_field(action)
            if missing_field is not None:
                reward.score = -0.12
                reward.components["invalid_action_penalty"] = -0.12
                reward.rationale = f"A {action.action_type} action requires `{missing_field}`."
                self._record_ticket_action(ticket, action, valid=False)
                self._state.total_reward += reward.score
                self._refresh_progress()
                return self._finalize_step(
                    reward=reward,
                    last_event=f"Action rejected for ticket {ticket.ticket_id}: `{missing_field}` is missing.",
                )

            if action.action_type == "classify":
                self._apply_classification(ticket, action, reward)
            elif action.action_type == "respond":
                self._apply_response(ticket, action, reward)
            elif action.action_type == "resolve":
                self._apply_resolution(ticket, action, reward)

            reward.components["efficiency_penalty"] = reward.components.get("efficiency_penalty", 0.0) - 0.01
            reward.score = round(sum(reward.components.values()), 4)
            if not reward.rationale:
                reward.rationale = "Ticket updated successfully."

            self._record_ticket_action(ticket, action, valid=True)
            self._state.total_reward = round(self._state.total_reward + reward.score, 4)
            self._refresh_progress()
            self._update_done_flags()
            return self._finalize_step(
                reward=reward,
                last_event=f"{action.action_type.title()} action applied to ticket {ticket.ticket_id}.",
            )

    @property
    def state(self) -> SupportState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SupportOps OpenEnv",
            description="Customer support triage environment with deterministic grading and shaped rewards.",
            version="1.0.0",
            author="OpenEnv Hackathon Build",
        )

    def _ticket_lookup(self, ticket_id: str) -> Optional[TicketState]:
        for ticket in self._state.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _missing_required_field(self, action: SupportAction) -> Optional[str]:
        for field_name in ACTION_REQUIRED_FIELDS[action.action_type]:
            if getattr(action, field_name, None) in (None, ""):
                return field_name
        return None

    def _apply_classification(self, ticket: TicketState, action: SupportAction, reward: SupportReward) -> None:
        ticket.current_category = action.category
        ticket.current_priority = action.priority
        ticket.current_route = action.route_to
        ticket.current_status = "triaged"
        reward.components["category"] = 0.12 if action.category == ticket.expected_category else -0.05
        reward.components["priority"] = 0.08 if action.priority == ticket.expected_priority else -0.04
        reward.components["routing"] = 0.10 if action.route_to == ticket.expected_route else -0.06
        if action.route_to == "frontline" and ticket.expected_route in {"security", "trust_safety", "tech_ops"}:
            reward.components["specialist_miss_penalty"] = -0.08
            reward.rationale = "The frontline queue kept a ticket that should be owned by a specialist team."

    def _apply_response(self, ticket: TicketState, action: SupportAction, reward: SupportReward) -> None:
        ticket.last_response_template = action.template_key
        ticket.current_status = "customer_updated"
        if action.internal_note:
            ticket.visible_notes.append(action.internal_note)
        reward.components["response_template"] = 0.08 if action.template_key == ticket.expected_template else -0.05
        if action.template_key not in ticket.allowed_templates:
            reward.components["template_policy_penalty"] = -0.09
            reward.rationale = "The selected response template is not available for this ticket."

    def _apply_resolution(self, ticket: TicketState, action: SupportAction, reward: SupportReward) -> None:
        ticket.resolution = action.resolution
        ticket.closed = action.close_ticket
        ticket.current_status = "closed" if ticket.closed else "pending_specialist"
        if action.internal_note:
            ticket.visible_notes.append(action.internal_note)
        reward.components["resolution"] = 0.12 if action.resolution == ticket.expected_resolution else -0.07
        reward.components["closure"] = 0.06 if action.close_ticket == ticket.must_close else -0.08
        if ticket.unsafe_if_closed_early and action.close_ticket:
            correctly_escalated = (
                ticket.current_route == ticket.expected_route and action.resolution == ticket.expected_resolution
            )
            if not correctly_escalated:
                reward.components["unsafe_closure_penalty"] = -0.25
                reward.rationale = "Unsafe closure: this ticket still requires specialist follow-up."

    def _record_ticket_action(self, ticket: TicketState, action: SupportAction, valid: bool) -> None:
        payload = action.model_dump()
        payload["valid"] = valid
        ticket.action_log.append(payload)
        self._state.action_history.append(payload)

    def _refresh_progress(self) -> None:
        report = grade_state(self._state)
        self._state.progress_score = float(report["score"])

    def _update_done_flags(self) -> None:
        all_minimally_handled = all(
            ticket.current_category
            and ticket.current_priority
            and ticket.current_route
            and ticket.last_response_template
            and ticket.resolution
            for ticket in self._state.tickets
        )
        if all_minimally_handled:
            self._state.completed = True
            self._state.failure_reason = None
            return

        if self._state.step_count >= self._state.max_steps:
            self._state.completed = True
            self._state.failure_reason = "step_limit_reached"

    def _build_observation(self, reward: SupportReward, last_event: str, done: bool) -> SupportObservation:
        tickets = [
            TicketView(
                ticket_id=ticket.ticket_id,
                customer_name=ticket.customer_name,
                subject=ticket.subject,
                body=ticket.body,
                channel=ticket.channel,
                customer_tier=ticket.customer_tier,
                order_value=ticket.order_value,
                hours_open=ticket.hours_open,
                sla_hours_remaining=ticket.sla_hours_remaining,
                sentiment=ticket.sentiment,
                prior_contacts=ticket.prior_contacts,
                tags=list(ticket.tags),
                visible_notes=list(ticket.visible_notes),
                allowed_templates=list(ticket.allowed_templates),
                current_status=ticket.current_status,
                current_category=ticket.current_category,
                current_priority=ticket.current_priority,
                current_route=ticket.current_route,
                last_response_template=ticket.last_response_template,
                resolution=ticket.resolution,
                closed=ticket.closed,
            )
            for ticket in self._state.tickets
        ]
        handled = sum(1 for ticket in self._state.tickets if ticket.resolution is not None)
        escalated = sum(
            1 for ticket in self._state.tickets if ticket.current_route in {"security", "trust_safety", "tech_ops", "billing", "logistics"}
        )
        closed = sum(1 for ticket in self._state.tickets if ticket.closed)
        queue_summary = QueueSummary(
            total_tickets=len(self._state.tickets),
            handled_tickets=handled,
            pending_tickets=len(self._state.tickets) - handled,
            escalated_tickets=escalated,
            closed_tickets=closed,
            max_steps=self._state.max_steps,
            step_count=self._state.step_count,
        )
        return SupportObservation(
            done=done,
            reward=reward.score,
            metadata={
                "episode_id": self._state.episode_id,
                "total_reward": self._state.total_reward,
                "failure_reason": self._state.failure_reason,
            },
            task_id=self._state.task_id,
            task_title=self._state.task_title,
            objective=self._state.objective,
            queue_summary=queue_summary,
            tickets=tickets,
            last_event=last_event,
            progress_score=self._state.progress_score,
            reward_details=reward,
            hints=list(self._state.guidance),
        )

    def _finalize_step(self, reward: SupportReward, last_event: str) -> SupportObservation:
        return self._build_observation(reward=reward, last_event=last_event, done=self._state.completed)


_ENV_SINGLETON: Optional[SupportOpsEnvironment] = None
_ENV_LOCK = Lock()


def get_environment() -> SupportOpsEnvironment:
    global _ENV_SINGLETON
    with _ENV_LOCK:
        if _ENV_SINGLETON is None:
            _ENV_SINGLETON = SupportOpsEnvironment()
        return _ENV_SINGLETON
