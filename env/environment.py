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
                total_reward=0,
                progress_score=0,
                completed=False,
                failure_reason=None,
                action_history=[],
            )

            reward = SupportReward(
                score=0,
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
                    score=-5,
                    components={"late_action_penalty": -5},
                    rationale="Episode already ended.",
                )
                return self._build_observation(
                    reward=reward,
                    last_event="Rejected: episode complete.",
                    done=True,
                )

            self._state.step_count += 1
            reward = SupportReward(score=0, components={}, rationale="")

            ticket = self._ticket_lookup(action.ticket_id)
            if ticket is None:
                reward.score = -20
                reward.components["invalid_ticket_penalty"] = -20
                reward.rationale = "Invalid ticket."
                self._state.total_reward += reward.score
                self._state.action_history.append({"action": action.model_dump(), "valid": False})
                self._refresh_progress()
                return self._finalize_step(reward, f"Ticket {action.ticket_id} not found.")

            missing = self._missing_required_field(action)
            if missing:
                reward.score = -12
                reward.components["invalid_action_penalty"] = -12
                reward.rationale = f"Missing `{missing}`."
                self._record_ticket_action(ticket, action, False)
                self._state.total_reward += reward.score
                self._refresh_progress()
                return self._finalize_step(reward, f"Missing field `{missing}`.")

            if action.action_type == "classify":
                self._apply_classification(ticket, action, reward)
            elif action.action_type == "respond":
                self._apply_response(ticket, action, reward)
            elif action.action_type == "resolve":
                self._apply_resolution(ticket, action, reward)

            reward.components["efficiency_penalty"] = reward.components.get("efficiency_penalty", 0) - 1
            reward.score = sum(reward.components.values())

            if not reward.rationale:
                reward.rationale = "Action applied."

            self._record_ticket_action(ticket, action, True)
            self._state.total_reward += reward.score

            self._refresh_progress()
            self._update_done_flags()

            return self._finalize_step(
                reward,
                f"{action.action_type} applied to {ticket.ticket_id}",
            )

    def _apply_classification(self, ticket, action, reward):
        ticket.current_category = action.category
        ticket.current_priority = action.priority
        ticket.current_route = action.route_to

        reward.components["category"] = 12 if action.category == ticket.expected_category else -5
        reward.components["priority"] = 8 if action.priority == ticket.expected_priority else -4
        reward.components["routing"] = 10 if action.route_to == ticket.expected_route else -6

    def _apply_response(self, ticket, action, reward):
        ticket.last_response_template = action.template_key

        reward.components["response_template"] = 8 if action.template_key == ticket.expected_template else -5

    def _apply_resolution(self, ticket, action, reward):
        ticket.resolution = action.resolution
        ticket.closed = action.close_ticket

        reward.components["resolution"] = 12 if action.resolution == ticket.expected_resolution else -7
        reward.components["closure"] = 6 if action.close_ticket == ticket.must_close else -8

    def _refresh_progress(self):
        report = grade_state(self._state)
        self._state.progress_score = int(report["score"])

    def _update_done_flags(self):
        if self._state.step_count >= self._state.max_steps:
            self._state.completed = True

    def _finalize_step(self, reward, last_event):
        return self._build_observation(reward, last_event, self._state.completed)
