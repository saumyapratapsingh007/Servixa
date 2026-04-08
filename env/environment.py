from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from .grader import grade_state
from .models import *


ACTION_REQUIRED_FIELDS = {
    "classify": ("category", "priority", "route_to"),
    "respond": ("template_key",),
    "resolve": ("resolution",),
}


class SupportOpsEnvironment(Environment):
    def __init__(self):
        self._lock = Lock()
        self._state = SupportState(episode_id=str(uuid4()))

    def reset(self, task_id=None, **kwargs):
        from .tasks import get_task
        task = get_task(task_id)

        tickets = [TicketState(**deepcopy(t)) for t in task["tickets"]]

        self._state = SupportState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task["id"],
            task_title=task["title"],
            objective=task["objective"],
            max_steps=task["max_steps"],
            tickets=tickets,
            total_reward=0.0,
            progress_score=0.0,
        )

        return self._obs(SupportReward(score=0.0), "reset")

    def step(self, action: SupportAction):
        self._state.step_count += 1
        reward = SupportReward(score=0.0, components={})

        reward.score = round(sum(reward.components.values()), 4)
        self._state.total_reward = round(self._state.total_reward + reward.score, 4)

        report = grade_state(self._state)
        self._state.progress_score = float(report["score"])

        return self._obs(reward, "step")

    def _obs(self, reward, event):
        return SupportObservation(
            done=False,
            reward=reward.score,
            metadata={},
            task_id=self._state.task_id,
            task_title=self._state.task_title,
            objective=self._state.objective,
            queue_summary=QueueSummary(
                total_tickets=len(self._state.tickets),
                handled_tickets=0,
                pending_tickets=0,
                escalated_tickets=0,
                closed_tickets=0,
                max_steps=self._state.max_steps,
                step_count=self._state.step_count,
            ),
            tickets=[],
            last_event=event,
            progress_score=self._state.progress_score,
            reward_details=reward,
        )


_ENV_SINGLETON: Optional[SupportOpsEnvironment] = None
_ENV_LOCK = Lock()


def get_environment():
    global _ENV_SINGLETON
    with _ENV_LOCK:
        if _ENV_SINGLETON is None:
            _ENV_SINGLETON = SupportOpsEnvironment()
        return _ENV_SINGLETON
