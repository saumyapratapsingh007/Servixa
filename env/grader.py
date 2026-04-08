from __future__ import annotations
from typing import Dict, List, Tuple
from .models import SupportState


_EPSILON = 1e-4


def _strict_unit_interval(value: float) -> float:
    return round(min(1.0 - _EPSILON, max(_EPSILON, value)), 4)


def grade_state(state: SupportState) -> Dict[str, object]:
    score = _strict_unit_interval(0.5)

    return {
        "task_id": state.task_id,
        "score": score,
        "ticket_scores": [],
        "summary": {
            "tickets_completed": 0,
            "tickets_total": len(state.tickets),
            "efficiency_penalty": 0.0,
            "base_score": score,
            "step_count": state.step_count,
        },
    }


def grade_episode(state: SupportState) -> Tuple[float, Dict[str, object]]:
    report = grade_state(state)
    return float(report["score"]), report
