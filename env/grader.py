from __future__ import annotations

from typing import Dict, List, Tuple

from .models import SupportState, TicketState


WEIGHTS: Dict[str, float] = {
    "category": 0.20,
    "priority": 0.15,
    "route": 0.20,
    "template": 0.15,
    "resolution": 0.20,
    "closure": 0.10,
}


def _ticket_breakdown(ticket: TicketState) -> Dict[str, float]:
    breakdown = {
        "category": WEIGHTS["category"] if ticket.current_category == ticket.expected_category else 0.0,
        "priority": WEIGHTS["priority"] if ticket.current_priority == ticket.expected_priority else 0.0,
        "route": WEIGHTS["route"] if ticket.current_route == ticket.expected_route else 0.0,
        "template": WEIGHTS["template"] if ticket.last_response_template == ticket.expected_template else 0.0,
        "resolution": WEIGHTS["resolution"] if ticket.resolution == ticket.expected_resolution else 0.0,
        "closure": WEIGHTS["closure"] if ticket.closed == ticket.must_close else 0.0,
    }

    if ticket.unsafe_if_closed_early and ticket.closed and not (
        ticket.current_route == ticket.expected_route and ticket.resolution == ticket.expected_resolution
    ):
        breakdown["closure"] = -WEIGHTS["closure"]

    return breakdown


def grade_state(state: SupportState) -> Dict[str, object]:
    if not state.tickets:
        return {
            "task_id": state.task_id,
            "score": 0.0,
            "ticket_scores": [],
            "summary": {"tickets_completed": 0, "tickets_total": 0, "efficiency_penalty": 0.0},
        }

    ticket_scores: List[Dict[str, object]] = []
    raw_scores: List[float] = []
    for ticket in state.tickets:
        breakdown = _ticket_breakdown(ticket)
        raw_score = max(0.0, min(1.0, round(sum(breakdown.values()), 4)))
        raw_scores.append(raw_score)
        ticket_scores.append(
            {
                "ticket_id": ticket.ticket_id,
                "score": raw_score,
                "breakdown": breakdown,
                "closed": ticket.closed,
                "route": ticket.current_route,
                "resolution": ticket.resolution,
            }
        )

    base_score = sum(raw_scores) / len(raw_scores)
    expected_steps = len(state.tickets) * 3
    overage = max(0, state.step_count - expected_steps)
    efficiency_penalty = min(0.18, overage * 0.02)
    final_score = max(0.0, min(1.0, round(base_score - efficiency_penalty, 4)))

    tickets_completed = sum(1 for score in raw_scores if score >= 0.75)
    return {
        "task_id": state.task_id,
        "score": final_score,
        "ticket_scores": ticket_scores,
        "summary": {
            "tickets_completed": tickets_completed,
            "tickets_total": len(state.tickets),
            "efficiency_penalty": round(efficiency_penalty, 4),
            "base_score": round(base_score, 4),
            "step_count": state.step_count,
        },
    }


def grade_episode(state: SupportState) -> Tuple[float, Dict[str, object]]:
    report = grade_state(state)
    return float(report["score"]), report
