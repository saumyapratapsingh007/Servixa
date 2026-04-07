from __future__ import annotations

from typing import Dict, List, Tuple

from .models import SupportState, TicketState


WEIGHTS: Dict[str, int] = {
    "category": 20,
    "priority": 15,
    "route": 20,
    "template": 15,
    "resolution": 20,
    "closure": 10,
}


def _ticket_breakdown(ticket: TicketState) -> Dict[str, int]:
    closure_score = 0
    if ticket.resolution is not None and ticket.closed == ticket.must_close:
        closure_score = WEIGHTS["closure"]

    breakdown = {
        "category": WEIGHTS["category"] if ticket.current_category == ticket.expected_category else 0,
        "priority": WEIGHTS["priority"] if ticket.current_priority == ticket.expected_priority else 0,
        "route": WEIGHTS["route"] if ticket.current_route == ticket.expected_route else 0,
        "template": WEIGHTS["template"] if ticket.last_response_template == ticket.expected_template else 0,
        "resolution": WEIGHTS["resolution"] if ticket.resolution == ticket.expected_resolution else 0,
        "closure": closure_score,
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
            "score": 0,
            "ticket_scores": [],
            "summary": {
                "tickets_completed": 0,
                "tickets_total": 0,
                "efficiency_penalty": 0,
                "base_score": 0,
                "step_count": state.step_count,
            },
        }

    ticket_scores: List[Dict[str, object]] = []
    raw_scores: List[int] = []

    for ticket in state.tickets:
        breakdown = _ticket_breakdown(ticket)
        raw_score = sum(breakdown.values()) 
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

    base_score = sum(raw_scores) // len(raw_scores)

    expected_steps = len(state.tickets) * 3
    overage = max(0, state.step_count - expected_steps)

    efficiency_penalty = min(18, overage * 2) 
    final_score = max(0, base_score - efficiency_penalty)

    tickets_completed = sum(1 for score in raw_scores if score >= 75)

    return {
        "task_id": state.task_id,
        "score": final_score,
        "ticket_scores": ticket_scores,
        "summary": {
            "tickets_completed": tickets_completed,
            "tickets_total": len(state.tickets),
            "efficiency_penalty": efficiency_penalty,
            "base_score": base_score,
            "step_count": state.step_count,
        },
    }


def grade_episode(state: SupportState) -> Tuple[float, Dict[str, object]]:
    report = grade_state(state)

    normalized_score = report["score"] / 100

    
    if normalized_score <= 0:
        normalized_score = 0.0001
    elif normalized_score >= 1:
        normalized_score = 0.9999

    return normalized_score, report
