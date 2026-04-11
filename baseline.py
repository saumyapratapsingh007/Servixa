from __future__ import annotations

import json
from typing import Any, Dict, List

from env.environment import SupportOpsEnvironment
from env.grader import grade_episode
from env.models import SupportAction
from env.tasks import TASKS, list_task_summaries


def _classify_policy(ticket: Dict[str, Any]) -> Dict[str, str]:
    tags = set(ticket["tags"])
    subject = ticket["subject"].lower()

    if "security" in tags or "account_compromise" in tags or "fraudulent" in subject:
        return {"category": "security", "priority": "urgent", "route_to": "security"}
    if "abuse_report" in tags or "safety" in tags:
        return {"category": "trust_safety", "priority": "urgent", "route_to": "trust_safety"}
    if "outage" in tags or "checkout" in tags:
        return {"category": "technical_outage", "priority": "urgent", "route_to": "tech_ops"}
    if "legal" in tags or "data_request" in tags:
        return {"category": "legal_request", "priority": "high", "route_to": "trust_safety"}
    if "shipping" in tags or "carrier_delay" in tags:
        return {"category": "shipping", "priority": "medium", "route_to": "logistics"}
    if "refund" in tags or "billing" in tags or "duplicate_charge" in tags or "duplicate_order" in tags:
        priority = "high" if "duplicate_charge" in tags or "escalation" in tags else "medium"
        return {"category": "billing", "priority": priority, "route_to": "billing"}
    if "login" in tags or "account" in tags:
        return {"category": "account_access", "priority": "high", "route_to": "frontline"}

    return {"category": "general_support", "priority": "medium", "route_to": "frontline"}


def _response_policy(ticket: Dict[str, Any]) -> str:
    tags = set(ticket["tags"])

    if "account_compromise" in tags or "security" in tags:
        return "security_lockdown_notice"
    if "outage" in tags:
        return "vip_outage_update"
    if "legal" in tags or "data_request" in tags:
        return "legal_request_acknowledgement"
    if "abuse_report" in tags or "safety" in tags:
        return "trust_safety_report_received"
    if "shipping" in tags or "carrier_delay" in tags:
        return "shipping_delay_empathy"
    if "duplicate_charge" in tags:
        return "duplicate_charge_escalation"
    if "refund" in tags or "duplicate_order" in tags or "billing" in tags:
        return "billing_refund_acknowledgement"

    return "password_reset_instructions"


def _resolution_policy(ticket: Dict[str, Any]) -> Dict[str, Any]:
    tags = set(ticket["tags"])

    if "account_compromise" in tags or "security" in tags:
        return {"resolution": "security_escalation_opened", "close_ticket": False}
    if "outage" in tags:
        return {"resolution": "incident_escalated", "close_ticket": False}
    if "legal" in tags or "data_request" in tags:
        return {"resolution": "legal_review_queued", "close_ticket": False}
    if "abuse_report" in tags or "safety" in tags:
        return {"resolution": "trust_safety_escalated", "close_ticket": False}
    if "shipping" in tags or "carrier_delay" in tags:
        return {"resolution": "awaiting_carrier_followup", "close_ticket": False}
    if "duplicate_charge" in tags:
        return {"resolution": "billing_investigation_opened", "close_ticket": False}
    if "refund" in tags or "duplicate_order" in tags or "escalation" in tags:
        return {"resolution": "refund_issued", "close_ticket": True}

    return {"resolution": "reset_link_sent", "close_ticket": True}


def run_baseline() -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    per_task_scores: List[float] = []

    for task in TASKS:
        env = SupportOpsEnvironment()
        observation = env.reset(task_id=str(task["id"]))

        for ticket in observation.tickets:
            data = ticket.model_dump()

            env.step(SupportAction(action_type="classify", ticket_id=ticket.ticket_id, **_classify_policy(data)))
            env.step(SupportAction(action_type="respond", ticket_id=ticket.ticket_id, template_key=_response_policy(data)))
            env.step(SupportAction(action_type="resolve", ticket_id=ticket.ticket_id, **_resolution_policy(data)))

        score, report = grade_episode(env.state)
        per_task_scores.append(score)

        results.append(
            {
                "task_id": task["id"],
                "difficulty": task["difficulty"],
                "score": score,
                "grader_report": report,
                "steps": env.state.step_count,
            }
        )

    avg = sum(per_task_scores) / len(per_task_scores)

    if avg <= 0.0:
        avg = 0.0001
    elif avg >= 1.0:
        avg = 0.9999

    return {
        "tasks": list_task_summaries(),
        "results": results,
        "average_score": round(avg, 4),
    }


if __name__ == "__main__":
    print(json.dumps(run_baseline(), indent=2))
