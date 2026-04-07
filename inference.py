from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from baseline import _baseline_adjustments, _classify_policy, _resolution_policy, _response_policy
from env.environment import SupportOpsEnvironment
from env.grader import grade_episode
from env.models import SupportAction, SupportObservation, TicketView
from env.tasks import TASKS
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "supportops_env"
TEMPERATURE = 0.0
MAX_TOKENS = 220

SUCCESS_SCORE_THRESHOLD = 90  # integer scale
MODEL_REVIEW_ENABLED = bool(HF_TOKEN)


SYSTEM_PROMPT = (
    "You are reviewing the next action for a customer support triage environment. "
    "Return exactly one JSON object with keys action_type, ticket_id, category, priority, "
    "route_to, template_key, resolution, internal_note, close_ticket. "
    "If the provided candidate action is already safe and sensible, return it unchanged."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: int, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: int, rewards: List[int]) -> None:
    rewards_str = ",".join(str(r) for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score} rewards={rewards_str}", flush=True)


def _ticket_to_dict(ticket: TicketView) -> Dict[str, Any]:
    return {
        "ticket_id": ticket.ticket_id,
        "subject": ticket.subject,
        "channel": ticket.channel,
        "customer_tier": ticket.customer_tier,
        "tags": ticket.tags,
        "current_status": ticket.current_status,
        "current_category": ticket.current_category,
        "current_priority": ticket.current_priority,
        "current_route": ticket.current_route,
        "last_response_template": ticket.last_response_template,
        "resolution": ticket.resolution,
        "closed": ticket.closed,
        "allowed_templates": ticket.allowed_templates,
        "visible_notes": ticket.visible_notes,
    }


def _baseline_action(task_id: str, ticket: TicketView) -> SupportAction:
    ticket_data = ticket.model_dump()
    adjustments = _baseline_adjustments(task_id, ticket_data)

    if ticket.current_category is None:
        classify_payload = _classify_policy(ticket_data)
        if "priority" in adjustments:
            classify_payload["priority"] = adjustments["priority"]
        return SupportAction(
            action_type="classify",
            ticket_id=ticket.ticket_id,
            internal_note="Inference baseline classification.",
            **classify_payload,
        )

    if ticket.last_response_template is None:
        template_key = str(adjustments.get("template_key", _response_policy(ticket_data)))
        return SupportAction(
            action_type="respond",
            ticket_id=ticket.ticket_id,
            template_key=template_key,
            internal_note="Inference baseline response.",
        )

    return SupportAction(
        action_type="resolve",
        ticket_id=ticket.ticket_id,
        internal_note="Inference baseline resolution.",
        **_resolution_policy(ticket_data),
    )


def _next_heuristic_action(observation: SupportObservation) -> SupportAction:
    for ticket in observation.tickets:
        if ticket.resolution is None:
            return _baseline_action(observation.task_id, ticket)

    return SupportAction(
        action_type="classify",
        ticket_id=observation.tickets[0].ticket_id,
        category=observation.tickets[0].current_category or "general_support",
        priority=observation.tickets[0].current_priority or "medium",
        route_to=observation.tickets[0].current_route or "frontline",
        internal_note="Fallback no-op.",
    )


def _safe_action_json(action: SupportAction) -> Dict[str, Any]:
    return {
        "action_type": action.action_type,
        "ticket_id": action.ticket_id,
        "category": action.category,
        "priority": action.priority,
        "route_to": action.route_to,
        "template_key": action.template_key,
        "resolution": action.resolution,
        "internal_note": action.internal_note,
        "close_ticket": action.close_ticket,
    }


def _parse_model_action(payload: str) -> Optional[SupportAction]:
    try:
        data = json.loads(payload)
        return SupportAction(**data)
    except Exception:
        return None


def _request_model_action(client: Optional[OpenAI], observation: SupportObservation, candidate: SupportAction) -> SupportAction:
    global MODEL_REVIEW_ENABLED
    if client is None or not MODEL_REVIEW_ENABLED:
        return candidate

    tickets = [_ticket_to_dict(ticket) for ticket in observation.tickets]

    user_prompt = json.dumps(
        {
            "task_id": observation.task_id,
            "objective": observation.objective,
            "tickets": tickets,
            "candidate_action": _safe_action_json(candidate),
        }
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        content = (response.choices[0].message.content or "").strip()
        parsed = _parse_model_action(content)
        return parsed or candidate

    except Exception:
        MODEL_REVIEW_ENABLED = False
        return candidate


def _action_str(action: SupportAction) -> str:
    return json.dumps(
        {k: v for k, v in _safe_action_json(action).items() if v not in (None, "", False)},
        separators=(",", ":"),
    )


def run_task(client: Optional[OpenAI], task_id: str) -> Dict[str, Any]:
    env = SupportOpsEnvironment()
    observation = env.reset(task_id=task_id)

    rewards: List[int] = []
    steps_taken = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, observation.queue_summary.max_steps + 1):
            if observation.done:
                break

            action = _request_model_action(client, observation, _next_heuristic_action(observation))
            observation = env.step(action)

            reward = int(observation.reward or 0)
            rewards.append(reward)
            steps_taken = step

            log_step(step, _action_str(action), reward, observation.done, None)

            if observation.done:
                break

    finally:
        score, report = grade_episode(env.state)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success, steps_taken, score, rewards)

    return {"task_id": task_id, "score": score, "report": report}


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=12.0) if MODEL_REVIEW_ENABLED else None

    for task in TASKS:
        run_task(client, str(task["id"]))


if __name__ == "__main__":
    main()
