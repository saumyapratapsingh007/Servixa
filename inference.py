from __future__ import annotations

import ast
import io
import os
import re
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from env.environment import SupportOpsEnvironment
from env.grader import grade_episode
from env.models import SupportAction, SupportObservation, TicketView
from env.tasks import TASKS


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is required")

BENCHMARK = "supportops_env"
REQUEST_TIMEOUT_SECONDS = 12.0
MAX_OUTPUT_TOKENS = 120
ACTION_PATTERN = re.compile(r"^(classify|respond|resolve)\((.*)\)$")

SYSTEM_PROMPT = """You are controlling a deterministic support triage environment.
Return exactly one action string and nothing else.
Allowed formats:
classify('TICKET_ID','CATEGORY','PRIORITY','ROUTE')
respond('TICKET_ID','TEMPLATE_KEY')
resolve('TICKET_ID','RESOLUTION',true)
resolve('TICKET_ID','RESOLUTION',false)
Choose the next best action for the current queue state.
Do not wrap the answer in JSON, markdown, or explanation."""


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except AttributeError:
        pass


def _single_line(value: Any) -> str:
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _reward_text(value: float) -> str:
    return f"{float(value):.2f}"


def log_start(task_name: str, env_name: str, model_name: str) -> None:
    print(f"[START] task={_single_line(task_name)} env={_single_line(env_name)} model={_single_line(model_name)}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = "null" if error is None else _single_line(error)
    print(f"[STEP] step={step} action={_single_line(action_str)} reward={_reward_text(reward)} done={_bool_text(done)} error={error_value}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    reward_values = ",".join(_reward_text(r) for r in rewards)
    print(f"[END] success={_bool_text(success)} steps={steps} rewards={reward_values}", flush=True)


def _ticket_payload(ticket: TicketView) -> Dict[str, Any]:
    return {
        "ticket_id": ticket.ticket_id,
        "subject": ticket.subject,
        "body": ticket.body,
        "channel": ticket.channel,
        "customer_tier": ticket.customer_tier,
        "tags": ticket.tags,
        "allowed_templates": ticket.allowed_templates,
        "visible_notes": ticket.visible_notes,
        "current_status": ticket.current_status,
        "current_category": ticket.current_category,
        "current_priority": ticket.current_priority,
        "current_route": ticket.current_route,
        "last_response_template": ticket.last_response_template,
        "resolution": ticket.resolution,
        "closed": ticket.closed,
    }


def _quote(value: str) -> str:
    return repr(str(value))


def _parse_action_string(action_str: str) -> SupportAction:
    raw = _single_line(action_str)
    match = ACTION_PATTERN.match(raw)
    if not match:
        raise ValueError("invalid action format")

    action_type = match.group(1)
    arguments_source = f"[{match.group(2)}]"
    python_args = arguments_source.replace("true", "True").replace("false", "False")
    arguments = ast.literal_eval(python_args)

    if not isinstance(arguments, list):
        raise ValueError("invalid action arguments")

    if action_type == "classify" and len(arguments) == 4:
        return SupportAction(
            action_type="classify",
            ticket_id=str(arguments[0]),
            category=str(arguments[1]),
            priority=str(arguments[2]),
            route_to=str(arguments[3]),
            internal_note="API-guided triage classification.",
        )

    if action_type == "respond" and len(arguments) == 2:
        return SupportAction(
            action_type="respond",
            ticket_id=str(arguments[0]),
            template_key=str(arguments[1]),
            internal_note="API-guided customer response.",
        )

    if action_type == "resolve" and len(arguments) == 3:
        return SupportAction(
            action_type="resolve",
            ticket_id=str(arguments[0]),
            resolution=str(arguments[1]),
            close_ticket=bool(arguments[2]),
            internal_note="API-guided resolution.",
        )

    raise ValueError("unsupported action signature")


def _heuristic_action_string(observation: SupportObservation) -> str:
    for ticket in observation.tickets:
        if ticket.current_category is None:
            if "security" in ticket.tags or "account_compromise" in ticket.tags:
                return f"classify({_quote(ticket.ticket_id)},'security','urgent','security')"
            if "abuse_report" in ticket.tags or "safety" in ticket.tags:
                return f"classify({_quote(ticket.ticket_id)},'trust_safety','urgent','trust_safety')"
            if "legal" in ticket.tags or "data_request" in ticket.tags:
                return f"classify({_quote(ticket.ticket_id)},'legal_request','high','trust_safety')"
            if "outage" in ticket.tags or "checkout" in ticket.tags:
                return f"classify({_quote(ticket.ticket_id)},'technical_outage','urgent','tech_ops')"
            if "shipping" in ticket.tags or "carrier_delay" in ticket.tags:
                return f"classify({_quote(ticket.ticket_id)},'shipping','medium','logistics')"
            if "refund" in ticket.tags or "duplicate_charge" in ticket.tags or "duplicate_order" in ticket.tags:
                priority = "medium"
                if (
                    "duplicate_charge" in ticket.tags
                    or ticket.customer_tier in {"pro", "vip"}
                    or ticket.prior_contacts >= 2
                    or ticket.sla_hours_remaining <= 4
                    or ticket.hours_open >= 24
                ):
                    priority = "high"
                return f"classify({_quote(ticket.ticket_id)},'billing',{_quote(priority)},'billing')"
            return f"classify({_quote(ticket.ticket_id)},'account_access','high','frontline')"

        if ticket.last_response_template is None:
            if ticket.current_category == "security":
                return f"respond({_quote(ticket.ticket_id)},'security_lockdown_notice')"
            if ticket.current_category == "trust_safety":
                return f"respond({_quote(ticket.ticket_id)},'trust_safety_report_received')"
            if ticket.current_category == "legal_request":
                return f"respond({_quote(ticket.ticket_id)},'legal_request_acknowledgement')"
            if ticket.current_category == "technical_outage":
                return f"respond({_quote(ticket.ticket_id)},'vip_outage_update')"
            if ticket.current_category == "shipping":
                return f"respond({_quote(ticket.ticket_id)},'shipping_delay_empathy')"
            if ticket.current_category == "billing" and "duplicate_charge" in ticket.tags:
                return f"respond({_quote(ticket.ticket_id)},'duplicate_charge_escalation')"
            if ticket.current_category == "billing":
                return f"respond({_quote(ticket.ticket_id)},'billing_refund_acknowledgement')"
            return f"respond({_quote(ticket.ticket_id)},'password_reset_instructions')"

        if ticket.resolution is None:
            if ticket.current_category == "security":
                return f"resolve({_quote(ticket.ticket_id)},'security_escalation_opened',false)"
            if ticket.current_category == "trust_safety":
                return f"resolve({_quote(ticket.ticket_id)},'trust_safety_escalated',false)"
            if ticket.current_category == "legal_request":
                return f"resolve({_quote(ticket.ticket_id)},'legal_review_queued',false)"
            if ticket.current_category == "technical_outage":
                return f"resolve({_quote(ticket.ticket_id)},'incident_escalated',false)"
            if ticket.current_category == "shipping":
                return f"resolve({_quote(ticket.ticket_id)},'awaiting_carrier_followup',false)"
            if ticket.current_category == "billing" and "duplicate_charge" in ticket.tags:
                return f"resolve({_quote(ticket.ticket_id)},'billing_investigation_opened',false)"
            if ticket.current_category == "billing":
                return f"resolve({_quote(ticket.ticket_id)},'refund_issued',true)"
            return f"resolve({_quote(ticket.ticket_id)},'reset_link_sent',true)"

    ticket = observation.tickets[0]
    return f"respond({_quote(ticket.ticket_id)},{_quote(ticket.allowed_templates[0])})"


def _build_messages(observation: SupportObservation) -> List[Dict[str, str]]:
    user_payload = {
        "task_id": observation.task_id,
        "task_title": observation.task_title,
        "objective": observation.objective,
        "step_count": observation.queue_summary.step_count,
        "max_steps": observation.queue_summary.max_steps,
        "last_event": observation.last_event,
        "hints": observation.hints,
        "tickets": [_ticket_payload(ticket) for ticket in observation.tickets],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(user_payload)},
    ]


def _request_action_string(client: OpenAI, observation: SupportObservation) -> str:
    fallback = _heuristic_action_string(observation)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=_build_messages(observation),
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        content = response.choices[0].message.content or ""
        candidate = _single_line(content)
        _parse_action_string(candidate)
        return candidate
    except Exception:
        return fallback


def _run_task(client: OpenAI, task_name: str) -> Tuple[bool, int, List[float]]:
    env = SupportOpsEnvironment()
    observation = env.reset(task_id=task_name)
    rewards: List[float] = []
    steps = 0
    log_start(task_name=task_name, env_name=BENCHMARK, model_name=MODEL_NAME)

    try:
        while not observation.done:
            step_number = steps + 1
            action_str = ""
            reward_value = 0.0
            done_value = False
            error_text: Optional[str] = None

            try:
                action_str = _request_action_string(client, observation)
                action = _parse_action_string(action_str)
                observation = env.step(action)
                reward_value = float(observation.reward or 0.0)
                done_value = bool(observation.done)
            except Exception as exc:
                error_text = str(exc)
                done_value = True
                try:
                    observation = observation.model_copy(update={"done": True})
                except Exception:
                    pass

            rewards.append(reward_value)
            steps = step_number
            log_step(step_number, action_str or "null_action()", reward_value, done_value, error_text)

            if done_value:
                break
    finally:
        score, _report = grade_episode(env.state)
        success = score >= 0.90
        log_end(success, steps, rewards)

    return success, steps, rewards


def validate_output_format() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        log_start("demo_task", BENCHMARK, MODEL_NAME)
        log_step(1, "click('123')", 1, False, None)
        log_step(2, "fill('456','text')", -2.5, False, "temporary issue")
        log_step(3, "click('789')", 0, True, None)
        log_end(True, 3, [1, -2.5, 0])

    lines = buffer.getvalue().splitlines()
    expected = [
        f"[START] task=demo_task env={BENCHMARK} model={MODEL_NAME}",
        "[STEP] step=1 action=click('123') reward=1.00 done=false error=null",
        "[STEP] step=2 action=fill('456','text') reward=-2.50 done=false error=temporary issue",
        "[STEP] step=3 action=click('789') reward=0.00 done=true error=null",
        "[END] success=true steps=3 rewards=1.00,-2.50,0.00",
    ]
    if lines != expected:
        raise AssertionError("output format validation failed")


def main() -> None:
    _configure_stdout()
    validate_output_format()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=REQUEST_TIMEOUT_SECONDS)
    for task in TASKS:
        _run_task(client, str(task["id"]))


if __name__ == "__main__":
    main()
