# SupportOps OpenEnv

SupportOps OpenEnv is a production-ready customer support automation environment built for OpenEnv. It simulates a realistic support queue where an agent must classify customer issues, choose the correct escalation path, send the right customer response, and decide whether each ticket can be safely closed.

## Motivation

Customer support operations are one of the highest-leverage automation domains for AI agents. Real teams must balance speed, compliance, customer experience, escalation safety, and specialist routing. This environment captures those tradeoffs with deterministic grading and shaped rewards, making it useful for evaluating agentic reasoning beyond simple toy tasks.

## Environment Description

The environment models a support inbox containing multiple tickets. Each ticket includes customer context, business value, SLA risk, sentiment, prior contacts, and available response templates. The agent must operate over the queue with structured actions and receives a rich observation after every step.

## OpenEnv Interface

The environment implements:

- `reset(task_id=...) -> SupportObservation`
- `step(action: SupportAction) -> SupportObservation`
- `state -> SupportState`

API endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /baseline`
- `GET /grader`

## Action Space

`SupportAction` fields:

- `action_type`
- `ticket_id`
- `category`
- `priority`
- `route_to`
- `template_key`
- `resolution`
- `internal_note`
- `close_ticket`

## Observation Space

`SupportObservation` contains task metadata, a queue summary, visible ticket states, progress score, shaped reward details, and the latest event description.

## Reward Design

- Positive reward for correct category, priority, route, response template, and resolution
- Partial credit for incremental progress
- Penalties for invalid actions, unsafe closure, bad routing, and unavailable templates
- Step-wise efficiency penalty to discourage wasteful interaction

## Tasks

### Easy

- Password reset that should be resolved and closed
- Shipping delay that should stay open with logistics

### Medium

- Refund that can be completed
- Duplicate charge that must stay open with billing
- Abuse report that must be urgently escalated to trust and safety

### Hard

- Account compromise requiring urgent security escalation
- VIP outage requiring tech-ops escalation
- Legal data request requiring specialist review
- Refund escalation that can be completed and closed

## Deterministic Grading

Each ticket is scored on category, priority, route, template, resolution, closure safety, and efficiency. Final task scores range from `0.0` to `1.0`.

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

```bash
docker build -t supportops-openenv .
docker run -p 7860:7860 supportops-openenv
```

```bash
openenv validate
```

## Baseline

Run:

```bash
python baseline.py
```

The baseline uses a deterministic support policy and optionally calls the OpenAI API for a short task brief when `OPENAI_API_KEY` is available, so it remains reproducible and will not crash when the key is missing.
