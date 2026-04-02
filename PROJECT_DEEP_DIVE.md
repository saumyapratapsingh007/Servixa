# Servixa Deep Dive

This document explains the project from first principles.

## 1. What problem is this project solving?

Servixa tries to answer a practical question:

How do we measure whether an AI agent is actually good at customer support operations?

A lot of demos only show that a model can produce fluent text. That is not enough in real support systems.

A useful support agent must make the right operational choice:

- Is this a billing issue or a security issue?
- How urgent is it?
- Should frontline support keep it, or should a specialist team take it?
- Is the customer response appropriate?
- Is it safe to close the ticket?

Servixa turns those questions into a benchmark.

## 2. What kind of environment is this?

It is a structured decision-making environment built in Python and served through FastAPI.

The environment is OpenEnv-compatible, which means it follows a standard interaction model:

1. `reset()` starts a fresh task
2. `step(action)` applies one structured action
3. `state()` returns the current internal state

This lets external agents interact with the environment in a consistent way.

## 3. What makes this a real-world task instead of a toy?

The environment models customer support triage, which is a real job performed by humans every day.

The agent has to handle realistic cases like:

- password resets
- shipping delays
- duplicate billing
- abuse reports
- legal requests
- security incidents
- VIP outages

These are real operational workflows with real business consequences.

## 4. What are the main project pieces?

### `server/app.py`

This is the HTTP layer.

It exposes routes such as:

- `/reset`
- `/step`
- `/state`
- `/tasks`
- `/grader`
- `/health`

It also serves the landing page and status page.

### `env/models.py`

This defines the typed schema used throughout the project.

Important models:

- `SupportAction`
- `SupportObservation`
- `SupportState`
- `SupportReward`
- `TicketView`
- `TicketState`

These models are built with Pydantic, which gives validation and JSON schema generation.

### `env/tasks.py`

This contains the actual task definitions.

Each task describes:

- the task objective
- the difficulty
- the step budget
- the tickets in the queue
- the expected correct behavior for each ticket

### `env/environment.py`

This is the core logic engine.

It handles:

- resets
- action validation
- ticket mutation
- reward shaping
- done logic
- progress updates

### `env/grader.py`

This is the deterministic evaluator.

It compares what the agent actually did with what the task expected.

### `baseline.py`

This is the reproducible non-oracle policy used as a reference point.

### `inference.py`

This is the hackathon-compliant inference script that runs the environment with an OpenAI-compatible client and emits the required logs.

## 5. How does one episode work?

Here is the basic flow:

1. The agent calls `/reset`
2. The environment loads one task and creates ticket state
3. The agent receives an observation with visible ticket information
4. The agent sends a structured action through `/step`
5. The environment updates the ticket
6. The environment returns:
   - new observation
   - reward
   - done flag
   - metadata
7. This repeats until all tickets are minimally handled or the step limit is reached

## 6. What is hidden and what is visible?

Visible to the agent:

- ticket text
- customer metadata
- tags
- allowed templates
- visible notes
- current triage state

Hidden from the agent:

- expected category
- expected priority
- expected route
- expected template
- expected resolution
- must-close flag

This is important because it makes the environment evaluative rather than trivial.

## 7. How does reward shaping work?

Reward shaping happens during each `step()`.

For example:

- correct category gives positive reward
- wrong route gives negative reward
- choosing a forbidden template gives a penalty
- unsafe closure gives a large penalty
- every step gets a small efficiency penalty

This creates a learning signal across the whole trajectory rather than only at the end.

## 8. How does the final grader work?

The grader gives each ticket a score using weighted checks.

The dimensions are:

- category
- priority
- route
- template
- resolution
- closure

Each ticket score is clamped to `[0.0, 1.0]`.

Then:

- ticket scores are averaged
- an efficiency penalty is subtracted if too many steps were used

That gives the final episode score.

## 9. Why are the tasks ordered easy, medium, hard?

The progression is meant to test increasing levels of operational judgment.

Easy:

- straightforward tickets
- low ambiguity

Medium:

- mixed issue types
- specialist routing
- partial closure nuance

Hard:

- time pressure
- higher stakes
- safety-sensitive escalation
- mixed ticket ownership

## 10. How does the baseline work?

The baseline is mostly a deterministic heuristic policy.

It uses helper functions to choose:

- category, priority, and route
- response template
- resolution decision

It is not meant to be unbeatable.

It intentionally includes a small number of realistic misses so the benchmark still has room to separate stronger agents from weaker ones.

## 11. Why use FastAPI?

FastAPI is a good fit because:

- it makes HTTP endpoints simple
- it works well with Pydantic models
- it generates docs automatically
- it is easy to deploy in Docker and Hugging Face Spaces

## 12. Why use Pydantic?

Pydantic gives:

- type validation
- structured schemas
- cleaner API contracts
- easier serialization
- model-generated JSON schema for `/schema`

That is especially useful for benchmark environments because consistency matters a lot.

## 13. Why use OpenEnv?

OpenEnv gives a standard contract for agent environments.

That matters because judges and external agents can interact with your project in a predictable way.

## 14. How do all the parts connect?

Here is the mental model:

1. `tasks.py` defines the scenarios
2. `environment.py` runs those scenarios
3. `models.py` defines the typed data contract
4. `grader.py` scores the outcome
5. `server/app.py` exposes the environment over HTTP
6. `baseline.py` and `inference.py` act as client-side runners
7. `Dockerfile` and `openenv.yaml` make deployment reproducible

## 15. What are the most important design decisions?

### Deterministic grading

This makes evaluation reproducible.

### Shaped reward

This gives better signal than a pure pass/fail design.

### Specialist routing and unsafe closure penalties

These make the task feel operational rather than cosmetic.

### Strong but imperfect baseline

This makes the benchmark credible.

## 16. What are the current strengths?

- strong real-world relevance
- clear task progression
- interpretable reward design
- deterministic grading
- live deployment
- clean typed interfaces

## 17. What are the natural future improvements?

- stochastic ticket variants
- multi-turn customer follow-ups
- partial observability
- multiple baseline agents
- leaderboard comparison

## 18. If you had to explain Servixa in one sentence

Servixa is a customer support triage benchmark that evaluates whether an AI agent makes the right operational decisions, not just whether it can write plausible support text.

