# Interview Q&A

This document is designed to help you answer project questions clearly and confidently.

## 1. What is Servixa?

Servixa is an OpenEnv-compatible benchmark for customer support triage. It evaluates whether an AI agent can make correct operational support decisions such as classifying issues, prioritizing them, routing them safely, choosing the right response, and deciding whether a ticket can be closed.

## 2. Why did you choose customer support as the domain?

I chose customer support because it is a real business workflow where automation is valuable but risky. Many companies want agents to help with support operations, but the important part is not just generating text. The important part is making the right operational decision. That makes support triage a strong benchmark domain.

## 3. What makes this project real-world instead of a toy?

It models tasks that real human support teams perform every day. The environment includes realistic tickets such as password resets, shipping delays, billing disputes, abuse reports, legal requests, security incidents, and VIP outages. Those are genuine operational workflows, not abstract games.

## 4. What is OpenEnv and why did you use it?

OpenEnv is a standard interface for agent environments. I used it so the environment has a clear reset-step-state contract, typed schemas, validation support, and a format that judges or external agents can interact with consistently.

## 5. What are the main components of the system?

There are four main parts:

1. typed models in `env/models.py`
2. task definitions in `env/tasks.py`
3. the environment engine in `env/environment.py`
4. the deterministic grader in `env/grader.py`

Then `server/app.py` exposes everything over FastAPI.

## 6. How does the environment work at a high level?

The agent resets into a task, receives an observation containing the ticket queue, and then acts one step at a time using structured actions. Each step updates the ticket state, returns a shaped reward, and eventually the grader computes a final score.

## 7. What actions can the agent take?

The agent can:

- classify a ticket
- respond using a template
- resolve the ticket and optionally close it

Those actions are intentionally simple and structured so the benchmark focuses on judgment, not prompt wording tricks.

## 8. Why did you use structured actions instead of free-form text?

Structured actions make the environment easier to evaluate fairly and deterministically. If the action space were purely free-form text, grading would become ambiguous. With structured actions, we can score the actual decision quality directly.

## 9. What is the observation space?

The observation includes task metadata, queue summary, ticket views, reward details, hints, progress score, and step-level metadata. The ticket views expose enough context for decision-making without exposing the hidden ground-truth answer.

## 10. What is hidden from the agent?

The expected correct category, priority, route, template, and resolution are hidden. Those are only used by the grader.

## 11. How does reward shaping work?

Reward shaping gives partial credit during the episode. The environment gives positive signal for useful progress like correct classification or correct routing and negative signal for bad choices like invalid actions, wrong specialist routing, forbidden templates, unsafe closure, and inefficient extra steps.

## 12. Why not just use a final binary score?

A pure binary score would be too sparse and less informative. Reward shaping provides signal across the full trajectory, which is better for debugging and for learning-oriented agent evaluation.

## 13. How does the grader work?

Each ticket is scored across six dimensions: category, priority, route, template, resolution, and closure. Those are combined into a ticket score from `0.0` to `1.0`, and the final episode score is the average ticket score minus an efficiency penalty if the agent used too many steps.

## 14. Why is the grader deterministic?

Because reproducibility matters. If the grader changes from run to run, it becomes harder to compare agents fairly. Deterministic grading makes the benchmark stable and easier to trust.

## 15. What are the three tasks and why are they different?

The easy task checks basic support handling. The medium task adds policy nuance and specialist routing. The hard task adds high-risk, high-pressure work like security incidents, legal requests, and VIP outages.

## 16. What makes the hard task genuinely difficult?

The hard task mixes urgency, safety, and ownership. The agent has to avoid dangerous mistakes such as closing a security issue early or failing to escalate a VIP outage correctly. It is difficult because it combines several kinds of operational reasoning at once.

## 17. Why is the baseline not perfect?

A perfect baseline would make the benchmark less informative. I wanted a strong baseline that proves the environment is solvable, but still leaves room for stronger policies to improve.

## 18. What technologies did you use and why?

I used:

- FastAPI for serving the environment over HTTP
- Pydantic for typed schemas and validation
- OpenEnv for the environment contract
- Uvicorn as the ASGI server
- Docker for reproducible deployment
- Hugging Face Spaces for hosting
- the OpenAI-compatible client for the required inference script

## 19. Why FastAPI specifically?

FastAPI works very naturally with Pydantic, makes endpoint creation simple, and gives clean HTTP behavior for environment-style APIs.

## 20. Why Pydantic specifically?

Pydantic makes the action, observation, and state schemas explicit and validated. That improves reliability and helps generate schemas automatically.

## 21. How does deployment work?

The Dockerfile installs dependencies, copies the project, exposes port `7860`, and starts Uvicorn with `server.app:app`. Hugging Face Spaces uses that container setup to host the environment.

## 22. What role does `openenv.yaml` play?

`openenv.yaml` tells OpenEnv how to interpret the project. It provides metadata such as the app entrypoint, runtime, and port.

## 23. What are the main strengths of your project?

- strong real-world relevance
- deterministic grading
- clear task progression
- meaningful reward shaping
- reproducible baseline
- live deployment

## 24. What are the main limitations?

The current environment is single-turn per ticket and deterministic. It does not yet model longer customer conversations, uncertainty, or stochastic variants. Those would be natural next steps.

## 25. How would you improve the project further?

I would add multi-turn conversations, stochastic variants, more baseline agents, comparative evaluation dashboards, and richer specialist workflows.

## 26. How do you defend the novelty of the project?

The novelty is in turning support operations into a structured benchmark for agent judgment. Many demos focus on reply generation. This project focuses on operational correctness, safety, and routing quality.

## 27. How do you know the environment is not trivial to exploit?

The benchmark is structured around hidden expected outcomes, closure safety, and deterministic grading across multiple dimensions. The agent does not see the answers directly, and a single superficial heuristic does not automatically solve every case.

## 28. Why did you include a landing page?

The landing page makes the project easier to understand quickly for judges and reviewers. It does not replace the benchmark, but it improves the first impression and helps explain the system visually.

## 29. What part of the project are you most proud of?

The strongest part is that it evaluates the kind of decisions real support teams actually care about. It is practical, measurable, and still simple enough for judges to understand quickly.

## 30. If an interviewer asks for the one-line summary

Servixa is a customer support triage benchmark that measures whether an AI agent makes correct support decisions under realistic operational constraints.

