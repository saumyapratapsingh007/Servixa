from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from baseline import run_baseline
from env.environment import get_environment
from env.grader import grade_state
from env.models import SupportAction, SupportObservation, SupportState
from env.tasks import list_task_summaries
from openenv.core.env_server.types import EnvironmentMetadata, HealthResponse


class ResetPayload(BaseModel):
    seed: Optional[int] = Field(default=None)
    episode_id: Optional[str] = Field(default=None)
    task_id: Optional[str] = Field(default=None)


class StepPayload(BaseModel):
    action: SupportAction
    timeout_s: Optional[float] = Field(default=None)


class SchemaPayload(BaseModel):
    action: Dict[str, Any]
    observation: Dict[str, Any]
    state: Dict[str, Any]


app = FastAPI(
    title="SupportOps OpenEnv API",
    version="1.0.0",
    description="A deterministic customer support triage environment that implements the OpenEnv HTTP contract.",
)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Servixa</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 32px;
            background: #f6f9fc;
            color: #1f2937;
          }
          .card {
            max-width: 760px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
          }
          h1 {
            margin-top: 0;
          }
          code {
            background: #eef2ff;
            padding: 2px 6px;
            border-radius: 6px;
          }
          ul {
            line-height: 1.7;
          }
          a {
            color: #2563eb;
            text-decoration: none;
          }
          a:hover {
            text-decoration: underline;
          }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Servixa</h1>
          <p>
            Servixa is a structured customer support automation environment for evaluating
            agent decisions around triage, routing, escalation, response selection, and ticket closure.
          </p>
          <p>Useful endpoints:</p>
          <ul>
            <li><a href="/docs"><code>/docs</code></a> - interactive API docs</li>
            <li><a href="/health"><code>/health</code></a> - health check</li>
            <li><a href="/tasks"><code>/tasks</code></a> - available tasks</li>
            <li><a href="/baseline"><code>/baseline</code></a> - baseline performance</li>
            <li><a href="/grader"><code>/grader</code></a> - current grader report</li>
            <li><a href="/metadata"><code>/metadata</code></a> - environment metadata</li>
            <li><a href="/schema"><code>/schema</code></a> - action, observation, and state schemas</li>
          </ul>
          <p>
            Typical agent flow: <code>POST /reset</code> -> <code>POST /step</code> -> <code>GET /state</code>.
          </p>
        </div>
      </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get("/metadata", response_model=EnvironmentMetadata)
def metadata() -> EnvironmentMetadata:
    return get_environment().get_metadata()


@app.get("/schema", response_model=SchemaPayload)
def schema() -> SchemaPayload:
    return SchemaPayload(
        action=SupportAction.model_json_schema(),
        observation=SupportObservation.model_json_schema(),
        state=SupportState.model_json_schema(),
    )


@app.post("/reset", response_model=SupportObservation)
def reset(payload: ResetPayload) -> SupportObservation:
    return get_environment().reset(
        seed=payload.seed,
        episode_id=payload.episode_id,
        task_id=payload.task_id,
    )


@app.post("/step", response_model=SupportObservation)
def step(payload: StepPayload) -> SupportObservation:
    return get_environment().step(payload.action, timeout_s=payload.timeout_s)


@app.get("/state", response_model=SupportState)
def state() -> SupportState:
    return get_environment().state


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": list_task_summaries()}


@app.get("/baseline")
def baseline() -> dict:
    return run_baseline()


@app.get("/grader")
def grader() -> dict:
    return grade_state(get_environment().state)


@app.post("/mcp")
def mcp(body: Dict[str, Any]) -> Dict[str, Any]:
    request_id = body.get("id")
    method = body.get("method")
    if method == "tools/list":
        result: Dict[str, Any] = {"tools": []}
    else:
        result = {
            "server": "supportops_env",
            "status": "ok",
            "note": "Basic MCP compatibility endpoint for OpenEnv validation.",
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
