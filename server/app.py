from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
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

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@app.get("/assets/{filename}")
def asset_file(filename: str) -> FileResponse:
    path = (ASSETS_DIR / filename).resolve()
    if not str(path).startswith(str(ASSETS_DIR.resolve())) or not path.exists():
        raise FileNotFoundError(filename)
    return FileResponse(path)


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
          :root { --bg:#07111f; --bg2:#0d1830; --panel:rgba(10,19,33,.76); --line:rgba(148,163,184,.16); --text:#e5eefc; --muted:#9db0cc; --cyan:#67e8f9; --violet:#8b5cf6; --green:#34d399; --shadow:0 24px 60px rgba(2,8,23,.45); --r:24px; --t:260ms ease; }
          * { box-sizing:border-box; }
          html { scroll-behavior:smooth; }
          body { margin:0; color:var(--text); font-family:"Segoe UI",Arial,sans-serif; background:linear-gradient(180deg, rgba(5,8,15,.96) 0%, rgba(7,10,18,.94) 34%, rgba(10,16,28,.96) 100%), radial-gradient(circle at top left, rgba(103,232,249,.08), transparent 26%), radial-gradient(circle at top right, rgba(139,92,246,.10), transparent 24%), linear-gradient(180deg, rgba(194,224,255,.06) 0%, rgba(119,170,221,.03) 22%, rgba(8,19,33,0) 50%); overflow-x:hidden; cursor:none; }
          body::before, body::after { content:""; position:fixed; width:24rem; height:24rem; border-radius:50%; filter:blur(90px); opacity:.34; z-index:0; pointer-events:none; animation:floatGlow 14s ease-in-out infinite; }
          body::before { top:-10rem; right:-8rem; background:rgba(103,232,249,.26); }
          body::after { left:-8rem; bottom:-12rem; background:rgba(139,92,246,.22); animation-delay:-4s; }
          body::marker { content:""; }
          a, button, .button, .nav-link, .nav-trigger, .endpoint { cursor:none; }
          a { color:inherit; text-decoration:none; }
          button { font:inherit; }
          .static-logos, .floating-scaler { position:fixed; pointer-events:none; user-select:none; }
          .static-logos { left:50%; top:12vh; z-index:0; width:min(94vw, 1260px); transform:translateX(-50%); opacity:.56; filter:drop-shadow(0 18px 42px rgba(2,8,23,.34)); }
          .floating-scaler { left:50%; top:4vh; z-index:12; width:min(96vw, 1540px); max-width:none; opacity:.92; transform:translate3d(-50%, 0, 0); filter:drop-shadow(0 24px 60px rgba(2,8,23,.42)); transition:opacity 180ms ease, transform 180ms ease; }
          .page { position:relative; z-index:1; width:min(1180px, calc(100% - 32px)); margin:0 auto; padding:20px 0 56px; }
          .page::before { content:""; position:fixed; inset:0; z-index:0; pointer-events:none; background:
            radial-gradient(circle at var(--cursor-x, 50%) var(--cursor-y, 24%), rgba(255,244,232,.22) 0%, rgba(255,236,214,.12) 10%, rgba(103,232,249,.12) 20%, transparent 42%);
            opacity:.9;
            transition:background-position 80ms linear; }
          .cursor-shell, .cursor-core { position:fixed; left:0; top:0; pointer-events:none; z-index:60; border-radius:999px; transform:translate3d(-50%, -50%, 0); transition:width 160ms ease, height 160ms ease, opacity 160ms ease, background 160ms ease, border-color 160ms ease, box-shadow 160ms ease; will-change:transform,width,height,opacity; }
          .cursor-shell { width:34px; height:34px; border:1px solid rgba(255,244,232,.45); background:radial-gradient(circle, rgba(255,255,255,.06), rgba(103,232,249,.04)); box-shadow:0 0 0 1px rgba(103,232,249,.08), 0 0 24px rgba(103,232,249,.12); backdrop-filter:blur(6px); opacity:.92; }
          .cursor-core { width:10px; height:10px; background:linear-gradient(135deg, rgba(255,255,255,.92), rgba(103,232,249,.92)); box-shadow:0 0 20px rgba(103,232,249,.45); }
          .cursor-shell.is-idle { width:42px; height:42px; border-color:rgba(255,244,232,.26); background:radial-gradient(circle, rgba(255,255,255,.03), rgba(103,232,249,.02)); opacity:.72; }
          .cursor-core.is-idle { width:8px; height:8px; background:linear-gradient(135deg, rgba(255,255,255,.72), rgba(103,232,249,.66)); box-shadow:0 0 14px rgba(103,232,249,.24); }
          .cursor-shell.is-pressed { width:24px; height:24px; border-color:rgba(255,255,255,.72); background:radial-gradient(circle, rgba(255,255,255,.18), rgba(103,232,249,.16)); box-shadow:0 0 28px rgba(255,255,255,.16), 0 0 34px rgba(103,232,249,.34); }
          .cursor-core.is-pressed { width:14px; height:14px; background:linear-gradient(135deg, rgba(255,255,255,.98), rgba(255,244,232,.98)); box-shadow:0 0 24px rgba(255,255,255,.42); }
          .nav, .panel { background:rgba(10,19,33,.34); border:1px solid rgba(229,238,252,.12); box-shadow:var(--shadow); backdrop-filter:blur(20px); }
          .nav { position:sticky; top:18px; z-index:20; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; margin-bottom:28px; border-radius:999px; }
          .brand { display:flex; align-items:center; gap:12px; }
          .brand-mark { width:44px; height:44px; display:grid; place-items:center; border-radius:14px; color:var(--cyan); font-weight:800; background:linear-gradient(135deg, rgba(103,232,249,.22), rgba(139,92,246,.28)); border:1px solid rgba(103,232,249,.24); box-shadow:0 0 24px rgba(103,232,249,.2); }
          .brand strong { display:block; letter-spacing:.04em; }
          .brand span { display:block; color:var(--muted); font-size:.82rem; }
          .nav-links { display:flex; align-items:center; gap:10px; }
          .nav-group { position:relative; }
          .nav-link, .nav-trigger { display:inline-flex; align-items:center; gap:8px; padding:12px 16px; border-radius:999px; background:transparent; border:1px solid transparent; color:#dbe7fb; cursor:pointer; transition:background var(--t), border-color var(--t), transform var(--t), box-shadow var(--t); }
          .nav-trigger::after { content:"v"; font-size:.7rem; color:var(--muted); }
          .nav-link:hover, .nav-link:focus-visible, .nav-trigger:hover, .nav-trigger:focus-visible, .nav-group:hover>.nav-trigger, .nav-group:focus-within>.nav-trigger { background:rgba(148,163,184,.08); border-color:rgba(103,232,249,.18); transform:translateY(-1px); box-shadow:0 10px 25px rgba(2,8,23,.22); outline:none; }
          .dropdown { position:absolute; top:calc(100% + 12px); left:0; width:290px; padding:10px; border-radius:20px; background:rgba(8,15,28,.92); border:1px solid rgba(103,232,249,.14); box-shadow:var(--shadow); opacity:0; visibility:hidden; transform:translateY(8px); transition:opacity var(--t), transform var(--t), visibility var(--t); }
          .nav-group:hover .dropdown, .nav-group:focus-within .dropdown, .nav-group.is-open .dropdown { opacity:1; visibility:visible; transform:translateY(0); }
          .dropdown a { display:block; padding:12px 14px; border-radius:16px; transition:background var(--t), transform var(--t); }
          .dropdown a:hover, .dropdown a:focus-visible { background:rgba(103,232,249,.09); transform:translateX(3px); outline:none; }
          .dropdown strong { display:block; margin-bottom:4px; font-size:.95rem; }
          .dropdown span { display:block; color:var(--muted); font-size:.84rem; line-height:1.45; }
          .menu-button { display:none; width:46px; height:46px; align-items:center; justify-content:center; border:1px solid rgba(103,232,249,.18); border-radius:16px; background:rgba(148,163,184,.08); color:var(--text); cursor:pointer; }
          .hero { position:relative; z-index:2; display:grid; grid-template-columns:minmax(0,1.18fr) minmax(320px,.82fr); gap:24px; margin-bottom:24px; min-height:720px; align-items:end; padding-top:120px; }
          .panel { position:relative; overflow:hidden; border-radius:var(--r); }
          .panel::before { content:""; position:absolute; inset:0; pointer-events:none; background:linear-gradient(120deg, rgba(103,232,249,.08), transparent 34%, rgba(139,92,246,.08)); }
          .hero-copy, .hero-side, .card, .task, .stack { position:relative; padding:28px; }
          .eyebrow, .label, .task-badge { display:inline-flex; align-items:center; padding:8px 12px; border-radius:999px; border:1px solid rgba(103,232,249,.14); background:rgba(103,232,249,.08); color:#c9fbff; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }
          .hero h1 { margin:16px 0 14px; font-size:clamp(2.7rem, 6vw, 4.9rem); line-height:.96; letter-spacing:-.05em; }
          .hero h1 span { color:var(--cyan); text-shadow:0 0 26px rgba(103,232,249,.34); }
          .hero p, .muted { margin:0; color:#cfdcf1; line-height:1.7; }
          .muted { color:var(--muted); }
          .cta-row { display:flex; flex-wrap:wrap; gap:14px; margin:24px 0; }
          .button { display:inline-flex; align-items:center; justify-content:center; min-height:50px; padding:0 22px; border-radius:16px; border:1px solid transparent; transition:transform var(--t), box-shadow var(--t), border-color var(--t), background var(--t); }
          .button:hover, .button:focus-visible { transform:translateY(-2px) scale(1.01); outline:none; }
          .button-primary { background:linear-gradient(135deg, rgba(103,232,249,.95), rgba(59,130,246,.86)); color:#04101d; font-weight:700; box-shadow:0 18px 36px rgba(37,99,235,.28); }
          .button-secondary { background:rgba(148,163,184,.06); border-color:rgba(148,163,184,.18); }
          .metrics, .cards, .tasks { display:grid; gap:16px; }
          .metrics { grid-template-columns:repeat(3, minmax(0,1fr)); }
          .metric, .card, .task, .stack { background:rgba(9,17,30,.26); border:1px solid rgba(229,238,252,.12); border-radius:20px; transition:transform var(--t), border-color var(--t), box-shadow var(--t), background var(--t); }
          .metric { padding:16px 18px; }
          .metric:hover, .card:hover, .task:hover, .stack:hover { transform:translateY(-6px); border-color:rgba(103,232,249,.22); box-shadow:0 24px 48px rgba(2,8,23,.32); }
          .metric strong { display:block; margin-bottom:8px; font-size:1.45rem; }
          .metric span { color:var(--muted); font-size:.9rem; }
          .hero-side { display:grid; gap:16px; }
          .orbit { position:relative; width:min(100%,270px); aspect-ratio:1; margin:8px auto 18px; border-radius:50%; border:1px solid rgba(103,232,249,.2); background:radial-gradient(circle at center, rgba(103,232,249,.16), transparent 36%), radial-gradient(circle at 30% 30%, rgba(139,92,246,.12), transparent 24%); }
          .orbit::before, .orbit::after { content:""; position:absolute; border-radius:50%; border:1px dashed rgba(148,163,184,.16); }
          .orbit::before { inset:18%; } .orbit::after { inset:34%; }
          .core, .node { position:absolute; display:grid; place-items:center; border-radius:50%; text-align:center; }
          .core { inset:50%; width:88px; height:88px; transform:translate(-50%,-50%); background:linear-gradient(135deg, rgba(103,232,249,.95), rgba(139,92,246,.88)); color:#06111e; font-weight:800; box-shadow:0 0 34px rgba(103,232,249,.34); }
          .node { width:74px; height:74px; padding:12px; background:rgba(11,23,39,.92); border:1px solid rgba(103,232,249,.18); color:#dff9ff; font-size:.8rem; line-height:1.25; box-shadow:0 14px 24px rgba(2,8,23,.26); }
          .n1 { top:8%; left:50%; transform:translateX(-50%); } .n2 { right:6%; top:50%; transform:translateY(-50%); } .n3 { bottom:8%; left:50%; transform:translateX(-50%); } .n4 { left:6%; top:50%; transform:translateY(-50%); }
          .flow { display:grid; gap:14px; margin-top:14px; }
          .flow-item { display:grid; grid-template-columns:auto 1fr; gap:12px; align-items:start; }
          .flow-step { width:34px; height:34px; display:grid; place-items:center; border-radius:12px; color:var(--cyan); background:rgba(103,232,249,.12); border:1px solid rgba(103,232,249,.2); font-weight:700; }
          .grid { position:relative; z-index:2; display:grid; grid-template-columns:repeat(12, minmax(0,1fr)); gap:20px; margin-bottom:20px; }
          .span-4 { grid-column:span 4; } .span-8 { grid-column:span 8; } .span-12 { grid-column:span 12; }
          h2, h3 { margin:0 0 10px; letter-spacing:-.03em; }
          .tasks { grid-template-columns:repeat(3, minmax(0,1fr)); margin-top:20px; }
          .task { position:relative; overflow:hidden; }
          .task::after { content:""; position:absolute; inset:auto 24px 0; height:3px; border-radius:999px; background:linear-gradient(90deg, rgba(103,232,249,.95), rgba(139,92,246,.9)); }
          .easy { background:rgba(52,211,153,.12); color:#7ef0c3; }
          .medium { background:rgba(251,191,36,.12); color:#ffd970; }
          .hard { background:rgba(244,114,182,.12); color:#ff9dcf; }
          .endpoint-list { display:grid; gap:12px; margin-top:18px; }
          .endpoint { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:15px 16px; border-radius:18px; background:rgba(9,17,30,.24); border:1px solid rgba(229,238,252,.12); transition:transform var(--t), border-color var(--t), background var(--t); }
          .endpoint:hover { transform:translateX(4px); border-color:rgba(103,232,249,.2); background:rgba(103,232,249,.06); }
          .endpoint code { display:inline-flex; min-width:110px; justify-content:center; padding:10px 14px; border-radius:14px; background:rgba(8,15,28,.82); border:1px solid rgba(103,232,249,.12); color:#caf5ff; }
          .endpoint small { display:block; color:var(--muted); margin-top:4px; }
          .footer { position:relative; z-index:2; padding:24px 28px; text-align:center; color:var(--muted); }
          .reveal { opacity:0; transform:translateY(28px); animation:riseIn 780ms ease forwards; }
          .d1 { animation-delay:.08s; } .d2 { animation-delay:.16s; } .d3 { animation-delay:.24s; } .d4 { animation-delay:.32s; }
          @keyframes riseIn { to { opacity:1; transform:translateY(0); } }
          @keyframes floatGlow { 0%,100% { transform:translate3d(0,0,0) scale(1); } 50% { transform:translate3d(0,18px,0) scale(1.04); } }
          @media (max-width:1024px) { .hero, .tasks, .metrics { grid-template-columns:1fr; } .span-4, .span-8 { grid-column:span 12; } .static-logos { width:min(96vw, 980px); top:14vh; opacity:.46; } .floating-scaler { width:min(100vw, 1320px); top:5vh; opacity:.88; } }
          @media (max-width:860px) {
            .page { width:min(100% - 22px, 1180px); } .nav { flex-wrap:wrap; border-radius:28px; } .menu-button { display:inline-flex; }
            .nav-links { display:none; width:100%; flex-direction:column; align-items:stretch; padding-top:10px; } .nav-links.open { display:flex; }
            .nav-group, .nav-link, .nav-trigger, .dropdown { width:100%; } .dropdown { position:static; margin-top:8px; display:none; opacity:1; visibility:visible; transform:none; }
            .nav-group.is-open .dropdown { display:block; } .hero-copy, .hero-side, .card, .task, .stack { padding:22px; } .hero { min-height:820px; } .hero-copy, .hero-side { z-index:2; }
          }
          @media (max-width:640px) { .page { padding-top:14px; } .hero { padding-top:88px; } .hero h1 { font-size:2.55rem; } .cta-row { flex-direction:column; } .button { width:100%; } .endpoint { flex-direction:column; align-items:flex-start; } .static-logos { width:98vw; top:18vh; opacity:.38; } .floating-scaler { width:100vw; top:6vh; opacity:.82; } }
          @media (pointer:coarse) { body, a, button, .button, .nav-link, .nav-trigger, .endpoint { cursor:auto; } .cursor-shell, .cursor-core { display:none; } }
        </style>
      </head>
      <body>
        <img class="static-logos" id="static-logos" src="/assets/alllogos.png" alt="" />
        <img class="floating-scaler" id="floating-scaler" src="/assets/floater.png" alt="" />
        <div class="cursor-shell" id="cursor-shell" aria-hidden="true"></div>
        <div class="cursor-core" id="cursor-core" aria-hidden="true"></div>
        <div class="page">
          <nav class="nav reveal" aria-label="Main navigation">
            <div class="brand">
              <div class="brand-mark">S</div>
              <div><strong>Servixa</strong><span>Support automation evaluation environment</span></div>
            </div>
            <button class="menu-button" id="menu-button" aria-expanded="false" aria-controls="nav-links" aria-label="Toggle navigation">Menu</button>
            <div class="nav-links" id="nav-links">
              <div class="nav-group">
                <button class="nav-trigger" type="button" aria-expanded="false">Platform</button>
                <div class="dropdown" role="menu">
                  <a href="#overview" role="menuitem"><strong>Overview</strong><span>See what Servixa measures and why the environment matters.</span></a>
                  <a href="#tasks" role="menuitem"><strong>Task Design</strong><span>Explore the easy, medium, and hard support scenarios.</span></a>
                </div>
              </div>
              <div class="nav-group">
                <button class="nav-trigger" type="button" aria-expanded="false">API</button>
                <div class="dropdown" role="menu">
                  <a href="/docs" role="menuitem"><strong>Interactive Docs</strong><span>Browse and test every endpoint from Swagger UI.</span></a>
                  <a href="#endpoints" role="menuitem"><strong>Endpoint Guide</strong><span>Jump to the routes most useful for demos and judges.</span></a>
                </div>
              </div>
              <div class="nav-group">
                <button class="nav-trigger" type="button" aria-expanded="false">Evaluation</button>
                <div class="dropdown" role="menu">
                  <a href="/baseline" role="menuitem"><strong>Baseline Results</strong><span>Review benchmark scores from the reference policy.</span></a>
                  <a href="/grader" role="menuitem"><strong>Grader Output</strong><span>Inspect the current per-ticket scoring report.</span></a>
                </div>
              </div>
              <a class="nav-link" href="/status">Status</a>
            </div>
          </nav>

          <section class="hero">
            <div class="panel hero-copy reveal d1" id="overview">
              <div class="eyebrow">OpenEnv-compatible simulation</div>
              <h1>Evaluate support agents with <span>speed</span> and clarity.</h1>
              <p>Servixa turns customer support triage into a structured environment where agents must classify issues, route them safely, choose the right response, and close tickets only when the outcome is truly correct.</p>
              <div class="cta-row">
                <a class="button button-primary" href="/docs">Open API Docs</a>
                <a class="button button-secondary" href="/baseline">View Baseline Scores</a>
              </div>
              <div class="metrics">
                <div class="metric"><strong>3</strong><span>Difficulty tiers across realistic support workflows</span></div>
                <div class="metric"><strong>0.9708</strong><span>Average baseline score across the benchmark suite</span></div>
                <div class="metric"><strong>HTTP</strong><span>Simple reset-step-state loop for fast agent integration</span></div>
              </div>
            </div>

            <div class="panel hero-side reveal d2" aria-label="Architecture preview">
              <div class="stack">
                <div class="label">System Map</div>
                <div class="orbit">
                  <div class="core">Serve</div>
                  <div class="node n1">Agent</div>
                  <div class="node n2">Grader</div>
                  <div class="node n3">Tasks</div>
                  <div class="node n4">API</div>
                </div>
                <p class="muted">A compact evaluation loop designed for triage, escalation, response selection, and resolution quality.</p>
              </div>
              <div class="stack">
                <div class="label">Interaction Loop</div>
                <div class="flow">
                  <div class="flow-item"><div class="flow-step">1</div><div><strong>Reset a task</strong><div class="muted">Start a scenario with realistic tickets, guidance, and a step budget.</div></div></div>
                  <div class="flow-item"><div class="flow-step">2</div><div><strong>Take structured steps</strong><div class="muted">Classify, respond, and resolve while preserving safe escalation paths.</div></div></div>
                  <div class="flow-item"><div class="flow-step">3</div><div><strong>Measure outcomes</strong><div class="muted">Use shaped rewards and deterministic grading to benchmark quality.</div></div></div>
                </div>
              </div>
            </div>
          </section>

          <section class="grid">
            <article class="panel card span-4 reveal d2"><div class="label">Structured</div><h2>Typed environment surfaces</h2><p class="muted">Actions, observations, and state are modeled clearly so agents can integrate without guessing the contract.</p></article>
            <article class="panel card span-4 reveal d3"><div class="label">Deterministic</div><h2>Reliable evaluation</h2><p class="muted">Every ticket is scored across category, priority, route, response template, resolution, and closure safety.</p></article>
            <article class="panel card span-4 reveal d4"><div class="label">Deployment-ready</div><h2>Simple to run anywhere</h2><p class="muted">FastAPI, Docker, and Hugging Face Spaces make the environment easy to test, host, and judge quickly.</p></article>
          </section>

          <section class="grid" id="tasks">
            <article class="panel stack span-12 reveal d1">
              <div class="label">Scenario Design</div>
              <h2>Built to reflect real support pressure</h2>
              <p class="muted">The benchmark mixes standard service requests with high-risk tickets such as abuse reports, legal requests, VIP outages, and possible account compromise.</p>
              <div class="tasks">
                <div class="task"><div class="task-badge easy">Easy</div><h3>Starter queue</h3><p class="muted">Password reset and shipping delay scenarios test clean triage, routing, and safe closure decisions.</p></div>
                <div class="task"><div class="task-badge medium">Medium</div><h3>Policy nuance</h3><p class="muted">Refunds, duplicate charges, and abuse reports add specialist routing and higher-stakes prioritization.</p></div>
                <div class="task"><div class="task-badge hard">Hard</div><h3>Operational pressure</h3><p class="muted">Security, VIP incidents, legal review, and refund escalations stress-test judgment under mixed risk.</p></div>
              </div>
            </article>
          </section>

          <section class="grid" id="endpoints">
            <article class="panel stack span-8 reveal d2">
              <div class="label">Core Endpoints</div>
              <h2>Everything needed to run the loop</h2>
              <p class="muted">Use the environment over standard HTTP with a clear reset, step, inspect, and grade workflow.</p>
              <div class="endpoint-list">
                <a class="endpoint" href="/docs"><div><strong>Interactive docs</strong><small>Browse all routes and test requests from the browser.</small></div><code>/docs</code></a>
                <a class="endpoint" href="/tasks"><div><strong>Task list</strong><small>See available scenarios, difficulty, and ticket counts.</small></div><code>/tasks</code></a>
                <a class="endpoint" href="/schema"><div><strong>Schema</strong><small>Inspect the action, observation, and state models.</small></div><code>/schema</code></a>
                <a class="endpoint" href="/baseline"><div><strong>Baseline performance</strong><small>Review benchmark results from the included reference policy.</small></div><code>/baseline</code></a>
                <a class="endpoint" href="/status"><div><strong>Status page</strong><small>Open the human-friendly system health dashboard.</small></div><code>/status</code></a>
              </div>
            </article>
            <article class="panel stack span-4 reveal d3">
              <div class="label">Quick Start</div>
              <h2>Evaluation flow</h2>
              <p class="muted">A typical agent interaction looks like this:</p>
              <div class="endpoint-list">
                <div class="endpoint"><div><strong>Start a task</strong><small>Initialize the queue and receive the first observation.</small></div><code>POST /reset</code></div>
                <div class="endpoint"><div><strong>Act on tickets</strong><small>Submit classify, respond, or resolve actions.</small></div><code>POST /step</code></div>
                <div class="endpoint"><div><strong>Inspect progress</strong><small>Check current state, rewards, and final grader output.</small></div><code>GET /state</code></div>
              </div>
            </article>
          </section>

          <footer class="panel footer reveal d4">Servixa is built for customer support agent benchmarking, with deterministic grading and lightweight deployment.</footer>
        </div>
        <script>
          const menuButton = document.getElementById("menu-button");
          const navLinks = document.getElementById("nav-links");
          const navGroups = Array.from(document.querySelectorAll(".nav-group"));
          const floatingScaler = document.getElementById("floating-scaler");
          const cursorShell = document.getElementById("cursor-shell");
          const cursorCore = document.getElementById("cursor-core");
          let idleTimer;
          let cursorFrame = 0;
          let targetX = window.innerWidth / 2;
          let targetY = window.innerHeight / 2;
          let shellX = targetX;
          let shellY = targetY;
          let coreX = targetX;
          let coreY = targetY;
          if (menuButton && navLinks) {
            menuButton.addEventListener("click", () => {
              const expanded = menuButton.getAttribute("aria-expanded") === "true";
              menuButton.setAttribute("aria-expanded", String(!expanded));
              navLinks.classList.toggle("open");
            });
          }
          navGroups.forEach((group) => {
            const trigger = group.querySelector(".nav-trigger");
            if (!trigger) { return; }
            trigger.addEventListener("click", () => {
              if (!window.matchMedia("(max-width: 860px)").matches) { return; }
              const isOpen = group.classList.contains("is-open");
              navGroups.forEach((item) => {
                item.classList.remove("is-open");
                const button = item.querySelector(".nav-trigger");
                if (button) { button.setAttribute("aria-expanded", "false"); }
              });
              if (!isOpen) {
                group.classList.add("is-open");
                trigger.setAttribute("aria-expanded", "true");
              }
            });
          });
          document.addEventListener("click", (event) => {
            if (!event.target.closest(".nav")) {
              navGroups.forEach((group) => {
                group.classList.remove("is-open");
                const trigger = group.querySelector(".nav-trigger");
                if (trigger) { trigger.setAttribute("aria-expanded", "false"); }
              });
            }
          });
          window.addEventListener("resize", () => {
            if (!window.matchMedia("(max-width: 860px)").matches && navLinks) {
              navLinks.classList.remove("open");
              menuButton?.setAttribute("aria-expanded", "false");
              navGroups.forEach((group) => {
                group.classList.remove("is-open");
                const trigger = group.querySelector(".nav-trigger");
                if (trigger) { trigger.setAttribute("aria-expanded", "false"); }
              });
            }
          });
          const root = document.documentElement;
          const setCursorState = (state) => {
            if (!cursorShell || !cursorCore) { return; }
            cursorShell.classList.toggle("is-idle", state === "idle");
            cursorCore.classList.toggle("is-idle", state === "idle");
            cursorShell.classList.toggle("is-pressed", state === "pressed");
            cursorCore.classList.toggle("is-pressed", state === "pressed");
          };
          const scheduleIdle = () => {
            window.clearTimeout(idleTimer);
            idleTimer = window.setTimeout(() => setCursorState("idle"), 140);
          };
          const animateCursor = () => {
            if (cursorShell && cursorCore) {
              shellX += (targetX - shellX) * 0.18;
              shellY += (targetY - shellY) * 0.18;
              coreX += (targetX - coreX) * 0.34;
              coreY += (targetY - coreY) * 0.34;
              cursorShell.style.transform = `translate3d(${shellX}px, ${shellY}px, 0) translate3d(-50%, -50%, 0)`;
              cursorCore.style.transform = `translate3d(${coreX}px, ${coreY}px, 0) translate3d(-50%, -50%, 0)`;
            }
            cursorFrame = window.requestAnimationFrame(animateCursor);
          };
          const updateGlow = (clientX, clientY) => {
            root.style.setProperty("--cursor-x", `${clientX}px`);
            root.style.setProperty("--cursor-y", `${clientY}px`);
            targetX = clientX;
            targetY = clientY;
          };
          window.addEventListener("pointermove", (event) => {
            setCursorState("active");
            updateGlow(event.clientX, event.clientY);
            scheduleIdle();
          }, { passive: true });
          window.addEventListener("touchmove", (event) => {
            const point = event.touches && event.touches[0];
            if (point) { updateGlow(point.clientX, point.clientY); }
          }, { passive: true });
          window.addEventListener("mousedown", () => {
            setCursorState("pressed");
            window.clearTimeout(idleTimer);
          });
          window.addEventListener("mouseup", scheduleIdle);
          window.addEventListener("mouseleave", () => {
            if (!cursorShell || !cursorCore) { return; }
            cursorShell.style.opacity = "0";
            cursorCore.style.opacity = "0";
            if (cursorFrame) {
              window.cancelAnimationFrame(cursorFrame);
              cursorFrame = 0;
            }
          });
          window.addEventListener("mouseenter", () => {
            if (!cursorShell || !cursorCore) { return; }
            cursorShell.style.opacity = "";
            cursorCore.style.opacity = "";
            if (!cursorFrame) { animateCursor(); }
            scheduleIdle();
          });
          const updateSceneOnScroll = () => {
            const scrollY = window.scrollY || window.pageYOffset || 0;
            if (floatingScaler) {
              const floaterOffset = Math.min(scrollY * -0.28, 0);
              const floaterFade = Math.max(0, 0.92 - (scrollY / 260));
              floatingScaler.style.transform = `translate3d(-50%, ${floaterOffset}px, 0)`;
              floatingScaler.style.opacity = String(floaterFade);
            }
          };
          updateSceneOnScroll();
          window.addEventListener("scroll", updateSceneOnScroll, { passive: true });
          if (cursorShell && cursorCore && !cursorFrame) { animateCursor(); }
          scheduleIdle();
        </script>
      </body>
    </html>
    """


@app.get("/status", response_class=HTMLResponse)
def status_page() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Servixa Status</title>
        <style>
          :root { --bg:#07111f; --panel:rgba(10,19,33,.8); --line:rgba(148,163,184,.16); --text:#e5eefc; --muted:#9db0cc; --cyan:#67e8f9; --green:#34d399; --shadow:0 24px 60px rgba(2,8,23,.45); --t:260ms ease; }
          * { box-sizing:border-box; }
          body { margin:0; min-height:100vh; font-family:"Segoe UI",Arial,sans-serif; color:var(--text); background:radial-gradient(circle at top left, rgba(103,232,249,.14), transparent 34%), radial-gradient(circle at top right, rgba(52,211,153,.12), transparent 26%), linear-gradient(160deg,#04101d 0%,#081321 42%,#0d1830 100%); }
          .wrap { width:min(980px, calc(100% - 28px)); margin:0 auto; padding:32px 0 48px; }
          .panel { background:var(--panel); border:1px solid var(--line); border-radius:24px; box-shadow:var(--shadow); backdrop-filter:blur(20px); }
          .topbar { display:flex; justify-content:space-between; align-items:center; gap:16px; padding:18px 22px; margin-bottom:22px; }
          .brand { display:flex; align-items:center; gap:12px; }
          .mark { width:42px; height:42px; display:grid; place-items:center; border-radius:14px; background:linear-gradient(135deg, rgba(103,232,249,.24), rgba(52,211,153,.22)); color:var(--cyan); font-weight:800; }
          .brand span, .muted { color:var(--muted); }
          .link { display:inline-flex; align-items:center; justify-content:center; min-height:46px; padding:0 18px; border-radius:14px; border:1px solid rgba(103,232,249,.18); background:rgba(148,163,184,.05); transition:transform var(--t), box-shadow var(--t), border-color var(--t); }
          .link:hover, .link:focus-visible { transform:translateY(-2px); border-color:rgba(103,232,249,.3); box-shadow:0 18px 36px rgba(2,8,23,.28); outline:none; }
          .hero { padding:30px; margin-bottom:22px; }
          .badge { display:inline-flex; align-items:center; gap:10px; padding:8px 14px; border-radius:999px; background:rgba(52,211,153,.12); border:1px solid rgba(52,211,153,.18); color:#8ef3cb; font-size:.84rem; text-transform:uppercase; letter-spacing:.08em; }
          .dot { width:10px; height:10px; border-radius:50%; background:var(--green); box-shadow:0 0 18px rgba(52,211,153,.65); animation:pulse 1.8s ease-in-out infinite; }
          h1 { margin:18px 0 12px; font-size:clamp(2.4rem, 6vw, 4rem); line-height:1; letter-spacing:-.05em; }
          p { margin:0; line-height:1.7; }
          .grid { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:18px; margin-bottom:22px; }
          .card { padding:22px; transition:transform var(--t), border-color var(--t), box-shadow var(--t); }
          .card:hover { transform:translateY(-6px); border-color:rgba(103,232,249,.22); box-shadow:0 20px 42px rgba(2,8,23,.3); }
          .card strong { display:block; margin-bottom:8px; font-size:1.2rem; }
          code { display:inline-flex; padding:8px 12px; border-radius:12px; background:rgba(8,15,28,.82); border:1px solid rgba(103,232,249,.12); color:#caf5ff; }
          .actions { display:flex; flex-wrap:wrap; gap:12px; padding:24px 30px 30px; }
          @keyframes pulse { 0%,100% { transform:scale(1); opacity:1; } 50% { transform:scale(1.18); opacity:.8; } }
          @media (max-width:900px) { .grid { grid-template-columns:1fr; } }
          @media (max-width:640px) { .topbar { flex-direction:column; align-items:flex-start; } .actions { flex-direction:column; } .link { width:100%; } }
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="panel topbar">
            <div class="brand">
              <div class="mark">S</div>
              <div>
                <strong>Servixa Status</strong>
                <span>Human-friendly health overview</span>
              </div>
            </div>
            <a class="link" href="/">Back to Home</a>
          </div>

          <section class="panel hero">
            <div class="badge"><span class="dot"></span>System Healthy</div>
            <h1>Environment is online and ready.</h1>
            <p class="muted">This page is for humans. Machine checks should still use <code>/health</code> and receive the raw API response as usual.</p>
          </section>

          <section class="grid">
            <article class="panel card">
              <strong>API Health</strong>
              <p class="muted">The FastAPI service is responding normally.</p>
              <div style="margin-top:14px;"><code>/health -> healthy</code></div>
            </article>
            <article class="panel card">
              <strong>Environment Loop</strong>
              <p class="muted">Core evaluation routes remain available for reset, step, state, grading, and baseline checks.</p>
              <div style="margin-top:14px;"><code>/reset /step /state</code></div>
            </article>
            <article class="panel card">
              <strong>Judge Flow</strong>
              <p class="muted">Use the docs for exploration, the baseline for proof, and the grader for deterministic scoring.</p>
              <div style="margin-top:14px;"><code>/docs /baseline /grader</code></div>
            </article>
          </section>

          <section class="panel actions">
            <a class="link" href="/docs">Open API Docs</a>
            <a class="link" href="/health">Raw Health Endpoint</a>
            <a class="link" href="/baseline">Baseline Results</a>
            <a class="link" href="/tasks">Task List</a>
          </section>
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
def reset(payload: Optional[ResetPayload] = None) -> SupportObservation:
    payload = payload or ResetPayload()
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
