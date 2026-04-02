# LLM Study Prompt

Use the prompt below with another LLM if you want a detailed tutoring-style walkthrough of this project.

```text
You are my technical tutor.

I am going to study a Python project called Servixa. It is an OpenEnv-compatible customer support triage benchmark built with FastAPI, Pydantic, Docker, and Hugging Face Spaces.

Your job is to teach me this project from absolute basics all the way to advanced understanding.

Please explain everything in a very clear, layered way:

1. First give me the high-level purpose of the project in simple words.
2. Then explain why this problem matters in the real world.
3. Then explain the architecture of the project:
   - server/app.py
   - env/models.py
   - env/environment.py
   - env/grader.py
   - env/tasks.py
   - baseline.py
   - inference.py
   - Dockerfile
   - openenv.yaml
4. Explain how data flows through the system when reset() is called.
5. Explain how data flows through the system when step(action) is called.
6. Explain the Pydantic models and why they matter.
7. Explain how the reward shaping works.
8. Explain how the deterministic grader works.
9. Explain how the baseline policy works and why it is intentionally not perfect.
10. Explain how the inference script works and why it uses the OpenAI-compatible client.
11. Explain how FastAPI, Pydantic, OpenEnv, Docker, and Hugging Face Spaces are connected in this project.
12. Explain the three tasks one by one and what makes each one easy, medium, or hard.
13. Explain the important libraries used in this project:
   - fastapi
   - pydantic
   - uvicorn
   - openenv-core
   - openai python client
14. Explain the API endpoints and what each one does.
15. Explain the most important engineering decisions and tradeoffs in the project.
16. Explain the likely interviewer questions I could get about this project.
17. Explain possible weaknesses, limitations, and future improvements.

Teaching style requirements:

- Start simple, then go deeper gradually.
- Use analogies where helpful.
- Use numbered steps when explaining flows.
- After each major section, give me a short summary.
- If you mention any file, explain what role it plays in the whole system.
- If you mention any library, explain why it is used here.
- Do not assume I already understand OpenEnv, FastAPI, or Pydantic deeply.
- Be extremely detailed but easy to follow.

At the end, give me:

- a concise one-minute explanation of the project
- a three-minute explanation of the project
- a list of possible viva/interview questions and strong answers
- a list of the most important things I must remember before presenting this project
```

