# Docker Troubleshooting Checklist

If Docker validation failed on your machine, use this checklist.

## What likely happened

The earlier failure was not a Dockerfile syntax error. It happened because the Docker daemon was not available on the machine.

The error looked like:

- Docker Desktop pipe not found
- Docker daemon not running

That means the fix is usually local Docker setup, not project code.

## Step 1: Start Docker Desktop

On Windows:

1. Open Docker Desktop manually
2. Wait until it says Docker is running
3. Then run:

```powershell
docker version
docker info
```

If both commands work, the daemon is running.

## Step 2: Re-run the build

From the project root:

```powershell
docker build -t servixa .
```

If that passes, run:

```powershell
docker run -p 7860:7860 servixa
```

Then open:

```text
http://localhost:7860/health
```

## Step 3: Check for permission issues

If Docker still fails:

- restart Docker Desktop
- restart your terminal
- make sure WSL2 is enabled if Docker Desktop expects it
- run Docker Desktop as your normal user first
- if needed, reboot once and try again

## Step 4: Check the exact endpoints

After the container starts, verify:

```powershell
curl http://localhost:7860/health
curl -Method Post http://localhost:7860/reset
```

## Step 5: Match the validator path

Your final validation checklist should be:

```powershell
docker build .
openenv validate
python baseline.py
python inference.py
```

## If you want me to verify Docker with you

Once Docker Desktop is running, ask me to:

- run `docker build .`
- run the local container
- ping `/health`
- ping `/reset`

That will let me confirm the Docker gate end to end.
