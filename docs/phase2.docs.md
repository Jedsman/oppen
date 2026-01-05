# Phase 2: The Logic Bridge

**Objective**: Connect the AI Agent to the local infrastructure (Docker & Terraform) reliably on Windows.

## 1. Architecture Shift

We initially planned to use **MCP Servers** over stdio pipes. However, due to Windows/Python pipe buffering issues ("The Pipe Deadlock"), we pivoted to an **Embedded Tool Architecture**:

- **Old Plan**: Agent -> Pipe -> MCP Server -> Tool
- **Implemented**: Agent -> Python Function -> `subprocess` -> Tool

This ensures reliable execution with zero latency.

## 2. Components Built

### **The Agent Script** (`scripts/phase2_agent.py`)

- **Type**: LangGraph ReAct Agent
- **LLM**: `llama3.2:3b` (via Ollama)
- **Tools**:
  - `list_containers`: Executes `docker ps --format json`.
  - `terraform_run`: Executes `terraform [plan|apply|exclude]`.

### **The Docker Runtime** (`oppen-agent`)

- **Dockerfile**: `Dockerfile.agent` (Python 3.12 + Terraform 1.11.0)
- **Service**: `agent` in `docker-compose.yml`
- **Networking**: Connects to host via `host.docker.internal`.

## 3. How to Use

### **Option A: Host Execution (Fastest)**

Run directly on Windows. Requires `docker` and `terraform` in PATH.

```bash
uv run scripts/phase2_agent.py
```

### **Option B: Isolated Docker Execution**

Run inside a clean Linux container.
**Prerequisite**: Set `OLLAMA_HOST=0.0.0.0` on your host machine.

```bash
docker-compose up -d agent
docker exec -it oppen-agent uv run scripts/phase2_agent.py
```

## 4. Verification

- **TF Version**: Successfully checked Terraform v1.11.0/v1.14.3.
- **Docker**: Successfully listed local `kind` and `grafana` containers.
- **Observability**: Tool calls are now streamed to the console for transparency.

---

**Status**: [COMPLETE]
**Next**: Phase 3 (Chaos Engineering & Diagnostic Agents)
