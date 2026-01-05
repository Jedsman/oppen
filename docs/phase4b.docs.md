# Phase 4b: Event-Driven Architecture (The Pager Upgrade)

**Objective**: Convert the proprietary Agent into a standard Web Service that can be triggered by Prometheus AlertManager (or any webhook).

## 1. Architecture

- **Server**: FastAPI application (`scripts/phase4b_agent_server.py`).
- **Endpoint**: `POST /webhook`.
- **Logic**:
  1. Accepts JSON payload: `{"alert": "..."}`.
  2. Wraps alert in a prompt: "Investigate this alert...".
  3. Reuses the **Robust Healer** (`phase4a_agent.py`) logic to solve the problem.

## 2. Integration

This bridges the gap between "Monitoring" and "Action".

- **Old Way**: Human sees Grafana -> Human runs `agent.py`.
- **New Way**: Prometheus sees spike -> AlertManager sends POST -> Agent fixes it.

## 3. Verification Results

- **Action**: Sent HTTP POST to `http://localhost:8090/webhook`.
- **Payload**: `{"alert": "podinfo deployment is degraded"}`.
- **Log Output**:
  ```
  [WEBHOOK] Received Alert: podinfo deployment is degraded
  [Agent] Starting investigation...
  [Agent] Calling: get_k8s_events({'namespace': 'default'})
  ```
- **Result**: Agent woke up and started using tools autonomously.

---

**Status**: [COMPLETE]
**Next**: Phase 4c (Incident Memory)
