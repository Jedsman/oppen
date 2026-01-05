# Phase 3: Chaos & Detection

## Objective
Introduce controlled failures into the environment and build a "Diagnostic Agent" capable of detecting and reasoning about these failures.

## 1. Chaos Engineering
Deploy tools to simulate faults.

### Setup:
- **Tool Selection**: Use **LitmusChaos** (Cloud-native) or a simple Python **Chaos Script**.
- **Scenario 1 (Pod Kill)**: Randomly terminate a pod in the `apps` namespace.
- **Scenario 2 (Network Latency)**: Introduce delay to a specific service.

## 2. Prometheus Integration for MCP
Enable the agent to read metrics.

### Steps:
- **Prometheus MCP Server**: Set up an MCP server that connects to your local Prometheus instance.
- **Tool Definition**: Expose tools like `query_prometheus(query: str)` to the agent.

## 3. The Diagnostic Agent (LangGraph)
Build the detection logic using graph nodes.

### Workflow (Graph Nodes):
1.  **Monitor Node**: Polls Prometheus (via MCP) or receives alerts.
2.  **Investigate Node**:
    - If alert -> query `docker_logs` (Docker MCP).
    - If alert -> query `query_prometheus`.
3.  **Reasoning Node**:
    - **Ollama** analyzes the logs and metrics to identify the root cause (e.g., "OOMKilled", "Connection Refused").

## 4. Simulation Run
- Manually trigger a chaos experiment.
- Observe if the agent correctly identifies the issue via the chat interface.

## Deliverables
- Chaos scripts/manifests.
- Prometheus MCP Server integration.
- A "Diagnostic Agent" workflow capable of identifying at least one failure type.