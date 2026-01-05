# Phase 1: Local Cloud Foundation - Walkthrough

We have successfully established the foundational "Local Cloud" environment.

## 1. Components Deployed

### **Core Infrastructure (Kubernetes)**

- **Cluster**: `kind` (Kubernetes in Docker) running a single control-plane node.
- **Config**: `kind-config.yaml`
- **Verification**: `kubectl get nodes` shows `oppen-local-control-plane`.

### **AI Agent (MCP)**

- **Script**: `scripts/phase1_agent.py` using `langgraph` and `langchain-ollama`.
- **Model**: `llama3.2:3b` (via Ollama).
- **Tools**: `docker-mcp` connects the agent to the local Docker socket.
- **Capabilities**: The agent can list and inspect running Docker containers.
  > [!NOTE] > **Windows Users**: Ensure Docker Desktop is configured to "Expose daemon on tcp://localhost:2375 without TLS" in Settings > General.

### **Observability Stack**

- **Prometheus**: Collecting metrics from the cluster.
- **Grafana**: Visualizing metrics.
- **Access**: Accessed via a persistent Docker tunnel.
  - **URL**: [http://localhost:3000](http://localhost:3000)
  - **User**: `admin`
  - **Password**: (Auto-generated, retrieved from K8s secrets)

## 2. Accessing the Environment

### Grafana Dashboard

1. Go to [http://localhost:3000](http://localhost:3000)
2. Login with the credentials provided in the chat.
3. Explore default dashboards (e.g., "Kubernetes / Compute Resources / Cluster").

### Running the Agent

```bash
uv run scripts/phase1_agent.py
```

This will start the AI agent which will query the local Docker environment and list containers.

## 3. Infrastructure as Code (IaC)

- **Terraform**: `terraform/main.tf` manages the `monitoring` and `apps` namespaces.
- **State**: Currently stored locally in `terraform/terraform.tfstate`.

## 4. Next Steps (Phase 2)

- Enhancing the Agent's capabilities (giving it access to K8s via MCP).
- Building a more complex "Diagnostic Agent" (Phase 3).
