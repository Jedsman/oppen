# Phase 1: Foundations & First Contact

## Objective
Establish the local cloud environment and immediately connect a local AI agent (Ollama) to it using the Docker MCP server.

## 1. Prerequisites & Environment Setup
- [ ] **Operating System**: Ensure WSL2 (Windows Subsystem for Linux) is active if on Windows.
- [ ] **Container Runtime**: Install Docker Desktop or Rancher Desktop.
- [ ] **CLI Tools**: Install `kubectl`, `helm`, and `terraform`.
- [ ] **AI Engine**: Install **Ollama** and pull a model (e.g., `ollama pull llama3`).
- [ ] **Python Stack**: Install Python 3.11+ and `UV` or `venv`.
    - Dependencies: `langchain`, `langgraph`, `mcp`.

## 2. Orchestration (Kubernetes)
We will use **Kind (Kubernetes in Docker)** or **k3s** for the local cluster.

### Steps:
1.  **Cluster Creation**:
    ```bash
    kind create cluster --name oppen-local --config kind-config.yaml
    ```
2.  **Verification**:
    - Run `kubectl get nodes` to ensure the control plane is ready.

## 3. The First MCP Connection (Docker)
Instead of waiting, we enable the agent to see the container runtime immediately.

### Steps:
1.  **Docker MCP Server**:
    - Install/Run the Docker MCP server locally (e.g., via `npx @modelcontextprotocol/server-docker` or Python equivalent).
2.  **Basic Agent (Python)**:
    - Create a simple script `agent.py` using **LangChain**.
    - Configure it to use **Ollama** as the LLM.
    - Connect it to the Docker MCP server using the `mcp` Python client.
3.  **Verification**:
    - Run the script and ask: "List all running containers".
    - Verify it sees the Kind control plane container.

## 4. Infrastructure as Code (IaC)
Use Terraform to manage the lifecycle of the local cluster resources to ensure reproducibility.

### Steps:
1.  **Provider Setup**: Configure the `kubernetes` and `helm` providers in a `main.tf` file.
2.  **State Management**: Use a local backend for Terraform state initially.
3.  **Resource Definition**: Define the base namespaces (e.g., `monitoring`, `apps`).

## 5. Observability Stack
Deploy the "eyes" of the system using the kube-prometheus-stack.

### Steps:
1.  **Helm Repository**:
    ```bash
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    ```
2.  **Deployment**:
    - Deploy Prometheus and Grafana to the `monitoring` namespace.
    - Ensure default ServiceMonitors are active for K8s components.
3.  **Access**:
    - Port-forward Grafana to localhost (e.g., port 3000) to verify dashboards.

## Deliverables
- Running K8s cluster.
- Functional Python+LangChain script querying Docker via MCP.
- Terraform code for base infrastructure.