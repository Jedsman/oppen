# Future Roadmap: Improving the Self-Healing Platform

## 1. Event-Driven Architecture (The "Pager" Upgrade)

**Current:** You manually run the Agent when you suspect an issue.
**Improvement:** Connect **Prometheus AlertManager** to the Agent.

- **How:** Wrap the Agent in a lightweight FastAPI server with a webhook endpoint.
- **Flow:** `Prometheus Alert` -> `AlertManager` -> `POST /webhook` -> `Agent Wakes Up` -> `Remediates`.
- **Benefit:** True "24/7" automated operations without human trigger.

## 2. Robust IaC Editing (The "Safety" Upgrade)

**Current:** The Healer Agent uses Regex to find/replace text in `podinfo.tf`. This is brittle; if whitespace changes, it breaks.
**Improvement:** Use `terraform.tfvars.json`.

- **How:** Modify Terraform to use variables (`var.replica_count`). The Agent simply writes a JSON file (`remediation.auto.tfvars.json`): `{"replica_count": 5}`.
- **Benefit:** Impossible to break the HCL syntax. Standard Terraform practice.

## 3. Incident Memory (The "Senior Engineer" Upgrade)

**Current:** The Agent solves a problem and forgets. If it happens again tomorrow, it re-derives the solution.
**Improvement:** Add a **Vector Database** (e.g., ChromaDB).

- **How:** After fixing an issue, the Agent writes a "Post-Mortem" to the DB.
- **Flow:** On new alert, Agent queries DB: _"Have I seen 'podinfo crash' before?"_ -> Retreives: _"Yes, last time scaling fixed it."_
- **Benefit:** Faster resolution time; learning from experience.

## 4. MCP over HTTP (The "Bridge" Repair)

**Current:** We use embedded tools (`subprocess`) because Windows pipes are flaky. This couples the tools to the Agent script.
**Improvement:** Use **MCP over SSE (Server-Sent Events)**.

- **How:** Run the Terraform/Docker tools as standalone HTTP servers (`fastmcp run --transport sse ...`).
- **Benefit:** Decouples Agent from Tools. The Agent can run in the cloud and control your local machine securely via a tunnel (e.g. `ngrok`), allowing for much more powerful hosted LLMs.
