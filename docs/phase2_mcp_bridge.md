# Phase 2: Expanding the Agent (IaC & LangGraph)

## Objective
Expand the agent's capabilities to include Infrastructure as Code (Terraform) and structure the reasoning loop using LangGraph.

## 1. MCP Server Configuration
Add the Terraform capability to the existing MCP setup.

### Terraform MCP Server
- **Goal**: Allow the agent to plan and apply Terraform configurations.
- **Setup**:
  - Create or deploy a local MCP server that wraps the Terraform CLI.
  - Expose tools like `terraform_plan`, `terraform_apply`, and `terraform_show`.

## 2. LangGraph Architecture
Transition from the simple script in Phase 1 to a stateful graph.

### Setup:
- **State Definition**: Define `AgentState` (messages, tools, current_step).
- **Nodes**:
  - `chatbot`: Calls **Ollama** with the current context.
  - `tools`: Executes MCP tool calls (Docker from Phase 1, Terraform from Phase 2).
- **Graph Construction**:
  - Create a graph where the LLM decides to call a tool or end the conversation.
  - Implement a "human-in-the-loop" check before `terraform_apply`.

## 3. Connectivity Testing
- **Test**: Ask the agent "Show me the current Terraform state resources".
- **Validation**: Ensure the agent receives structured JSON data back from the tools.

## Deliverables
- Running MCP server for Terraform.
- A LangGraph application connecting Ollama to both Docker and Terraform MCPs.