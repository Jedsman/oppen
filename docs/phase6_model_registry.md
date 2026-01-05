# Phase 6: Model Registry & Versioning

## Objective
Implement a centralized model registry to manage model artifacts, versions, and metadata. Enable multi-model serving with version control and rollback capabilities.

## 1. MLflow Model Registry
Set up the Model Registry component of MLflow (built on Phase 5).

### Steps:
1. **Enable Model Registry**:
   - Use MLflow's built-in Model Registry (requires backend store: SQL database or filesystem).
   - For local setup, use PostgreSQL (lightweight) or SQLite.
2. **Register Models**:
   - After training (Phase 5), programmatically register the best model:
     ```python
     mlflow.register_model(f"runs:/{run_id}/model", "mnist-classifier")
     ```
3. **Version Management**:
   - Each registration creates a new version (v1, v2, v3...).
   - Transition versions between stages: `Staging`, `Production`, `Archived`.

## 2. Model Metadata & Governance
Define model contracts and metadata.

### Steps:
1. **Model Card Documentation**:
   - For each model, store metadata in MLflow:
     - Model name, description, owner.
     - Performance baseline (accuracy, latency).
     - Training dataset, preprocessing steps.
     - Intended use cases and limitations.
2. **Tagging**:
   - Tag models: `environment:dev`, `framework:pytorch`, `task:classification`.
   - Query by tags: "Find all production models".
3. **Audit Trail**:
   - MLflow automatically logs who registered a model and when.
   - Store decision rationale (e.g., "Selected v3 for 2% accuracy improvement").

## 3. Model Serving Setup (Multi-Model)
Deploy multiple models simultaneously with different resource requirements.

### Steps:
1. **Model Serving Framework** (choose one):
   - **TorchServe**: Lightweight, PyTorch-native.
   - **vLLM**: Optimized for LLMs, batch inference.
   - **KServe**: K8s-native, supports multiple frameworks.

   *Recommendation: Start with TorchServe for simplicity.*

2. **TorchServe Deployment**:
   - Create model archives (`.mar` files) from MLflow models.
   - Deploy TorchServe as a K8s Deployment.
   - Expose via service (port 8080 for inference, 8081 for management).

3. **Model Loading**:
   - Configure TorchServe to load models from MLflow Model Registry.
   - Example: `torchserve --model-store /models --ncs --load-models path/to/v1/pytorch_model.pt`.

## 4. Version Control & Rollback
Manage model transitions and enable quick rollbacks.

### Steps:
1. **Version Promotion Workflow**:
   - New models start in `Staging`.
   - Agent or human reviews performance in staging.
   - Approve promotion to `Production` (TorchServe loads this version).
2. **Rollback Mechanism**:
   - If a new production model degrades, instantly revert to previous version.
   - Implement via Terraform + K8s: Update ConfigMap with model version, restart pods.
3. **Terraform Integration**:
   - Define desired model version in Terraform state.
   - Changes to model version trigger pod restart.

## 5. Model Registry API & Agent Integration
Enable agents to interact with the model registry.

### New MCP Server:
1. **Model Registry MCP Server**:
   - Expose tools:
     - `list_models()`: List all registered models.
     - `get_model_info(model_name, version)`: Fetch metadata.
     - `promote_model(model_name, version, stage)`: Transition stages.
     - `get_serving_version(model_name)`: Check current production version.

### Agent Nodes (LangGraph):
1. **Model Selection Node**:
   - Agent queries registry to recommend a model based on use case.
   - "For classification tasks, I recommend `mnist-classifier:v3` (99.2% accuracy)".
2. **Promotion Node**:
   - Agent proposes: "Promote `model-v4` to Production (2% improvement over v3)".
   - Requires human-in-the-loop approval.

## 6. Multi-Model Orchestration
Support serving multiple models with intelligent routing.

### Steps:
1. **Model Configuration**:
   - Define model metadata: task (classification, generation), input schema, latency SLA.
   - Store in K8s ConfigMaps or a central config file.
2. **Routing Logic**:
   - Simple routing: Route by model name in URL path.
   - Advanced: Route by input features (e.g., "high-accuracy model for critical requests").
3. **Resource Allocation**:
   - Assign replicas and resource limits per model.
   - Example: `classifier:v3` gets 2 replicas + 1 GPU, `feature-extractor:v1` gets 4 replicas + 0 GPU.

## 7. Simulation Run
Test the complete model registry workflow.

### Scenario:
1. Register two models from Phase 5 experiments (v1 and v2).
2. Deploy both to TorchServe with different resource allocations.
3. Agent queries registry: "What models are in production?"
4. Agent proposes promotion: "Promote v2 to production".
5. Verify promotion via MLflow UI and TorchServe management API.
6. Rollback to v1 and verify service continuity.

## 8. Monitoring & Drift Detection (Preview to Phase 8)
Track which model version is serving production traffic.

### Steps:
1. **Version Tracking**:
   - Log TorchServe version metrics to Prometheus.
   - Grafana dashboard: Show which model version is active, change history.
2. **Performance Metrics**:
   - Log inference latency and error rates per model version.
   - Track accuracy on production data (foundation for Phase 8 drift detection).

## Deliverables
- MLflow Model Registry configured with SQL backend.
- Model metadata and governance structure (cards, tags).
- TorchServe (or alternative) multi-model serving deployment on K8s.
- Terraform code managing model versions and rollbacks.
- Model Registry MCP Server with promotion/querying tools.
- Agent nodes for model selection and promotion workflows.
- Multi-model routing logic with resource allocation.
- Grafana dashboard tracking active model versions and performance.

## Learning Outcomes
- Model versioning and promotion workflows.
- Multi-model serving and resource allocation.
- Model governance and audit trails.
- Seamless rollback and disaster recovery for model changes.
- Integrating ML systems (MLflow) with infrastructure (K8s, Terraform).
