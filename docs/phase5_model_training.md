# Phase 5: Model Training Pipeline & Experiment Tracking

## Objective
Establish a local model training environment with experiment tracking, GPU resource management, and cost monitoring. Move beyond inference-only agents to building and fine-tuning models.

## 1. Prerequisites & Environment Setup
- [ ] **GPU Support**: Ensure Docker/Kubernetes supports GPU access (NVIDIA Container Toolkit, nvidia-device-plugin).
- [ ] **Training Frameworks**: Install PyTorch or TensorFlow locally and in containers.
- [ ] **MLflow Server**: Deploy MLflow as a K8s service for experiment tracking and model registry.
- [ ] **Python Stack**: Add `pytorch`, `scikit-learn`, `pandas`, `mlflow` to dependencies.
- [ ] **Storage**: Set up persistent volumes for training datasets and model checkpoints.

## 2. GPU Resource Configuration (Kubernetes)
Configure GPU requests and limits for training workloads.

### Steps:
1. **NVIDIA Device Plugin**:
   ```bash
   helm repo add nvidia https://nvidia.github.io/k8s-device-plugin
   helm install nvidia-device-plugin nvidia/nvidia-device-plugin -n kube-system
   ```
2. **Namespace Setup**:
   - Create a `ml-training` namespace with resource quotas.
   - Define pod limits: GPU count (e.g., 1 GPU per training pod), memory, CPU.
3. **Terraform IaC**:
   - Define GPU node affinity in Terraform.
   - Add resource limits to K8s manifests.

Example pod spec (Terraform):
```hcl
resources {
  limits = {
    "nvidia.com/gpu" = "1"
    memory           = "8Gi"
    cpu              = "4"
  }
  requests = {
    "nvidia.com/gpu" = "1"
    memory           = "8Gi"
    cpu              = "4"
  }
}
```

## 3. MLflow Deployment
Set up MLflow to track experiments locally.

### Steps:
1. **MLflow Server**:
   - Deploy MLflow as a K8s Deployment with persistent storage for artifacts.
   - Expose via port-forward to `localhost:5000`.
2. **Backend Setup**:
   - Use local filesystem for backend store (SQLite for simplicity initially).
   - Point artifact store to a persistent volume (for model checkpoints).
3. **Access Verification**:
   - Visit `http://localhost:5000` to confirm the UI is accessible.

## 4. Training Script & Experiment Tracking
Create a basic model training pipeline with MLflow integration.

### Steps:
1. **Dataset Preparation**:
   - Download a small public dataset (e.g., MNIST, CIFAR-10, or a small LLM fine-tuning dataset).
   - Store in a persistent volume mount.
2. **Training Script** (`train.py`):
   - Use PyTorch to train a simple model (e.g., CNN on MNIST or fine-tune a small LLM like DistilBERT).
   - Integrate MLflow:
     ```python
     import mlflow
     mlflow.set_experiment("phase5-training")
     with mlflow.start_run():
         mlflow.log_param("learning_rate", lr)
         mlflow.log_metric("train_loss", loss)
         mlflow.pytorch.log_model(model, "model")
     ```
3. **K8s Job**:
   - Create a K8s Job manifest that runs `train.py` with GPU resources.
   - Use Terraform to manage the Job lifecycle.

## 5. Cost Tracking & GPU Utilization
Monitor compute costs per experiment.

### Steps:
1. **Cost Calculation**:
   - Define hourly costs for GPU, CPU, and memory.
   - Log training duration and resource usage to MLflow.
   - Example: `cost = (gpu_hours * $0.25) + (cpu_hours * $0.05)`.
2. **Prometheus Metrics**:
   - Export GPU utilization from nvidia-gpu-prometheus-exporter.
   - Query metrics: `nvidia_smi_utilization_gpu`, `nvidia_smi_memory_used_mb`.
3. **MLflow Integration**:
   - Log cost per experiment as a metric.
   - Create Grafana dashboard showing cost trends.

## 6. Agent Integration (LangGraph)
Extend the Phase 4 agent to trigger and monitor training runs.

### New Agent Nodes:
1. **Training Trigger Node**:
   - Agent can propose: "Run training with learning_rate=0.001".
   - Uses K8s API (via MCP) to submit a job.
2. **Monitoring Node**:
   - Agent polls MLflow API to check experiment progress.
   - Queries Prometheus for GPU utilization.
3. **Cost Analysis Node**:
   - Agent reports: "Training cost: $2.45, GPU utilization: 85%".

## 7. Simulation Run
Test the complete training pipeline.

### Scenario:
1. Start a training job via agent: "Train a CNN on MNIST with 3 epochs".
2. Agent monitors job status and GPU metrics in real-time.
3. Upon completion, agent reports metrics (accuracy, cost, duration).
4. Inspect MLflow UI to verify experiment logged correctly.

## Deliverables
- NVIDIA GPU plugin and K8s resource quotas configured.
- MLflow server deployed on K8s with persistent storage.
- Training script (PyTorch/TensorFlow) with MLflow integration.
- K8s Job manifests (Terraform-managed) for training.
- Cost tracking integrated with MLflow (hourly rates defined).
- Agent nodes for triggering and monitoring training runs.
- Example dashboard in Grafana showing cost trends.

## Learning Outcomes
- GPU resource management in Kubernetes.
- Experiment tracking and artifact management.
- Cost attribution and tracking for ML workloads.
- Agent orchestration of training pipelines.
