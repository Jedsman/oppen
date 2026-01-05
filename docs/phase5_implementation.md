# Phase 5: Model Training Pipeline & Experiment Tracking - Implementation

## Overview

Phase 5 adds a complete ML training pipeline to Oppen with:
- **MLflow** for experiment tracking and model registry (K8s-deployed)
- **Training scripts** for MNIST (quick validation) and LLM fine-tuning (realistic)
- **GPU support** via NVIDIA device plugin (optional, auto-detects)
- **Cost tracking** for training runs (GPU + CPU hours)
- **Agent tools** to trigger and monitor training jobs
- **Production-ready** containerization for training workloads

## Architecture

```
Agent (LangGraph)
  ├─ trigger_training_job() → K8s Job (MNIST/LLM)
  ├─ get_training_status() → Monitor progress & logs
  ├─ list_mlflow_experiments() → View experiments
  ├─ get_experiment_metrics() → Detailed metrics
  └─ calculate_training_cost() → Cost breakdown

K8s ml-training namespace
  ├─ mlflow-server (Deployment) → SQLite backend
  └─ training-jobs (Batch Jobs) → MNIST/LLM containers

Docker images
  ├─ oppen-training-mnist:latest → CNN on MNIST dataset
  └─ oppen-training-llm:latest → Transformer fine-tuning

MLflow UI
  └─ http://localhost:5000 → Experiment tracking
```

## Quick Start

### 1. Deploy Infrastructure

```bash
cd C:\Users\theje\code\oppen\terraform

# Apply Terraform (MLflow + namespace)
terraform apply -auto-approve

# Verify
kubectl get pods -n ml-training
# Should show: mlflow-server pod in Running state
```

### 2. Build Training Containers

```bash
cd C:\Users\theje\code\oppen

# Build both images
docker build -t oppen-training-mnist:latest -f Dockerfile.training-mnist .
docker build -t oppen-training-llm:latest -f Dockerfile.training-llm .

# Load to Kind cluster
kind load docker-image oppen-training-mnist:latest --name oppen-local
kind load docker-image oppen-training-llm:latest --name oppen-local

# Verify
docker images | grep oppen-training
```

### 3. Access MLflow UI

```bash
# Port-forward (background)
kubectl port-forward -n ml-training svc/mlflow-server 5000:5000 &

# Open browser
open http://localhost:5000
# or: navigate to http://localhost:5000
```

### 4. Test Agent Integration

```bash
# Start agent REPL
uv run -m app.main repl

# Example commands:
> list mlflow experiments
# Shows: No experiments yet (will appear after first training run)

> trigger training job mnist with 2 epochs
# Submits K8s Job, returns job name

> get training status mnist-training-20260103-140000
# Shows: job status, logs, resource usage

> calculate training cost mnist-training-20260103-140000
# Shows: CPU cost, GPU cost, duration (after job completes)
```

## Training Scripts

### MNIST Training (Quick Validation)

**File**: `scripts/training/train_mnist.py`

- **Dataset**: MNIST (handwritten digits)
- **Model**: Simple CNN (32→64 filters)
- **Duration**: 2-3 min for 2 epochs
- **Output**: 98-99% validation accuracy
- **Cost**: ~$0.02-0.05

**Run locally**:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/training/train_mnist.py --epochs 2 --lr 0.001 --batch-size 64
```

**Via agent**:
```
> trigger training job mnist with 2 epochs and 0.001 learning rate
```

### LLM Fine-Tuning (Realistic)

**File**: `scripts/training/train_llm.py`

- **Dataset**: WikiText-2 (language modeling)
- **Model**: DistilGPT2 (82M params)
- **Duration**: 5-10 min for 1 epoch
- **Output**: Reduced perplexity
- **Cost**: ~$0.10-0.25

**Run locally**:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/training/train_llm.py --model-name distilgpt2 --epochs 1 --batch-size 4
```

**Via agent**:
```
> trigger training job llm with 1 epoch
```

## Agent Tools Reference

### `list_mlflow_experiments(experiment_name=None)`
Lists all MLflow experiments with recent runs and metrics.

```
> list mlflow experiments
# Output:
# MLflow Experiments:
#   mnist-training (ID: 1)
#     - a1b2c3d4: Acc=98.5, Cost=$0.023
#     - e5f6g7h8: Acc=98.2, Cost=$0.021
```

### `trigger_training_job(model_type, epochs=3, lr=0.001, batch_size=64, gpu_enabled=True)`
Submits a K8s training job. **Requires human approval.**

```
> trigger training job mnist with 5 epochs
# Output:
# Job submitted: mnist-training-20260103-140000
# Status: pending
# View: kubectl logs -n ml-training -l job-name=mnist-training-20260103-140000 -f
```

### `get_training_status(job_name)`
Monitors job progress and shows recent logs.

```
> get training status mnist-training-20260103-140000
# Output:
# Job: mnist-training-20260103-140000
# Status: running
# Active: 1, Succeeded: 0, Failed: 0
#
# Logs (last 20 lines):
# Epoch 1/5 - Loss: 0.0523, Train: 98.2%, Val: 97.8%
# ...
```

### `get_experiment_metrics(run_id)`
Fetches detailed metrics and parameters for a training run.

```
> get experiment metrics a1b2c3d4
# Output:
# Run a1b2c3d4
# Status: FINISHED
# Metrics:
#   train_loss: 0.0421
#   train_accuracy: 98.5
#   val_accuracy: 98.2
#   duration_seconds: 213.4
#   total_cost_usd: 0.023
```

### `calculate_training_cost(job_name)`
Computes cost of completed job (GPU + CPU hours).

```
> calculate training cost mnist-training-20260103-140000
# Output:
# Duration: 213.4s
# CPU Cost: $0.0147 (4 cores × 0.059 hrs × $0.05/core-hour)
# GPU Cost: $0.0083 (1 GPU × 0.059 hrs × $0.25/hour)
# Total Cost: $0.023
```

## MLflow Features

### Experiment Tracking
- **Automatic logging**: All hyperparameters, metrics, artifacts
- **UI dashboard**: View experiments, compare runs, download models
- **Model registry**: Version control for trained models

### Key Metrics Logged
| Metric | Meaning |
|--------|---------|
| `train_loss` | Training loss per epoch |
| `train_accuracy` | Training accuracy per epoch |
| `val_accuracy` | Validation accuracy per epoch |
| `duration_seconds` | Total training time |
| `cpu_cost_usd` | Compute cost (CPU hours) |
| `gpu_cost_usd` | Compute cost (GPU hours) |
| `total_cost_usd` | Total training cost |

### Model Artifacts
- `model/` directory contains trained PyTorch/HuggingFace model
- Downloadable from MLflow UI for inference

## Monitoring Training

### Real-time Logs

```bash
# Watch training progress
kubectl logs -n ml-training -l job-name=mnist-training-20260103-140000 -f

# Example output:
# Training on: cuda
# GPU: NVIDIA A100
# Epoch 1/2 - Loss: 0.2145, Train: 95.2%, Val: 94.8%
# Epoch 2/2 - Loss: 0.0523, Train: 98.5%, Val: 98.1%
# Complete! Duration: 213.1s, Cost: $0.023
```

### Pod Status

```bash
# Check job status
kubectl get jobs -n ml-training

# Check pod status
kubectl get pods -n ml-training

# Describe pod (debug)
kubectl describe pod -n ml-training <pod-name>
```

### Job Completion

```bash
# Wait for job completion
kubectl wait --for=condition=complete job/mnist-training-20260103-140000 -n ml-training --timeout=600s

# Or watch until complete
watch kubectl get jobs -n ml-training
```

## Cost Model

### Pricing

Default rates (configurable via Terraform):
- **GPU**: $0.25/hour (A100 equivalent)
- **CPU**: $0.05/core-hour (4 cores typical)

### Example Costs

| Workload | Duration | GPU | CPU | Total |
|----------|----------|-----|-----|-------|
| MNIST (2 epochs) | 3.5 min | $0.002 | $0.012 | $0.014 |
| MNIST (10 epochs) | 17.5 min | $0.010 | $0.060 | $0.070 |
| LLM (1 epoch) | 8 min | $0.033 | $0.027 | $0.060 |
| LLM (5 epochs) | 40 min | $0.167 | $0.133 | $0.300 |

### Customize Rates

Edit `terraform/variables.tf`:

```hcl
variable "gpu_hourly_rate" {
  default = "0.25"  # Change to your rate
}

variable "cpu_hourly_rate" {
  default = "0.05"  # Change to your rate
}
```

Then apply: `terraform apply -auto-approve`

## Troubleshooting

| Issue | Symptom | Fix |
|-------|---------|-----|
| MLflow pod not ready | `kubectl get pods -n ml-training` shows pending | Wait 30s, check logs: `kubectl logs -n ml-training mlflow-server` |
| Training image not found | `ImagePullBackOff` error | Reload image: `kind load docker-image oppen-training-mnist:latest --name oppen-local` |
| Training job fails | Job status = failed | Check logs: `kubectl logs -n ml-training job/mnist-training-...` |
| Connection refused to MLflow | Agent tool returns connection error | Verify port-forward: `kubectl port-forward -n ml-training svc/mlflow-server 5000:5000` |
| Out of memory (OOM) | Pod killed unexpectedly | Reduce batch size: `trigger training job mnist with epochs=2 batch_size=32` |
| GPU not detected | Training on CPU only | Check nvidia-device-plugin: `kubectl get pods -n kube-system \| grep nvidia` |

## Terraform Resources Created

**Namespace**:
- `ml-training` - Isolated namespace for all ML workloads

**Deployment**:
- `mlflow-server` - MLflow tracking server (SQLite + emptyDir storage)

**Service**:
- `mlflow-server` - ClusterIP service (internal K8s access)

**ConfigMap**:
- `training-cost-rates` - GPU/CPU hourly rates

## Files Modified/Created

### New Files
- `pyproject.toml` - Project dependencies
- `scripts/training/train_mnist.py` - MNIST training script
- `scripts/training/train_llm.py` - LLM fine-tuning script
- `Dockerfile.training-mnist` - MNIST container
- `Dockerfile.training-llm` - LLM container
- `app/training/__init__.py` - Training module
- `app/training/mlflow_client.py` - MLflow API wrapper
- `app/training/job_manager.py` - K8s Job orchestration
- `terraform/mlflow.tf` - MLflow K8s resources
- `terraform/gpu_support.tf` - NVIDIA GPU plugin
- `terraform/cost_config.tf` - Cost configuration

### Modified Files
- `app/core/tools.py` - Added 5 training tools
- `app/core/agent.py` - Registered training tools
- `terraform/variables.tf` - Added cost rate variables

## Success Checklist

Phase 5 is complete when:
- ✅ `terraform apply` succeeds (MLflow running in K8s)
- ✅ `docker images | grep oppen-training` shows 2 images loaded
- ✅ `kubectl port-forward` reaches MLflow UI at http://localhost:5000
- ✅ Agent command `list mlflow experiments` returns (no experiments yet)
- ✅ Agent command `trigger training job mnist with 2 epochs` submits job
- ✅ Job completes: `kubectl get jobs -n ml-training` shows COMPLETIONS=1/1
- ✅ MLflow UI shows new "mnist-training" experiment with metrics
- ✅ Agent command `calculate training cost` returns cost breakdown

## Learning Outcomes

After Phase 5, you understand:
- **Model training orchestration** in Kubernetes
- **Experiment tracking** with MLflow
- **Cost attribution** for ML workloads
- **GPU resource scheduling** (optional)
- **Agent-driven ML workflows** (trigger → monitor → analyze)
- **Production containerization** for training scripts

## Next: Phase 6

Phase 6 will add:
- **Model registry** and versioning
- **Model serving** for inference
- **A/B testing** and canary deployments
- **Automated model promotion** workflows

See `docs/phase6_model_registry.md` for details.

## Quick Reference Commands

```bash
# Infrastructure
terraform apply -auto-approve                    # Deploy MLflow
kubectl get pods -n ml-training                  # Check pods
kubectl port-forward -n ml-training svc/mlflow-server 5000:5000 &  # MLflow UI

# Containers
docker build -t oppen-training-mnist:latest -f Dockerfile.training-mnist .
kind load docker-image oppen-training-mnist:latest --name oppen-local

# Agent
uv run -m app.main repl                          # Start REPL
# > trigger training job mnist with 2 epochs
# > get training status mnist-training-20260103-140000
# > calculate training cost mnist-training-20260103-140000

# Monitoring
kubectl logs -n ml-training -l job-name=mnist-training-20260103-140000 -f
kubectl get jobs -n ml-training
kubectl describe job mnist-training-20260103-140000 -n ml-training
```
