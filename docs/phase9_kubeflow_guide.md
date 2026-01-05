# Phase 9: Kubeflow PyTorchJob Integration Guide

## Overview

Phase 9 transitions from Phase 8's educational DDP simulation to **production-grade distributed training** using Kubernetes and the Kubeflow Training Operator.

**What is Kubeflow?**
- Open-source ML platform for Kubernetes
- Training Operator provides PyTorchJob custom resource
- Automatically orchestrates multi-GPU/multi-node distributed training
- Auto-injects environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)

**PyTorchJob vs Kubernetes batch/v1 Job:**

| Aspect | batch/v1 Job | PyTorchJob |
|--------|--------------|-----------|
| Use Case | Single-node training | Distributed training |
| Replicas | 1 pod | Master + Workers |
| Env Vars | Manual setup | Auto-injected by Kubeflow |
| Backend | N/A | NCCL (GPU) or Gloo (CPU) |
| Scaling | Limited | 2-1000+ workers |
| Preferred For | Development | Production |

---

## Prerequisites

### 1. Kubernetes Cluster
You need an accessible Kubernetes cluster (local Kind, GKE, EKS, AKS, etc.)

### 2. Kubeflow Training Operator

Install the Training Operator:
```bash
# Latest version
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=master"

# Verify installation
kubectl get crd | grep kubeflow
# Should show: pytorchjobs.kubeflow.org
```

**For Kind (local testing):**
```bash
kind create cluster
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=master"
kubectl wait --for=condition=ready pod -l app=training-operator -n kubeflow --timeout=300s
```

### 3. kubectl Access
```bash
# Test access
kubectl get nodes
kubectl get namespaces
```

### 4. Docker Registry (for image storage)
- Docker Hub account, or
- Private registry (Harbor, ECR, GCR), or
- Local Kind registry

---

## Quick Start

### Step 1: Build the Distributed Training Image

```bash
# Build locally
docker build -t oppen-training-mnist-ddp:latest -f Dockerfile.training-mnist-ddp .

# For Kind cluster
kind load docker-image oppen-training-mnist-ddp:latest

# For remote cluster, push to registry
docker tag oppen-training-mnist-ddp:latest <registry>/oppen-training-mnist-ddp:latest
docker push <registry>/oppen-training-mnist-ddp:latest
```

### Step 2: Submit a Distributed Training Job via Agent

```
Agent: "Submit a 2-worker distributed training job for MNIST"
submit_distributed_training(model_type="mnist", world_size=2, epochs=3, lr=0.001, batch_size=64)
```

Or directly:
```bash
# Using kubectl
kubectl apply -f - <<EOF
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: mnist-ddp-2w
  namespace: ml-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: oppen-training-mnist-ddp:latest
            imagePullPolicy: Never
            args: ["--epochs", "3", "--lr", "0.001", "--batch-size", "64"]
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-server.ml-training.svc.cluster.local:5000"
            - name: GPU_HOURLY_RATE
              valueFrom:
                configMapKeyRef:
                  name: training-cost-rates
                  key: gpu_hourly_rate
            resources:
              limits:
                nvidia.com/gpu: "1"
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: oppen-training-mnist-ddp:latest
            imagePullPolicy: Never
            args: ["--epochs", "3", "--lr", "0.001", "--batch-size", "64"]
            env:
            - name: MLFLOW_TRACKING_URI
              value: "http://mlflow-server.ml-training.svc.cluster.local:5000"
            resources:
              limits:
                nvidia.com/gpu: "1"
EOF
```

### Step 3: Monitor the Job

```bash
# Watch job status
kubectl get pytorchjobs -n ml-training -w

# Check pod status
kubectl get pods -n ml-training -l pytorch-job-name=mnist-ddp-2w

# View logs (all workers)
kubectl logs -n ml-training -l pytorch-job-name=mnist-ddp-2w -f

# Agent tool
get_distributed_status("mnist-ddp-2w")
```

### Step 4: Check Results

```bash
# MLflow UI (only rank 0 logs)
mlflow ui
# http://localhost:5000 → mnist-training experiment

# Model checkpoint (in training pod)
kubectl exec -it <pod-name> -n ml-training -- ls -la ./outputs/
```

---

## Agent Tools Reference

### 1. submit_distributed_training()

**Purpose**: Submit a PyTorchJob to Kubernetes cluster

**Signature**:
```python
submit_distributed_training(
    model_type: str = "mnist",        # "mnist" or "llm"
    world_size: int = 2,               # Workers (2-8)
    epochs: int = 3,                   # Training epochs
    lr: float = 0.001,                 # Learning rate
    batch_size: int = 64,              # Batch per worker
    gpu_per_replica: int = 1           # GPUs per worker
) -> str
```

**Returns**: Job submission status with monitoring commands

**Examples**:
```
# Quick 2-worker test (20 min)
submit_distributed_training(model_type="mnist", world_size=2, epochs=1, lr=0.001)

# Full 4-worker training (30 min)
submit_distributed_training(model_type="mnist", world_size=4, epochs=3, lr=0.001)

# Custom configuration
submit_distributed_training(world_size=8, batch_size=128, gpu_per_replica=2)
```

### 2. get_distributed_status()

**Purpose**: Get real-time status and logs from a PyTorchJob

**Signature**:
```python
get_distributed_status(job_name: str) -> str
```

**Example**:
```
get_distributed_status("mnist-ddp-4w-20240115-143022")
```

**Output**:
```
[INFO] Distributed Job Status: mnist-ddp-4w-20240115-143022

Status: running
Master: active=1 succeeded=0 failed=0
Workers: 3 total, 0 succeeded

Recent Logs:
...
```

### 3. estimate_distributed_speedup()

**Purpose**: Estimate training time and cost for different worker counts

**Signature**:
```python
estimate_distributed_speedup(
    model_type: str = "mnist",
    epochs: int = 3,
    world_sizes: str = "[1,2,4,8]"
) -> str
```

**Example**:
```
# Compare 1, 2, 4, 8 workers
estimate_distributed_speedup(model_type="mnist", world_sizes="[1,2,4,8]")

# Just compare 2 vs 4 workers
estimate_distributed_speedup(world_sizes="[2,4]")
```

**Output** (example for MNIST, 3 epochs):
```
[INFO] Distributed Training Speedup Estimation
Model: MNIST
Epochs: 3
==========================================================================================
Config               Duration        Cost (USD)      Speedup         Efficiency
------------------------------------------------------------------------------------------
single-node          15.0m           $0.21           1.0x            100%
2-worker             10.7m           $0.30           1.4x            70%
4-worker             5.4m            $0.38           2.8x            70%
8-worker             2.7m            $0.54           5.6x            70%
------------------------------------------------------------------------------------------

Note: Assumes 70% scaling efficiency (typical for distributed training)
```

---

## Architecture Deep Dive

### PyTorchJob Manifest Structure

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: <job-name>
  namespace: ml-training
spec:
  pytorchReplicaSpecs:
    Master:                    # Master pod (rank 0)
      replicas: 1              # Always 1
      restartPolicy: OnFailure
      template:
        spec:
          containers: [...]    # Container spec
    Worker:                    # Worker pods (rank 1, 2, ...)
      replicas: N-1            # world_size - 1
      restartPolicy: OnFailure
      template:
        spec:
          containers: [...]    # Same container spec
```

### Environment Variables (Auto-Injected by Kubeflow)

Kubeflow automatically injects these into each pod:

| Variable | Master | Worker | Value |
|----------|--------|--------|-------|
| RANK | 0 | 1, 2, ... | Global rank |
| WORLD_SIZE | 2 | 2 | Total workers |
| LOCAL_RANK | 0 | 0 | Rank within pod |
| MASTER_ADDR | localhost | master-pod-ip | Master address |
| MASTER_PORT | 29500 | 29500 | Master port |

**train_mnist_ddp.py uses these to initialize:**
```python
def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0  # Single-node fallback

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # NCCL for GPU, Gloo for CPU
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",  # Read from environment
        rank=rank,
        world_size=world_size
    )

    return rank, world_size, local_rank
```

### Master vs Worker Pods

**Master Pod (Rank 0)**:
- First pod to start
- Waits for all workers to connect
- Only rank 0 logs to MLflow (prevents duplicates)
- Outputs final metrics and model

**Worker Pods (Rank 1, 2, ...)**:
- Start after master
- Connect to MASTER_ADDR:MASTER_PORT
- Synchronize gradients during backward pass
- Quiet output (no MLflow logging)

### Gradient Synchronization

```
Epoch 1, Batch 1:
  All Ranks: Forward pass → loss
  All Ranks: Backward pass → gradients
  ↓↓↓ (Synchronization via NCCL/Gloo)
  All Ranks: Update model with averaged gradients

Result: All replicas have identical model weights
```

---

## Examples

### Example 1: 2-Worker MNIST (Quick Test)

**Time**: ~15-20 minutes
**Resources**: 2 GPUs or 2 CPU cores
**Accuracy**: ~97-98%

```
Agent: "Test 2-worker distributed training on MNIST"
submit_distributed_training(model_type="mnist", world_size=2, epochs=1)
```

**What happens**:
1. Creates 1 Master pod + 1 Worker pod
2. Each processes 30k samples (half of 60k)
3. Gradients synchronized every batch
4. Rank 0 logs to MLflow, worker stays quiet
5. ~15 min: Job completes with same accuracy as single-node

### Example 2: 4-Worker Production Training

**Time**: ~30 minutes
**Resources**: 4 GPUs or 4 CPU cores
**Accuracy**: ~98-99%

```
Agent: "Run production 4-worker distributed training"
submit_distributed_training(model_type="mnist", world_size=4, epochs=3, lr=0.001, batch_size=64)
```

**What happens**:
1. Creates 1 Master + 3 Worker pods
2. Each processes 15k samples (quarter of 60k)
3. 3 epochs × 4 workers = 900 batches total
4. 2.8x speedup (vs single-node)
5. MLflow tracks: params (world_size=4), metrics (accuracy per epoch)

### Example 3: Cost-Optimized Configuration

**Goal**: Train LLM with limited budget

```
# First estimate costs
estimate_distributed_speedup(model_type="llm", world_sizes="[1,2,4]")

# Output shows 2-worker is most cost-efficient
# Submit with optimal config
submit_distributed_training(model_type="llm", world_size=2, epochs=5, batch_size=32)
```

---

## Integration with Previous Phases

### Phase 5 (Training Infrastructure) Integration
- **Training scripts**: train_mnist_ddp.py adds DDP support (same MLflow tracking)
- **Compatibility**: Works with existing cost calculation
- **Extension**: Adds RANK-based logging (only rank 0 writes)

### Phase 6 (Job Queue) Integration
- **Distributed jobs in queue**: QueuedJob now has world_size, distributed, gpu_per_replica
- **Cost estimation**: Multiplies cost by world_size (e.g., 2-worker = 2× cost)
- **Scheduling**: Mixed workloads (single-node + distributed) schedule independently
- **Example**:
  ```python
  job_queue.submit_job(
      job_id="ddp-mnist-01",
      name="4-worker MNIST",
      model_type="mnist",
      world_size=4,
      distributed=True,
      epochs=3
  )
  ```

### Phase 7 (AutoML) Integration
- **Distributed hyperparameter search**: Each Optuna trial can specify world_size
- **Faster convergence**: Distributed trials complete in 1/N time
- **Trial metadata**: OPTUNA_TRIAL_NUMBER env var passes through (for experiment tracking)
- **Future**: Distributed Optuna with multi-worker optimization

### Phase 8 (DDP Simulation) → Phase 9 (Production) Transition
- **Same code pattern**: train_mnist_ddp.py auto-detects RANK env var
- **Simulation without RANK**: Runs single-node (Phase 8 fallback)
- **Production with RANK**: Runs distributed (Phase 9)
- **Educational flow**: Learn concepts in simulation → Deploy in production

---

## Troubleshooting

### Problem: PyTorchJob Not Starting

**Symptom**: Job stays in "Pending" state

**Diagnosis**:
```bash
kubectl describe pytorchjob mnist-ddp-2w -n ml-training
# Look for "Events" section
```

**Common Causes**:
1. **Image not found**: `imagePullPolicy: Never` on remote cluster
   - Solution: Use `imagePullPolicy: Always` or push image to registry

2. **Insufficient resources**: Not enough GPUs available
   - Solution: Reduce world_size or gpu_per_replica
   - Check: `kubectl describe nodes | grep -A 5 "Allocated resources"`

3. **Training Operator not installed**
   - Solution: Install Kubeflow: `kubectl apply -k "github.com/kubeflow/training-operator/..."`

### Problem: NCCL Initialization Timeout

**Symptom**: "Timeout: gloo not available"

**Cause**: NCCL backend requires:
- NVIDIA GPUs
- NCCL library
- GPU peer access

**Solution**:
1. Verify GPU availability: `nvidia-smi` on node
2. Check NCCL: `python -c "import torch; print(torch.version.nccl)"`
3. **Fallback to Gloo** (CPU-friendly):
   ```python
   # In train_mnist_ddp.py, change:
   backend = "nccl" if torch.cuda.is_available() else "gloo"
   ```

### Problem: Only Rank 0 Logging to MLflow

**Symptom**: Missing metrics from workers

**Cause**: Intentional design! Only rank 0 logs to prevent duplicates

**Why**:
- 4 workers → 4× identical metrics (confusing MLflow)
- Only rank 0 (master) has authoritative values
- Workers' metrics calculated identically (same model, same data shard)

**Verification**:
```bash
# MLflow should show 1 run per job, not 4
mlflow ui
# Look for "mnist-training" experiment
# Should have 1 run per PyTorchJob (not 4)
```

**If you need worker logs**:
```bash
kubectl logs <worker-pod-name> -n ml-training
```

### Problem: Effective Batch Size Mismatch

**Symptom**: Different accuracy vs single-node training

**Cause**: Batch size interacts with world_size

**Single-node**: 64 batch → updates per epoch = 60000/64 = 937
**4-worker**: Each gets 64 batch (15k samples each) → updates per epoch = (15000/64)×4 = 937 ✓

**Solution**: Keep batch_size consistent, let Kubeflow handle sharding

---

## Cost Considerations

### Cost Model

For distributed training:
```
Total Cost = (GPU cost per worker + CPU cost per worker) × world_size × duration_hours

GPU cost per worker = 1 GPU × $0.25/hour = $0.25/hour
CPU cost per worker = 4 cores × $0.05/core/hour = $0.20/hour
Per-worker-hour = $0.45

Example:
  2-worker, 30 minutes:
  - Duration per worker: 30 min = 0.5 hours (due to speedup, actual runtime is ~20 min)
  - Speedup factor: 2 × 0.7 = 1.4x
  - Effective duration: 30/1.4 = 21.4 min = 0.357 hours
  - Total cost: $0.45 × 2 workers × 0.357 hours = $0.32
```

### Optimization Tips

1. **Choose optimal world_size**: Use estimate_distributed_speedup() first
   - Too many workers: Communication overhead > speedup
   - Too few workers: Limited speedup

2. **Batch size tuning**:
   - Smaller batches (32): Less stable but more updates
   - Larger batches (256): More stable but fewer updates

3. **Resource sharing**: Colocate training and inference on same node when possible

---

## Next Steps

### Immediate (This Phase)
1. Build training image: `docker build -f Dockerfile.training-mnist-ddp .`
2. Test 2-worker job: `submit_distributed_training(world_size=2, epochs=1)`
3. Monitor: `get_distributed_status(job_name)`
4. Check MLflow: Verify only 1 run (from rank 0)

### Future (Phase 9 Follow-up)
1. Add train_llm_ddp.py for LLM distributed training
2. Multi-node PyTorchJob (GPU cluster orchestration)
3. Distributed AutoML (Optuna with multiple trials on multiple workers)
4. Real-time distributed monitoring dashboard

### Advanced (Phase 10+)
1. Kubeflow Pipelines: Multi-stage ML workflows
2. Distributed data preprocessing (Apache Spark integration)
3. Model serving at scale (KServe)
4. Federated learning (private multi-party training)

---

## Commands Summary

```bash
# Build image
docker build -t oppen-training-mnist-ddp:latest -f Dockerfile.training-mnist-ddp .

# Load into Kind
kind load docker-image oppen-training-mnist-ddp:latest

# Submit job (via agent)
submit_distributed_training(model_type="mnist", world_size=2, epochs=3)

# Monitor
kubectl get pytorchjobs -n ml-training -w
kubectl logs -n ml-training -l pytorch-job-name=<job-name> -f

# Get status (via agent)
get_distributed_status("<job-name>")

# Estimate costs
estimate_distributed_speedup(world_sizes="[1,2,4]")

# Cleanup
kubectl delete pytorchjob <job-name> -n ml-training
```

---

## Key Differences: Phase 8 vs Phase 9

| Aspect | Phase 8 (Simulation) | Phase 9 (Production) |
|--------|--------------------|--------------------|
| Hardware | 1 GPU or CPU | Multi-GPU cluster |
| Process Model | Python multiprocessing | Kubernetes pods |
| Backend | Gloo (simulated) | NCCL (real GPU sync) |
| Environment | Local machine | Kubernetes cluster |
| Speedup | 1.0x (no parallelism) | 1.4-2.8x (actual) |
| Use Case | Learning concepts | Production training |
| Code | train_mnist_ddp_sim.py | train_mnist_ddp.py |
| Training Time | 45s (4 workers) | 15-30m (4 GPUs) |

---

**Phase 9 Ready!** You now have production-grade distributed training with Kubernetes orchestration. Ready for Phase 10+ (advanced ML ops)?
