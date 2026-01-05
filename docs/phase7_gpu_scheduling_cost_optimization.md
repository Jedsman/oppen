# Phase 7: GPU Scheduling & Cost Optimization

## Objective
Master GPU resource optimization, implement cost-aware scheduling, and explore trade-offs between performance and cost. This phase focuses on practical cost reduction strategies.

## 1. GPU Utilization Monitoring
Understand actual GPU usage patterns.

### Steps:
1. **NVIDIA GPU Metrics**:
   - Deploy `nvidia-gpu-prometheus-exporter` to expose GPU metrics.
   - Key metrics:
     - `nvidia_smi_utilization_gpu`: GPU compute utilization (%).
     - `nvidia_smi_memory_used_mb`: GPU memory consumption.
     - `nvidia_smi_temperature_gpu`: GPU temperature.
2. **Prometheus Queries**:
   - Create dashboards for utilization per pod, namespace, and model.
   - Query examples:
     ```promql
     avg(nvidia_smi_utilization_gpu) by (pod_namespace)
     max(nvidia_smi_memory_used_mb) by (pod_name)
     ```
3. **Grafana Visualization**:
   - Dashboard: "GPU Utilization by Workload".
   - Identify underutilized GPUs (utilization < 30%).

## 2. GPU Sharing & Time-Slicing
Maximize GPU utilization by sharing GPUs across multiple workloads.

### Option A: NVIDIA Multi-Process Service (MPS)
- Allows multiple processes to share GPU memory and compute.
- Suitable for inference workloads (lower latency sensitivity).

### Steps:
1. **Enable MPS**:
   - Configure GPU nodes to run NVIDIA MPS daemon.
   - Set in Terraform: GPU node pool with MPS enabled.
2. **Pod Configuration**:
   - Set pod limits to fraction of GPU:
     ```yaml
     nvidia.com/gpu: "0.5"  # Share GPU between 2 pods
     ```
3. **Testing**:
   - Run two inference workloads on same GPU.
   - Measure throughput and latency impact.

### Option B: Time-Slicing (GPU Operator)
- NVIDIA GPU Operator can time-slice GPUs at kernel level.
- More flexible but higher overhead.

### Steps:
1. **Deploy GPU Operator**:
   ```bash
   helm repo add nvidia https://nvidia.github.io/gpu-operator
   helm install gpu-operator nvidia/gpu-operator -n gpu-operator-system
   ```
2. **Configure Time-Slicing** in `device-plugin-config.yaml`:
   ```yaml
   sharing:
     timeSlicing:
       replicas: 4  # 4 pods can share 1 GPU
   ```

## 3. Cost Attribution & Tracking
Assign costs to workloads for chargeback and optimization.

### Steps:
1. **Cost Model Definition**:
   - Define resource costs (hourly rates):
     - GPU: $0.35/hour (A100), $0.15/hour (T4), $0.10/hour (CPU-only).
     - Memory: $0.01/GB/hour.
     - CPU: $0.05/core/hour.
   - Store in ConfigMap for easy updates.
2. **Cost Calculation**:
   - Create a sidecar or job that calculates per-pod costs:
     ```python
     pod_cost = (gpu_hours * gpu_rate) + (cpu_hours * cpu_rate) + (memory_gb * memory_rate)
     ```
3. **Cost Logging to Prometheus**:
   - Export custom metric: `pod_hourly_cost_usd`.
   - Label by: pod name, namespace, model, workload type.
4. **MLflow Integration** (from Phase 5):
   - Log cost per experiment/training run.
   - Calculate ROI: `(model_improvement%) / cost`.

## 4. Cost Optimization Strategies
Implement and measure cost reduction techniques.

### Strategy 1: GPU Type Selection
- Experiment with different GPUs for same task.
- Compare: T4 (cheap) vs V100 vs A100 (fast).
- Metric: `cost per inference` vs `latency`.

**Terraform Example**:
```hcl
variable "gpu_type" {
  default = "nvidia.com/gpu-t4"  # Toggle between t4, v100, a100
}
```

### Strategy 2: Batch Size Optimization
- Larger batches → lower cost per sample, but higher latency.
- Test: Batch size 1, 4, 8, 16, 32.
- Metric: `cost per 1000 inferences` vs `p99 latency`.

**Experiment in Phase 5 Training Script**:
```python
for batch_size in [1, 4, 8, 16, 32]:
    # Train/infer, measure cost and latency
    mlflow.log_metric(f"batch_{batch_size}_cost_usd", cost)
    mlflow.log_metric(f"batch_{batch_size}_latency_ms", latency)
```

### Strategy 3: Model Quantization
- Reduce model precision: FP32 → FP16 → INT8.
- Benefits: Smaller memory, faster inference, lower GPU requirements.
- Trade-off: Slight accuracy loss.

**Implementation**:
```python
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# Log quantized model to MLflow
mlflow.pytorch.log_model(quantized_model, "model_quantized_int8")
# Compare inference cost and latency vs original
```

### Strategy 4: Auto-Scaling Based on Demand
- Scale down GPUs during low-traffic periods.
- Use K8s Horizontal Pod Autoscaler (HPA) with custom metrics.

**Terraform/K8s HPA**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: torchserve
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Strategy 5: Request Prioritization & Queueing
- Batch low-priority requests offline.
- Process high-priority requests immediately (higher cost).
- Metric: Cost savings from batching vs latency trade-off.

## 5. Cost Optimization Dashboard
Create a comprehensive cost visibility dashboard.

### Grafana Panels:
1. **Total Cost Trends**: Line chart of daily/weekly costs.
2. **Cost by Workload**: Pie chart (training, inference, monitoring).
3. **Cost by Model**: Bar chart showing top cost consumers.
4. **GPU Utilization**: Scatter plot (GPU utilization % vs cost/hour).
5. **Cost Savings Potential**: Highlight underutilized GPUs.

### Example Queries:
```promql
# Daily cost per namespace
sum(increase(pod_hourly_cost_usd[24h])) by (pod_namespace)

# GPU utilization efficiency (lower is better)
pod_hourly_cost_usd / (nvidia_smi_utilization_gpu / 100)

# Identify wasteful pods (high cost, low utilization)
pod_hourly_cost_usd > 0.1 AND nvidia_smi_utilization_gpu < 30
```

## 6. Agent Integration (LangGraph)
Extend the agent to optimize costs autonomously.

### New Agent Nodes:
1. **Cost Analysis Node**:
   - Agent queries Prometheus: "Show me the most expensive workloads".
   - Reports: "Training jobs cost $50/day, 40% GPU utilization".
2. **Optimization Proposal Node**:
   - Agent suggests: "Quantize model-v4 to INT8 (saves $0.05/inference, 2% accuracy loss)".
   - Or: "Switch from T4 to A100 for batch training (saves $20/day, 30% faster)".
3. **Cost Simulation Node**:
   - Agent estimates savings before applying changes.
   - "Enabling GPU time-slicing saves $15/day (20% reduction) with <10ms latency increase".
4. **Execution Node** (optional, human-in-loop):
   - Agent applies optimization (resize pod, change model config).
   - Tracks cost impact over time.

## 7. Simulation Run: Cost Optimization Challenge
Test cost reduction strategies.

### Scenario:
1. Run baseline: Inference workload with batch size 1, FP32 model, T4 GPU.
   - Measure: cost, latency, accuracy.
2. Agent proposes optimization 1: Increase batch size to 8.
   - Measure: cost savings, latency impact.
3. Agent proposes optimization 2: Quantize to FP16.
   - Measure: cost savings, accuracy impact.
4. Agent proposes optimization 3: Switch GPU type (T4 → A100).
   - Compare ROI across strategies.
5. Dashboard: Show cumulative cost savings and trade-offs.

## 8. Cost Anomaly Detection
Detect unexpected cost spikes.

### Steps:
1. **Baseline Calculation**:
   - Calculate expected daily cost from historical data (7-day rolling average).
2. **Anomaly Alert**:
   - Alert if actual cost > baseline + 20%.
   - Example: "Daily cost spike: Expected $40, actual $55 (37% increase)".
3. **Agent Investigation**:
   - Agent diagnoses cause: "3 new GPU training jobs started unexpectedly".
   - Proposes action: "Limit parallel training jobs to 1 (reduces cost back to $42)".

## Deliverables
- NVIDIA GPU metrics exported to Prometheus.
- GPU sharing configured (MPS or time-slicing) with cost impact measured.
- Cost attribution model implemented per pod/workload.
- 5 cost optimization experiments with before/after metrics.
- Grafana dashboard showing GPU utilization, costs, and efficiency.
- Agent nodes for cost analysis and optimization proposals.
- Cost anomaly detection and alerting.
- Documentation: GPU optimization checklist (batch size, quantization, GPU type selection).

## Learning Outcomes
- GPU resource utilization and optimization.
- Cost modeling and attribution in cloud/local infrastructure.
- Trade-offs between performance, cost, and model accuracy.
- Practical optimization strategies (quantization, batching, GPU sharing).
- Autonomous cost optimization using agents and observability data.
- Cost-aware infrastructure decisions and business impact analysis.
