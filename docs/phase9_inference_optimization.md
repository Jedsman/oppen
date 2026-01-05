# Phase 9: Inference Optimization & Scaling

## Objective
Optimize model inference for production scale: maximize throughput, minimize latency and cost, and handle multi-tenancy. Master inference-specific optimizations separate from training.

## 1. Inference Frameworks & Serving
Choose and deploy an optimized inference framework.

### Framework Comparison:
| Framework | Best For | Latency | Throughput | Cost | Complexity |
|-----------|----------|---------|-----------|------|------------|
| **TorchServe** | PyTorch models | Low-mid | High | Low | Low |
| **vLLM** | LLMs (text generation) | Low | Very High | Low | Mid |
| **ONNX Runtime** | Model interoperability | Very Low | Very High | Low | Low |
| **TensorFlow Serving** | TensorFlow models | Mid | High | Mid | Mid |
| **Ray Serve** | Multi-model, complex routing | Mid | High | Mid | High |

### Steps:
1. **Select Framework**:
   - For LLMs: Use **vLLM** (optimized for batching).
   - For general PyTorch: Use **TorchServe** or **ONNX Runtime**.
   - For multi-model complex routing: Use **Ray Serve**.

2. **Baseline Deployment**:
   - Deploy chosen framework on K8s (1 replica, 1 GPU).
   - Measure baseline latency, throughput, cost.
   - Establish as performance baseline for Phase 9.

## 2. Batching Strategies
Increase throughput by processing multiple requests together.

### Static vs Dynamic Batching:

**Static Batching** (simpler):
- Fixed batch size (e.g., 32).
- Pros: Predictable latency, high throughput.
- Cons: Latency spikes if batch not full, lower GPU utilization.

**Dynamic Batching** (complex but better):
- Variable batch size with max-wait timeout.
- Collect requests up to max batch size or timeout (e.g., 100ms), then process.
- Pros: Low latency, high GPU utilization.
- Cons: More complex, latency variability.

### Implementation:

**vLLM Dynamic Batching**:
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b",
    tensor_parallel_size=1,
    max_model_len=2048,
    # Dynamic batching
    max_num_seqs=128,  # Max batch size
    enable_prefix_caching=True,  # Optimize for repeated inputs
)
```

**TorchServe Batching**:
```yaml
# config.properties
batch_size=32
max_batch_delay=100  # Wait up to 100ms for batch to fill
```

### Experimentation:
1. Vary batch size: 1, 4, 8, 16, 32, 64.
2. Measure per experiment:
   - Throughput (requests/sec).
   - Latency p50, p95, p99.
   - Cost per 1000 requests.
3. Log to MLflow:
   ```python
   mlflow.log_metric(f"batch_{size}_throughput_req_sec", throughput)
   mlflow.log_metric(f"batch_{size}_latency_p99_ms", latency_p99)
   mlflow.log_metric(f"batch_{size}_cost_usd", cost)
   ```

## 3. Request Queuing & Load Balancing
Handle burst traffic efficiently.

### Steps:
1. **Queue Configuration**:
   - Set max queue size (prevent memory exhaustion).
   - Configure timeout for queued requests (reject if too old).
   - Metrics: Queue length, wait time.

2. **Load Balancing Strategy**:
   - **Round-Robin**: Simple, even distribution.
   - **Least Connections**: Route to least-busy replica.
   - **Token Bucket**: Rate limit by user/model.

3. **Auto-Scaling Based on Queue Depth**:
   - Scale up if queue_length > threshold.
   - Scale down if queue_length < threshold for sustained period.
   - K8s HPA with custom metric:
     ```yaml
     metrics:
     - type: Pods
       pods:
         metric:
           name: request_queue_length
         target:
           type: AverageValue
           averageValue: "30"  # Scale if avg queue > 30
     ```

## 4. Model Optimization Techniques
Reduce model size and inference latency.

### Technique 1: Quantization
**FP32 → FP16 → INT8 trade-offs**.
- FP16: 2x faster, 2x smaller, imperceptible accuracy loss.
- INT8: 4x faster, 4x smaller, ~1% accuracy loss.

**Implementation** (PyTorch):
```python
import torch
import torch.quantization

# Quantize to INT8
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Compare inference
with torch.no_grad():
    out_fp32 = model(x)
    out_int8 = model_int8(x)

# Log accuracy diff
accuracy_loss = (out_fp32 - out_int8).abs().mean()
mlflow.log_metric("int8_accuracy_loss", accuracy_loss)
```

### Technique 2: Pruning
Remove unnecessary parameters.
- Remove 30-50% of weights with < 1% accuracy loss.
- Reduces memory and latency.

**Implementation**:
```python
import torch.nn.utils.prune as prune

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)  # Prune 30%
        prune.remove(module, 'weight')  # Make pruning permanent

# Measure latency improvement
```

### Technique 3: Knowledge Distillation
Train a small "student" model to mimic a large "teacher" model.
- Student: 10x faster, smaller.
- Teacher accuracy → Student performance (95%+ of teacher accuracy).

**Integration with Phase 5**:
```python
# Phase 5 training script: distillation mode
def distillation_loss(student_out, teacher_out, temperature=3.0):
    return torch.nn.KLDivLoss()(
        F.log_softmax(student_out / temperature, dim=1),
        F.softmax(teacher_out / temperature, dim=1)
    ) * temperature ** 2

# Train student with distillation loss
```

### Technique 4: Lazy Loading & Caching
- Load model weights on-demand.
- Cache hot models in memory.
- Reduces startup latency for cold models.

## 5. Latency Optimization
Minimize inference time.

### Analysis:
1. **Profile Inference** (PyTorch Profiler):
   ```python
   with torch.profiler.profile() as prof:
       output = model(input)

   prof.key_averages().table(sort_by="cpu_time_total")
   ```
   Identify bottleneck layers.

2. **Layer Fusion**:
   - Fuse multiple operations (Conv+BatchNorm+ReLU → single op).
   - vLLM and ONNX Runtime do this automatically.

3. **Reduce Sequence Length** (for LLMs):
   - Shorter prompts = lower latency.
   - Truncate non-critical context.

4. **GPU Kernels**:
   - Use optimized implementations (FlashAttention for transformers).
   - vLLM uses FlashAttention by default.

## 6. Multi-Tenancy & Resource Isolation
Serve multiple customers/models with fair resource sharing.

### Steps:
1. **Namespace Isolation**:
   - Separate K8s namespaces per tenant (inference-tenant-a, inference-tenant-b).
   - Resource quotas per tenant.

2. **Model Assignment**:
   - Assign models to tenants (Model A → Tenant 1, Model B → Tenant 2).
   - Or share models but isolate quotas.

3. **QoS (Quality of Service)**:
   - Priority classes: Premium (SLA 50ms p99), Standard (100ms p99), Batch (no SLA).
   - Process premium requests first.

4. **Cost Allocation**:
   - Track per-tenant costs (GPU hours, requests).
   - Chargeback to teams.
   - Example:
     ```python
     tenant_cost = (gpu_hours_used * gpu_rate) / num_tenants_sharing_gpu
     ```

## 7. Agent Integration
Extend agent to optimize inference autonomously.

### New Agent Nodes:
1. **Inference Profiler Node**:
   - Agent triggers latency profiling.
   - Identifies bottlenecks: "Layer 3 (attention) uses 60% of time".
   - Proposes optimization: "Enable FlashAttention (save 30% latency)".

2. **Batch Tuning Node**:
   - Agent experiments with batch sizes.
   - Reports: "Batch 16: 200 req/s, p99=45ms, cost=$0.001/req".
   - Recommends: "Use batch 8 for SLA-critical requests, batch 32 for batch jobs".

3. **Quantization Evaluation Node**:
   - Agent quantizes model and measures accuracy loss.
   - Proposes: "INT8 quantization saves 60% inference cost, 0.5% accuracy loss".

4. **Scaling Recommendation Node**:
   - Agent analyzes traffic patterns and SLA requirements.
   - Recommends: "2 replicas serve 95% of traffic, 3rd replica for spikes".

## 8. Simulation Run: Inference Optimization Challenge
Test optimization across dimensions.

### Scenario:
1. **Baseline**: Deploy model with FP32, batch size 1, 1 replica.
   - Throughput: 10 req/s, latency p99: 200ms, cost: $0.01/req.

2. **Optimization 1: Quantization**.
   - Agent: "Quantize to FP16".
   - Result: 20 req/s (+100%), p99: 120ms (-40%), cost: $0.006/req (-40%), 0.2% accuracy loss.

3. **Optimization 2: Batching**.
   - Agent: "Use batch size 8".
   - Result: 150 req/s (+650%), p99: 80ms (-33%), cost: $0.001/req (-83%).

4. **Optimization 3: Pruning**.
   - Agent: "Prune 30% of weights".
   - Result: 200 req/s (+33%), p99: 60ms (-25%), cost: $0.0008/req (-20%), 0.8% accuracy loss.

5. **Final Dashboard**: Show cumulative improvements.
   - Throughput: 10 → 200 req/s (20x).
   - Cost: $0.01 → $0.0008/req (12.5x).
   - Latency: 200ms → 60ms (3.3x).
   - Accuracy: 96% → 95% (acceptable trade-off).

## 9. Serving Framework Comparison Experiment
Compare frameworks on same workload.

### Experiment:
- Deploy same model (Llama-2-7B) on: TorchServe, vLLM, ONNX Runtime.
- Load test: 100 concurrent requests.
- Measure: Latency p99, throughput, memory, CPU.
- Cost: Which framework is cheapest for this workload?
- Log results to MLflow for reproducibility.

## Deliverables
- Inference serving framework deployed (TorchServe, vLLM, or ONNX).
- Batching configuration (static/dynamic) with throughput benchmarks.
- Queue management and load balancing (K8s Service).
- Auto-scaling HPA based on queue depth or latency.
- Quantization benchmark (FP32 vs FP16 vs INT8).
- Pruning and knowledge distillation experiments with MLflow tracking.
- Latency profiling and optimization (layer-wise analysis).
- Multi-tenancy setup with resource quotas and cost allocation.
- Agent nodes for profiling, optimization recommendations, and scaling.
- Comprehensive simulation: 20x throughput, 12.5x cost reduction, minimal accuracy loss.

## Learning Outcomes
- Inference-specific optimizations vs training optimizations.
- Quantization, pruning, distillation trade-offs.
- Batching and queuing for throughput optimization.
- Production inference serving frameworks and their trade-offs.
- Multi-tenancy and fair resource sharing.
- Latency analysis and optimization techniques.
- Cost per inference and throughput optimization.
