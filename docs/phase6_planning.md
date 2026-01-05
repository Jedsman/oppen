# Phase 6: GPU Scheduling & Cost Optimization - Implementation Plan

## Vision

Phase 6 elevates the training infrastructure to production-grade with intelligent resource scheduling, cost optimization, and advanced observability. This phase targets ML/LLMOps expertise development with focus on:
- GPU allocation efficiency
- Cost forecasting and optimization
- Resource contention management
- Workload prioritization
- Real-time monitoring and alerting

## Phase 6 Scope

### 1. GPU Scheduling & Management (Tier 1 - High Value)
**Goal**: Optimize GPU allocation and prevent resource waste

#### 1.1 GPU Device Plugin Enhancement
- [ ] Install NVIDIA device plugin with proper configuration
- [ ] Add GPU metrics discovery (memory, compute capability)
- [ ] Implement GPU topology awareness (multi-GPU support)
- [ ] Add GPU request/limit validation in job_manager.py
- [ ] Create GPU availability checker tool for agent

#### 1.2 Resource Quotas & Limits
- [ ] Create ResourceQuota per experiment (e.g., 1 GPU max per team)
- [ ] Implement LimitRange for safe defaults (CPU, memory, GPU)
- [ ] Add quota warning system (alert when 80% consumed)
- [ ] Create quota calculator tool for agent
- [ ] Document quota allocation strategy

#### 1.3 Pod Priority & Preemption
- [ ] Define PriorityClasses (Urgent, Normal, Background)
- [ ] Assign priorities to job types (LLM > MNIST > validation)
- [ ] Implement preemption: background jobs yield to urgent
- [ ] Create priority decision tool for agent workflows
- [ ] Add priority explanation to training cost calculation

**Output**: GPU scheduling system that prevents resource conflicts and prioritizes critical jobs

---

### 2. Cost Optimization Engine (Tier 1 - High Value)
**Goal**: Reduce training costs while maintaining performance

#### 2.1 Dynamic Cost Calculation
- [ ] Replace static cost rates with time-based pricing
  - Off-peak (22:00-06:00): 50% discount
  - Peak (06:00-22:00): Full price
- [ ] Add spot instance simulation (70% cheaper, 5% interruption rate)
- [ ] Implement resource efficiency metrics (cost per accuracy point)
- [ ] Track cost trend over time (improving/degrading)

#### 2.2 Cost Optimization Recommender
- [ ] Analyze training runs for inefficiencies:
  - Early stopping opportunities
  - Batch size optimization
  - Learning rate tuning impact
  - GPU utilization percentage
- [ ] Generate cost-saving recommendations:
  - "Use batch_size=128 instead of 32 (20% faster, same accuracy)"
  - "Schedule non-urgent jobs during off-peak hours"
  - "GPU utilization only 30% - consider using CPU"
- [ ] Create agent tool: `recommend_cost_optimization(run_id)`

#### 2.3 Budget & Forecasting
- [ ] Track cumulative training costs per experiment
- [ ] Implement budget alerts (warn at 70%, block at 100%)
- [ ] Forecast costs for remaining epochs
  - "Current pace: $5/epoch. Continuing 5 more epochs: $25 total"
- [ ] Create forecasting tool for agent: `forecast_training_cost(job_name, remaining_epochs)`

**Output**: Cost optimization engine that identifies saving opportunities and prevents budget overruns

---

### 3. Monitoring & Observability (Tier 2 - Medium Value)
**Goal**: Real-time visibility into training performance and resource health

#### 3.1 Metrics Collection (Prometheus)
- [ ] Deploy Prometheus to ml-training namespace
- [ ] Export metrics from training containers:
  - Training metrics (loss, accuracy, perplexity)
  - Resource metrics (CPU, memory, GPU %)
  - Cost metrics (GPU hours, CPU hours, total cost)
  - System metrics (queue depth, job duration)
- [ ] Scrape MLflow metrics API every minute
- [ ] Store 15-day retention (balance disk vs coverage)

#### 3.2 Visualization (Grafana)
- [ ] Deploy Grafana with Prometheus datasource
- [ ] Create dashboards:
  - **Training Overview**: Active jobs, completion rate, avg duration
  - **Resource Usage**: GPU %, memory %, CPU % across jobs
  - **Cost Dashboard**: Cost per job, cost trends, budget consumption
  - **Job Queue**: Pending jobs, average wait time, priority distribution
- [ ] Set up dashboard sharing/access control

#### 3.3 Alerting Rules
- [ ] Alert when GPU utilization < 30% (inefficient usage)
- [ ] Alert when job duration > 2x historical average
- [ ] Alert when memory pressure > 80%
- [ ] Alert when cost forecast exceeds budget
- [ ] Alert when job fails or crashes
- [ ] Send alerts to agent for auto-remediation capability

**Output**: Real-time monitoring system with cost tracking and performance insights

---

### 4. Job Queue & Batch Scheduling (Tier 2 - Medium Value)
**Goal**: Manage multiple training jobs without resource conflicts

#### 4.1 Job Queue Manager
- [ ] Implement FIFO queue for job submissions
- [ ] Add job prioritization (urgent jobs skip queue)
- [ ] Schedule based on resource availability:
  - Don't start new job if GPU utilization > 50%
  - Don't start if would exceed budget
- [ ] Create agent tool: `queue_training_job(model, params, priority=Normal, run_name)`
- [ ] Add job status tracking: Queued → Scheduled → Running → Completed

#### 4.2 Batch Mode (Submit Multiple Jobs)
- [ ] Create batch job manifest (list of training configs)
- [ ] Implement sequential vs parallel execution modes
- [ ] Add checkpointing (resume interrupted batch)
- [ ] Create agent tool: `submit_batch_training(jobs=[...], parallel=False)`

**Output**: Job queue system that optimizes resource utilization and prevents conflicts

---

### 5. Advanced Agent Tools (Tier 2 - Medium Value)
**Goal**: Empower agent to make intelligent training decisions

#### 5.1 New Training Tools
```python
# GPU & Resource Management
get_gpu_availability() → Available GPUs, memory, queued jobs
validate_job_resources(model, epochs, batch_size) → Feasible? Cost estimate?
check_training_quota(team) → Used/limit, remaining budget

# Cost Optimization
recommend_cost_optimization(run_id) → List of optimization suggestions
forecast_training_cost(job_name, remaining_epochs) → Cost estimate
compare_training_configs(config_list) → Cost/performance trade-offs

# Job Management
queue_training_job(model, priority, name) → Queued, position in queue
get_job_queue_status() → Pending jobs, ETAs, total queue cost
cancel_training_job(job_id, reason) → Cancel with cleanup

# Monitoring
get_training_health_status() → System health, alerts, performance
generate_cost_report(start_date, end_date) → Monthly/quarterly costs
get_resource_utilization_report() → GPU %, memory %, efficiency
```

#### 5.2 Agentic Workflows
```
User: "Train LLM for 5 epochs with budget limit of $5"
Agent workflow:
  1. forecast_training_cost("llm", 5 epochs) → "$6.50 estimated"
  2. get_gpu_availability() → "1 GPU available in 2 min"
  3. Agent response: "Exceeds budget by $1.50. Options:
     - Use smaller batch size (reduce cost 20%)
     - Use spot instances (70% cheaper, risky)
     - Train 3 epochs instead ($3.90)
     Which do you prefer?"
  4. User chooses smaller batch_size
  5. queue_training_job("llm", epochs=5, batch_size=32)
  6. Agent monitors and alerts if exceeds $5
```

---

### 6. Production Deployment Patterns (Tier 3 - Nice to Have)
**Goal**: Ready for multi-user, multi-team production environment

#### 6.1 Multi-Tenancy
- [ ] Implement namespace per team (e.g., team-a, team-b)
- [ ] ResourceQuota per team (GPU, memory, cost budgets)
- [ ] RBAC: teams can only see/manage own jobs
- [ ] Shared MLflow (all teams log to same server, filtered views)

#### 6.2 Model Registry & Versioning
- [ ] Fix MLflow model logging (use file-based artifacts instead of registry)
- [ ] Track model lineage (training run → model version)
- [ ] Model promotion workflow: Dev → Staging → Production
- [ ] Create agent tools: `register_model(run_id, name, version)`

#### 6.3 Kubernetes Integration
- [ ] Custom Resource Definition (CRD) for TrainingJob
- [ ] Operator to manage training job lifecycle
- [ ] Webhook for cost validation (prevent expensive jobs)
- [ ] Integration with cluster autoscaling (scale GPU nodes)

---

## Implementation Roadmap

### Week 1-2: GPU Scheduling (Tier 1)
```
Day 1-2:   GPU device plugin + resource quotas
Day 3-4:   Pod priority & preemption
Day 5:     Testing & validation
```

### Week 3-4: Cost Optimization (Tier 1)
```
Day 1-2:   Dynamic cost calculation + recommender
Day 3-4:   Budget tracking & forecasting
Day 5:     Agent tools integration
```

### Week 5-6: Monitoring (Tier 2)
```
Day 1-2:   Prometheus + metric collection
Day 3-4:   Grafana dashboards
Day 5:     Alerting rules
```

### Week 7-8: Job Queue (Tier 2)
```
Day 1-2:   Job queue manager implementation
Day 3-4:   Batch scheduling
Day 5:     Agent tools
```

### Week 9+: Advanced Features (Tier 3)
```
Multi-tenancy, model registry, K8s operators
```

---

## Success Criteria

### GPU Scheduling
- [ ] Jobs with different GPU requirements don't conflict
- [ ] Pod priority system prevents resource starvation
- [ ] GPU utilization > 70% on average
- [ ] No failed job due to "insufficient GPU"

### Cost Optimization
- [ ] Cost per training run < Phase 5 baseline (with same accuracy)
- [ ] Agent can recommend 5+ cost-saving strategies
- [ ] Budget forecasting accuracy > 90% within first epoch
- [ ] Off-peak scheduling reduces costs 30-40%

### Monitoring
- [ ] All metrics collected with < 1min latency
- [ ] Cost dashboard shows accurate totals
- [ ] Alerts triggered within 2min of anomaly
- [ ] Historical trend data retained for analysis

### Job Queue
- [ ] No resource conflicts between concurrent jobs
- [ ] Queue processing time < 5 min average
- [ ] Batch jobs complete without manual intervention
- [ ] Priority system works correctly (urgent jobs start first)

### Agent Integration
- [ ] Agent can make cost-aware training decisions
- [ ] Agent tools have < 500ms response time
- [ ] Natural language commands ("train cheaper") understood
- [ ] Agent provides clear recommendations with confidence

---

## Technical Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| GPU Scheduling | K8s Device Plugin | Native K8s GPU support |
| Resource Quotas | K8s ResourceQuota | Built-in, no external deps |
| Job Queue | Custom Python + K8s API | Lightweight, control over logic |
| Metrics | Prometheus | Industry standard |
| Visualization | Grafana | Rich dashboards, free tier |
| Alerting | Prometheus AlertManager | Integrated with Prometheus |
| Cost Tracking | Custom + MLflow | Precise control over calcs |

---

## Estimated Effort

| Task | Effort | Priority |
|------|--------|----------|
| GPU scheduling | 20 hours | High |
| Cost optimization | 25 hours | High |
| Monitoring (Prometheus + Grafana) | 15 hours | Medium |
| Job queue | 20 hours | Medium |
| Advanced agent tools | 15 hours | Medium |
| Multi-tenancy | 15 hours | Low |
| Documentation | 10 hours | High |
| **Total** | **~120 hours** | - |

---

## Learning Outcomes

By completing Phase 6, you will understand:

1. **GPU Orchestration**
   - How Kubernetes schedules GPU resources
   - Device plugins and resource discovery
   - Pod priority and preemption policies

2. **Cost Engineering**
   - Cost allocation and forecasting
   - Optimization techniques (spot instances, scheduling)
   - Cost-performance trade-offs

3. **Observability**
   - Metrics collection and aggregation
   - Dashboard design for operational insights
   - Alerting and incident response

4. **Job Scheduling**
   - Queue management algorithms
   - Resource-aware scheduling
   - Batch processing patterns

5. **Production Patterns**
   - Multi-tenancy implementation
   - RBAC and namespace isolation
   - Kubernetes operators (optional)

---

## Dependencies on Phase 5

Phase 6 builds directly on Phase 5:
- ✅ MLflow experiment tracking (store metrics)
- ✅ Training scripts with metrics logging (data source)
- ✅ Agent tools foundation (extend with new tools)
- ✅ K8s ml-training namespace (add monitoring)
- ✅ Cost calculation basics (enhance with optimization)

---

## Optional: Phase 7 - Advanced Topics

Future phases could cover:
- **Distributed Training**: Multi-GPU, multi-node synchronization
- **Model Serving**: KServe for inference + A/B testing
- **AutoML**: Hyperparameter optimization (Ray Tune, Optuna)
- **Data Pipeline**: Data versioning, feature stores
- **Compliance**: Cost auditing, compliance reporting

---

## Getting Started with Phase 6

```bash
# Verify Phase 5 is stable
kubectl get pods -n ml-training
kubectl get jobs -n ml-training

# Check MLflow has data
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Ready to start Phase 6!
# Begin with GPU scheduling (highest value, cleanest implementation)
```

---

**Phase 6 will transform the training infrastructure from a simple pipeline into a production-grade ML/LLMOps platform capable of managing real-world training workloads at scale.**
