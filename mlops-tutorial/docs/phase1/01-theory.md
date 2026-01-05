# Theory: What is Kubernetes?

**Estimated Reading Time**: 30 minutes

## Overview

Kubernetes (K8s) is a production-grade **container orchestration platform**. In plain English: it manages dozens of containers across multiple machines, keeping your applications running 24/7 without human intervention.

Think of it like an **orchestra conductor**: individual containers are musicians, Kubernetes is the conductor who decides who plays when, what to do if someone misses a note, and how to keep the whole orchestra in sync.

## The Problem Kubernetes Solves

Imagine you've built a web application that consists of:
- 3 instances of your API server
- 2 instances of a database
- 1 cache service
- 1 logging service

Without Kubernetes, you'd need to:

**Manually manage all of this:**
```
❌ SSH into 6 different servers
❌ Start/stop containers on each machine
❌ Monitor if containers crash → manually restart them
❌ Load balance traffic across API instances
❌ Update containers without downtime
❌ Keep all servers configured identically
❌ Handle networking between containers on different machines
❌ Scale up/down when demand changes
```

**With Kubernetes, you describe your desired state:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
spec:
  replicas: 3  # "I want 3 copies running"
  # ... rest of config ...
```

✅ Kubernetes automatically:
- Starts 3 API server containers
- Restarts any that crash
- Load balances traffic between them
- Updates them without downtime
- Scales to 5 if needed, scales back to 2 later
- Moves containers between machines as needed

## The Key Insight: Declarative vs Imperative

**Traditional approach (Imperative)**:
"Do this, then do that, then do this other thing"
```bash
ssh server1
docker run my-api:v1 ...
ssh server2
docker run my-api:v1 ...
ssh server3
docker run my-api:v1 ...
```

**Kubernetes approach (Declarative)**:
"Here's what I want running"
```yaml
kind: Deployment
metadata:
  name: api-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: my-api:v1
```

Then: `kubectl apply -f deployment.yaml` → Kubernetes figures out what to do.

**Why is this powerful?**
- Easier to understand (no bash script complexity)
- Reproducible (same YAML always produces same result)
- Version controllable (track infrastructure in git)
- Idempotent (applying twice = same result as applying once)

## Kubernetes Architecture

Kubernetes splits into two main parts:

### Control Plane (Management)

The "brain" that makes decisions:

```
┌─────────────────────────────────────┐
│      Control Plane (Master)         │
├─────────────────────────────────────┤
│ API Server                          │
│ - Entry point for all commands      │
│ - kubectl talks to this             │
│                                     │
│ Scheduler                           │
│ - Decides which node gets which pod │
│ - Considers CPU, memory, affinity   │
│                                     │
│ Controller Manager                  │
│ - Monitors and repairs the system   │
│ - "Is replicas=3? Actually it's 2"  │
│ - Restarts crashed containers       │
│                                     │
│ etcd (database)                     │
│ - Stores ALL state                  │
│ - What pods exist, what config, etc │
└─────────────────────────────────────┘
```

### Worker Nodes (Execution)

The "muscles" that run your containers:

```
┌──────────────────────────────────┐
│        Worker Node (Machine)     │
├──────────────────────────────────┤
│                                  │
│  ┌─────────┐  ┌──────────┐      │
│  │ Pod     │  │ Pod      │      │
│  │ API v1  │  │ Cache    │      │
│  │ (cont)  │  │ (cont)   │      │
│  └─────────┘  └──────────┘      │
│                                  │
│  kubelet (watches control plane) │
│  - "Start this pod"              │
│  - "Stop that pod"               │
│                                  │
│  Container Runtime (Docker)      │
│  - Actually runs the containers  │
└──────────────────────────────────┘
```

**Communication Flow:**

```
User runs: kubectl apply -f app.yaml
                 ↓
        API Server (Control Plane)
                 ↓
        etcd: "Store this deployment"
                 ↓
        Scheduler: "Which node should this go to?"
                 ↓
        Node1, Node2, Node3: kubelet receives "run these pods"
                 ↓
        Containers start and stay running
                 ↓
        If container crashes → Controller Manager → Restart it
```

## Kubernetes Objects (Building Blocks)

### Pod (Smallest Unit)

A **Pod** is a wrapper around one or more containers.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  containers:
  - name: app
    image: python-server:latest
    ports:
    - containerPort: 8000
```

**Key Point**: Usually you don't create Pods directly. You use higher-level objects like Deployments.

### Deployment

A **Deployment** manages multiple Pod replicas.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
spec:
  replicas: 3  # Keep 3 copies running
  selector:
    matchLabels:
      app: api-server
  template:  # Template for creating Pods
    metadata:
      labels:
        app: api-server
    spec:
      containers:
      - name: api
        image: api-server:v1.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

**What this means:**
- Keep 3 Pod replicas running
- If one crashes → Restart it
- If all 3 are running → Do nothing
- To scale to 5: Change `replicas: 5` and apply again

### Service

A **Service** provides a stable network entry point to Pods.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: api-server
spec:
  selector:
    app: api-server  # Route to Pods with this label
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP  # Internal only, or LoadBalancer for external
```

**Why needed?**

Without Service:
```
Pod1 (IP: 10.0.1.5)
Pod2 (IP: 10.0.1.6)  ← Which one do I talk to?
Pod3 (IP: 10.0.1.7)
```

With Service:
```
Service "api-server" (stable IP: 10.0.2.1)
    ↓ Load balances ↓
Pod1 (IP: 10.0.1.5)
Pod2 (IP: 10.0.1.6)  ← Service handles routing
Pod3 (IP: 10.0.1.7)
```

**Service Types:**
- **ClusterIP**: Internal only (default)
- **NodePort**: Accessible on each node's IP
- **LoadBalancer**: Cloud provider assigns external IP
- **ExternalName**: Route to external DNS

### Namespace

A **Namespace** is a virtual cluster (logical partition).

```bash
kubectl get pods -n kube-system     # System namespace
kubectl get pods -n default         # Default namespace
kubectl get pods -n my-custom-ns    # Custom namespace
```

**Use cases:**
- Team separation (team-a, team-b namespaces)
- Environment separation (dev, staging, prod)
- Resource quotas per team

## Kubernetes vs Alternatives

### Docker Compose

**What it is**: Simple orchestration for single machine

**Pros:**
- Easy to learn
- Perfect for local development
- Simple YAML syntax

**Cons:**
- ❌ Only works on one machine
- ❌ No automatic restart/healing
- ❌ No load balancing
- ❌ Doesn't scale

**When to use**: Local dev environment

### Docker Swarm

**What it is**: Lightweight multi-machine orchestration

**Pros:**
- Simpler than Kubernetes
- Works across multiple machines

**Cons:**
- ❌ Less feature-rich
- ❌ Smaller ecosystem
- ❌ Fewer managed services available

**When to use**: Small clusters with simple requirements

### Kubernetes

**What it is**: Enterprise-grade orchestration

**Pros:**
- ✅ Works on any cloud or on-premises
- ✅ Scales from 1 to 1000s of machines
- ✅ Self-healing (auto-restart, auto-replacement)
- ✅ Rolling updates (zero-downtime deployments)
- ✅ Massive ecosystem (databases, message queues, monitoring, etc.)
- ✅ Standardized across clouds (AWS, Azure, GCP all use same Kubernetes)

**Cons:**
- More complex to learn
- Operational overhead
- Can be overkill for simple apps

**When to use**: Production ML/data systems, microservices, anything that must be reliable 24/7

## Kubernetes Workflow (How It Actually Works)

### Step 1: Describe (Declarative YAML)

You create a file describing what you want:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: my-app:v1.0
```

### Step 2: Apply (Send to Control Plane)

```bash
kubectl apply -f deployment.yaml
```

This sends to the API Server: "I want a Deployment with 2 replicas of my-app"

### Step 3: Store (etcd records state)

etcd records: "Deployment my-app wants 2 replicas"

### Step 4: Schedule (Scheduler makes decisions)

Scheduler decides:
- "Pod 1 → Node 1"
- "Pod 2 → Node 2"

### Step 5: Execute (Kubelet runs containers)

kubelet on each node receives: "Run this Pod"

Docker on each node runs: `docker run my-app:v1.0`

### Step 6: Monitor (Controller Manager maintains state)

Controller Manager continuously checks:
- "Does etcd say 2 replicas?" (Answer: yes)
- "Are 2 Pods actually running?"
  - If yes: Do nothing (desired state = actual state)
  - If no: Fix it (restart crashed pod, create missing pod)

### Step 7: Update (Rolling deployment)

To update to v1.1:

```yaml
# deployment.yaml (change image)
image: my-app:v1.1
```

```bash
kubectl apply -f deployment.yaml
```

Kubernetes does **rolling update** by default:
1. Start 1 new Pod (v1.1)
2. Stop 1 old Pod (v1.0)
3. Start 1 new Pod (v1.1)
4. Stop 1 old Pod (v1.0)

Result: Zero-downtime update ✅

## Why Kubernetes for Machine Learning?

### Traditional ML Setup (Manual)

```bash
# Developer's laptop
python train.py --epochs 100 --batch-size 32
# Wait 2 hours...
# Manually copy model to production server
# Manually update serving code
# Cross fingers that it works
```

### Kubernetes ML Setup (Automated)

```bash
# Local machine
kubectl apply -f training-job.yaml
# Kubernetes automatically:
# - Schedules on available GPU nodes
# - Tracks with MLflow
# - Saves checkpoints to shared storage
# - When done, updates serving model
# - Rolls out new predictions automatically
# - Monitors for degradation
```

**Benefits:**
- ✅ Reproducible experiments
- ✅ Track training metrics/costs
- ✅ Auto-scaling based on data
- ✅ Multi-experiment parallelization
- ✅ Cost monitoring and optimization
- ✅ Production models identical to training models

## Key Concepts Summary

| Concept | What It Is | Example |
|---------|-----------|---------|
| **Pod** | Smallest deployable unit (1+ containers) | Single nginx container |
| **Deployment** | Manages multiple Pod replicas | 3 copies of nginx, auto-restart if crashes |
| **Service** | Stable network entry point | DNS name that routes to all Pods |
| **Namespace** | Virtual cluster / logical partition | Separate dev/prod environments |
| **Node** | Physical/virtual machine in cluster | Linux server in data center |
| **Control Plane** | Decision-making system | Schedules Pods, repairs failures |
| **kubelet** | Agent on each node | Executes Control Plane decisions |
| **kubectl** | Command-line tool | Talk to Kubernetes cluster |

## What's Next

Now you understand the **concepts**. In the hands-on labs, you'll:

1. **Lab 1.1**: Install tools (Kind, kubectl)
2. **Lab 1.2**: Create a real Kubernetes cluster on your laptop
3. **Lab 1.3**: Deploy nginx and access it in your browser
4. **Lab 1.4**: Connect an AI agent that can query your cluster

The concepts you've learned will suddenly "click" when you see them in action.

## Quick Reference: Common kubectl Commands

```bash
# View resources
kubectl get pods              # List all pods
kubectl get deployments       # List all deployments
kubectl get services          # List all services
kubectl get nodes             # List all nodes

# Describe (detailed info)
kubectl describe pod my-pod   # Details about a pod
kubectl describe deployment my-app

# Logs
kubectl logs my-pod           # Print container logs
kubectl logs -f my-pod        # Follow logs in real-time

# Create/Update
kubectl apply -f app.yaml     # Apply YAML file
kubectl apply -f config/      # Apply all files in directory

# Delete
kubectl delete pod my-pod     # Delete a pod
kubectl delete -f app.yaml    # Delete resources from file

# Port forward (local access)
kubectl port-forward svc/my-service 8080:8000  # localhost:8080 → service:8000

# Execute command in pod
kubectl exec -it my-pod -- bash  # Interactive shell in pod
```

## Common Questions

**Q: Do I need to know Docker well to use Kubernetes?**

A: Not deeply. You need to understand:
- Containers are like lightweight VMs
- Images are templates, containers are running instances
- You can run Docker locally first to verify your app works

**Q: Can I run Kubernetes on my laptop?**

A: Yes! That's what Kind (Kubernetes in Docker) does. It runs Kubernetes inside Docker on your machine.

**Q: Do I need GPUs to learn Kubernetes?**

A: No. GPUs are for training models faster. Kubernetes works fine on CPU-only for learning.

**Q: Is Kubernetes free?**

A: Kubernetes itself is open-source (free). You only pay for cloud infrastructure (servers, storage, bandwidth) to run it on.

## Troubleshooting Theory Content

**Having trouble understanding Pods vs Deployments?**
- Think: Pod = single container, Deployment = fleet of containers with management
- In practice, always use Deployment (it creates Pods for you)

**Confused about Services?**
- Service = stable DNS name that never changes
- Pods = come and go (can crash, restart, get updated)
- Service abstracts away these details

**Wondering why Kubernetes instead of just Docker?**
- Docker runs 1 machine
- Kubernetes runs many machines
- Kubernetes = reliability + automation for production

---

**Ready to get your hands dirty?** → Next: [Lab 1.1: Set Up Your Environment](./02-lab1-setup.md)
