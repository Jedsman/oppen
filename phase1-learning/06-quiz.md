# Phase 1 Quiz: Test Your Knowledge

**Estimated Time**: 20 minutes
**Difficulty**: Easy

This quiz helps you validate your understanding of Phase 1 concepts. These are **self-check questions** - you decide if you can answer them. There's no scoring, just learning!

## Before You Start

✅ You've completed:
- [ ] Theory section (01-theory.md)
- [ ] Lab 1.1: Setup (02-lab1-setup.md)
- [ ] Lab 1.2: Cluster (03-lab2-cluster.md)
- [ ] Lab 1.3: App deployment (04-lab3-app.md)
- [ ] Lab 1.4: AI Agent (05-lab4-agent.md)

## Question 1: Kubernetes Basics

**What is the smallest deployable unit in Kubernetes?**

A) Container
B) Pod
C) Deployment
D) Service

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: B) Pod**

A Pod is the smallest unit you can deploy. It wraps one or more containers.
- Containers are images (templates)
- Pods run containers (instances)
- Deployments manage multiple Pods

**Real-world analogy:** If Docker is "one shipping container with your app inside", Kubernetes Pod is "the container running on a dock", and Deployment is "managing a fleet of these containers".

</details>

---

## Question 2: Control Plane vs Worker Nodes

**Which component decides which node gets which pod?**

A) kubelet
B) kube-scheduler
C) etcd
D) Container runtime

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: B) kube-scheduler**

The scheduler is part of the Control Plane that makes placement decisions.

**Quick Reference:**
- **Scheduler**: "Where should this Pod go?"
- **kubelet**: Executes the scheduler's decision ("Run this Pod")
- **etcd**: Stores the decision ("Pod should run here")
- **Container runtime**: Actually runs the container (Docker)

</details>

---

## Question 3: Service Networking

**Why do we need Services in Kubernetes?**

A) To store container images
B) To manage GPU resources
C) To provide stable network entry points to Pods
D) To delete pods automatically

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: C) To provide stable network entry points to Pods**

Pods are temporary (can crash, restart, get replaced during updates). Services provide a stable DNS name + IP that routes to the current Pods.

**Analogy:** Pods are like people (come and go), Services are like phone numbers (stay the same even if the person holding the phone changes).

**Example:**
```bash
kubectl create deployment nginx --image=nginx
# Creates Pods: nginx-abc123, nginx-def456, etc.
# These IPs change when Pods restart

kubectl expose deployment nginx --port=80
# Creates Service "nginx" with stable IP
# Always routes to current nginx Pods
```

</details>

---

## Question 4: Declarative vs Imperative

**What's the difference between these two approaches?**

**Imperative:**
```bash
kubectl create deployment nginx --image=nginx:alpine
```

**Declarative:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 1
  template:
    # ... pod spec ...
```

**Which approach is more production-ready?**

A) Imperative (faster to type)
B) Declarative (reproducible, versionable)
C) Both are equivalent
D) Neither, use cloud console instead

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: B) Declarative (reproducible, versionable)**

**Why Declarative is Better:**

| Aspect | Imperative | Declarative |
|--------|-----------|-------------|
| Versionable? | ❌ Only lives in bash history | ✅ Saved in git |
| Reproducible? | ❌ Depends on command typing | ✅ Same YAML = same result |
| Team collaboration? | ❌ Hard to review | ✅ Easy code review |
| Idempotent? | ❌ Running twice = trouble | ✅ Safe to run multiple times |

**Use Imperative for:** Quick testing, learning (like these labs)

**Use Declarative for:** Production, team projects, anything you'll maintain

</details>

---

## Question 5: What is Model Context Protocol (MCP)?

**In Lab 1.4, you connected an AI agent to Kubernetes. How did the agent access kubectl?**

A) The agent ran kubectl directly in a subprocess
B) The agent used an HTTP API to reach Kubernetes
C) The agent used MCP "tools" that wrapped kubectl
D) The agent used machine learning to guess pod names

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: A) The agent ran kubectl directly in a subprocess**

In Lab 1.4, your agent used Python's `subprocess` module to call kubectl:

```python
subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True)
```

This is simple and works great for learning.

**For context on MCP:**
- **MCP** (Model Context Protocol) is an emerging standard for AI agents to access tools
- Your Phase 1 agent doesn't use full MCP, just direct subprocess calls
- Phase 2 will introduce proper MCP servers for infrastructure tools
- By Phase 4, your agent will be fully autonomous with MCP integration

**Real MCP Tool Example (Phase 2+):**
```python
from mcp import Client

k8s_server = Client("@modelcontextprotocol/server-kubernetes")
pods = k8s_server.tools.list_pods()  # Uses MCP protocol
```

</details>

---

## Question 6: Scaling and Updates

**You ran these commands. In what order do they happen, and what's the outcome?**

```bash
kubectl create deployment my-app --image=my-app:v1
# (1 pod running)

kubectl scale deployment my-app --replicas=3
# (3 pods running)

kubectl set image deployment/my-app my-app=my-app:v2
# (pods update)
```

**What is the final state?**

A) 3 pods running v1.0, 0 pods running v2.0
B) 1 pod running v1.0, 2 pods running v2.0
C) 0 pods running v1.0, 3 pods running v2.0 ✅
D) 0 pods running (update deleted everything)

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: C) 0 pods running v1.0, 3 pods running v2.0**

**What happens step by step:**

```
Step 1: kubectl create deployment my-app --image=my-app:v1
Result: 1 pod (v1)
┌─────┐
│ v1  │
└─────┘

Step 2: kubectl scale deployment my-app --replicas=3
Result: 3 pods (all v1)
┌─────┬─────┬─────┐
│ v1  │ v1  │ v1  │
└─────┴─────┴─────┘

Step 3: kubectl set image deployment/my-app my-app=my-app:v2
Result: Rolling update (1 at a time)
  ┌─────┬─────┬─────┐
  │ v1  │ v1  │ v2  │  (1 new v2, 1 old v1 terminates)
  └─────┴─────┴─────┘
    ↓
  ┌─────┬─────┬─────┐
  │ v1  │ v2  │ v2  │  (2 new v2, 1 old v1 terminates)
  └─────┴─────┴─────┘
    ↓
  ┌─────┬─────┬─────┐
  │ v2  │ v2  │ v2  │  (All v2, zero downtime!)
  └─────┴─────┴─────┘
```

**Key:** Rolling updates = zero downtime deployments ✅

</details>

---

## Question 7: Troubleshooting

**You deploy a pod but `kubectl get pods` shows STATUS = CrashLoopBackOff. What does this mean?**

A) The pod is updating (status = in progress)
B) The pod crashed, Kubernetes is restarting it repeatedly
C) The pod is waiting for more resources
D) The pod is deliberately sleeping

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: B) The pod crashed, Kubernetes is restarting it repeatedly**

CrashLoopBackOff = "Container exited with error, restarting, exited with error, restarting..."

**How to debug:**

```bash
# See what's wrong
kubectl describe pod <pod-name>
# Look for "State:" and "Last State:"

# Check logs
kubectl logs <pod-name>
# See the error message

# Common causes:
# - Misconfigured container command
# - Port already in use
# - Missing required file/env var
# - Insufficient memory/CPU
```

**In this course:**
You probably won't see this because our apps (nginx, Python) are well-tested. But in real production, debugging CrashLoopBackOff is common.

</details>

---

## Question 8: AI Agent Understanding

**In your agent, what happens when you ask "What pods are running?"**

A) The agent has a hardcoded list of pod names
B) The LLM (Ollama) calls the `list_pods()` tool, which runs `kubectl get pods`
C) The agent connects to Kubernetes API directly with client library
D) The agent guesses based on container image names

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: B) The LLM (Ollama) calls the `list_pods()` tool, which runs `kubectl get pods`**

**The flow:**

```
"What pods are running?"
           ↓
     Ollama (LLM) thinks:
     "I need to use list_pods() tool"
           ↓
     Python calls: list_pods(namespace="default")
           ↓
     Python subprocess runs: kubectl get pods -n default
           ↓
     Returns: ["nginx-abc123", "nginx-def456"]
           ↓
     Ollama formats response:
     "There are 2 pods running: nginx-abc123, nginx-def456"
           ↓
     Displays to user
```

**This is "ReAct" pattern:**
1. **Reasoning**: LLM thinks about what to do
2. **Acting**: LLM calls tools
3. **Observing**: LLM sees results
4. **Repeat until done**

By Phase 4, your agent will do this autonomously without user prompts!

</details>

---

## Question 9: Kind Cluster Architecture

**Your Kind cluster runs inside Docker. What actually gets containerized?**

A) Only the Kubernetes control plane
B) Only the worker nodes
C) The entire Kubernetes cluster (control plane + nodes) inside Docker containers
D) Nothing, Kind doesn't use Docker

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: C) The entire Kubernetes cluster (control plane + nodes) inside Docker containers**

**Architecture:**

```
┌──────────────────────────────────┐
│    Docker Desktop / Docker        │
├──────────────────────────────────┤
│                                  │
│  ┌──────────────────────────┐   │
│  │ Container: oppen-lab-cp  │   │
│  │ (Control Plane)          │   │
│  │ - API Server             │   │
│  │ - Scheduler              │   │
│  │ - etc.                   │   │
│  └──────────────────────────┘   │
│                                  │
│  ┌──────────────────────────┐   │
│  │ Container: oppen-lab-w1  │   │
│  │ (Worker Node)            │   │
│  │ - Runs your Pods         │   │
│  └──────────────────────────┘   │
│                                  │
└──────────────────────────────────┘
```

**Why this is clever:**
- You get real Kubernetes without cloud infrastructure
- Perfect for learning and local development
- When you move to production (Phase 2+), same code works on real cloud Kubernetes!

</details>

---

## Question 10: What's Next After Phase 1?

**You've learned Kubernetes basics and built an AI agent. What comes in Phase 2+?**

A) More complex Kubernetes concepts only
B) Kubernetes + Infrastructure as Code + Autonomous agents
C) Learning a different orchestrator (Docker Swarm, Nomad)
D) Kubernetes cloud deployment (AWS EKS)

<details>
<summary>💡 Show Answer</summary>

**Correct Answer: B) Kubernetes + Infrastructure as Code + Autonomous agents**

**Phase 1 → Phase 2+ Progression:**

| Phase | Focus | Skills |
|-------|-------|--------|
| **Phase 1** (You are here) | K8s basics + agent foundation | kubectl, YAML, basic agent |
| **Phase 2** | IaC + agent expansion | Terraform, LangGraph state machines |
| **Phase 3** | Monitoring + diagnostics | Prometheus, agent reasoning |
| **Phase 4** | Self-healing loops | Autonomous remediation |
| **Phase 5** | ML training pipeline | MLflow, GPU scheduling |
| **Phase 6** | Cost optimization | Budget enforcement, job queues |
| **Phase 7** | AutoML | Hyperparameter tuning (Optuna) |
| **Phase 8** | Distributed training (simulation) | DDP concepts, multi-worker |
| **Phase 9** | Production distributed training | Kubeflow, PyTorchJob |

**Each phase builds on previous ones:**
```
Phase 1: Foundation (K8s + agent basics)
    ↓
Phase 2-4: Infrastructure Automation (agent controls infrastructure)
    ↓
Phase 5-7: ML Automation (agent trains models, optimizes)
    ↓
Phase 8-9: Advanced ML (distributed training at scale)
```

You're just getting started! 🚀

</details>

---

## Score Your Understanding

**Count how many questions you could answer (in your head or by checking):**

- [ ] 10/10 - Master! Ready for Phase 2 immediately
- [ ] 8-9/10 - Strong! Minor review recommended
- [ ] 6-7/10 - Good! Review theory sections
- [ ] 4-5/10 - Fair - Re-read key sections before Phase 2
- [ ] <4/10 - Review Phase 1 content thoroughly

## What to Review If Stuck

**Understanding unclear on:**

- **Pods, Deployments, Services?** → Re-read [01-theory.md](./01-theory.md) "Kubernetes Objects" section
- **How to run commands?** → Refer to quick reference at end of [04-lab3-app.md](./04-lab3-app.md)
- **How agent works?** → Re-run [05-lab4-agent.md](./05-lab4-agent.md) Lab 1.4 and ask questions
- **Tools/installation?** → Check [troubleshooting.md](./troubleshooting.md)

## Celebrating Phase 1 Completion! 🎉

You've accomplished:

✅ **Installed** Kind, kubectl, Ollama, Python packages
✅ **Created** a real Kubernetes cluster on your laptop
✅ **Deployed** an application (nginx) to Kubernetes
✅ **Exposed** service to access your app from browser
✅ **Built** an AI agent using LangGraph + Ollama
✅ **Connected** agent to query your cluster
✅ **Demonstrated** LLM tool use for infrastructure

**Phase 1 Summary:**
- 6-8 hours of learning
- Hands-on with real Kubernetes
- Understanding of K8s architecture
- AI agent foundation

## Ready for Phase 2?

**Phase 2: Infrastructure as Code & Agent Expansion**
- Learn Terraform to manage infrastructure
- Build more sophisticated agent workflows
- Add human-in-the-loop approval gates

→ **Next Phase**: [Phase 2 Overview](../phase2-learning/README.md)

---

## Quick Reference: Key Concepts

```
Kubernetes: Container orchestration platform
Pod: Smallest deployable unit (container wrapper)
Deployment: Manages multiple Pod replicas
Service: Stable network entry point to Pods
Control Plane: Decision maker (scheduling, monitoring)
Worker Node: Runs Pods, executes control plane decisions
kubectl: Command-line tool to talk to Kubernetes
ReAct: AI pattern (Reasoning → Acting → Observing)
Ollama: Local LLM (no API keys, private)
LangGraph: Framework for building agent workflows
```

## Feedback on Phase 1

What worked well? What was confusing? Did you get stuck anywhere?

The learning platform is continuously improved based on feedback. If you'd like to share:
- Found a confusing section?
- Typos or errors?
- Topics you'd like more depth on?

Feedback helps make Phase 2+ even better! 📝

---

**Congratulations on completing Phase 1! 🏆**

You're now ready to automate infrastructure with Phase 2. See you there! 🚀
