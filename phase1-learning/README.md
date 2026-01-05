# Phase 1: Kubernetes Foundations - Learning Path

Welcome to Phase 1 of the Zero-to-Hero ML/LLMOps tutorial!

This phase will take you from zero to understanding Kubernetes, running a local cluster, and connecting an AI agent to automate your infrastructure.

## What You'll Learn (6-8 hours)

- ✅ Kubernetes architecture and core concepts
- ✅ Run a production-like local cluster with Kind
- ✅ Deploy and manage applications with kubectl
- ✅ Connect an AI agent to Kubernetes via Model Context Protocol (MCP)

## Prerequisites

Before starting, make sure you have:
- **Docker Desktop** or **Rancher Desktop** installed and running
- **Terminal/CLI** comfort (basic commands: cd, ls, cat)
- **Python 3.10+** installed
- **2-3 hours** of uninterrupted time (or break into multiple sessions)

## Learning Path

1. **[Theory: What is Kubernetes?](./01-theory.md)** (30 min)
   - Container orchestration explained
   - K8s vs Docker Compose
   - Architecture overview

2. **[Lab 1.1: Set Up Your Environment](./02-lab1-setup.md)** (1 hour)
   - Install Kind, kubectl, helm
   - Verify installations

3. **[Lab 1.2: Create Your First Cluster](./03-lab2-cluster.md)** (45 min)
   - Create a Kind cluster
   - Explore cluster structure

4. **[Lab 1.3: Deploy Your First App](./04-lab3-app.md)** (1 hour)
   - Deploy nginx to Kubernetes
   - Expose service
   - Access from browser

5. **[Lab 1.4: Connect Agent to Kubernetes](./05-lab4-agent.md)** (2 hours)
   - Install Ollama (local LLM)
   - Set up MCP servers
   - Build a basic agent
   - Agent queries Kubernetes

6. **[Quiz: Test Your Knowledge](./06-quiz.md)** (20 min)
   - 5 self-check questions
   - Validate your understanding

## Estimated Time Breakdown

| Section | Time | Difficulty |
|---------|------|------------|
| Theory | 30 min | Easy |
| Lab 1.1 (Setup) | 1 hour | Easy |
| Lab 1.2 (Cluster) | 45 min | Medium |
| Lab 1.3 (App) | 1 hour | Medium |
| Lab 1.4 (Agent) | 2 hours | Hard |
| Quiz | 20 min | Easy |
| **Total** | **6-8 hours** | |

## How to Use This Tutorial

### On Your Tablet/Browser (Read)
- Read theory sections
- Review lab instructions
- Take quiz
- Reference commands

### On Your Laptop (Practice)
- Install tools
- Run commands
- Build the cluster
- Execute labs

### Validation Checkpoints
Each lab has validation checkpoints (✅). These help you confirm you're on the right track:
- Check commands work correctly
- Verify outputs match expectations
- Ensure resources created successfully

## What You'll Build

By the end of Phase 1, you'll have:

1. **A running Kind cluster** on your machine
   ```
   oppen-lab (Kubernetes cluster)
   └─ kube-system (system pods)
   └─ default (your namespace)
      └─ nginx deployment
   ```

2. **kubectl mastery** - command-line control of Kubernetes

3. **An AI agent** that can:
   - List pods and deployments
   - Describe Kubernetes resources
   - Execute kubectl commands via natural language

## Next Steps After Phase 1

Once you complete Phase 1:
- Explore the Kubernetes dashboard
- Try creating other deployments (postgres, redis)
- Experiment with scaling
- Ready for Phase 2: Infrastructure as Code with Terraform

## Troubleshooting

See [troubleshooting.md](./troubleshooting.md) for common issues:
- Docker not running
- Kind cluster creation fails
- kubectl not found
- Agent connection issues

## Getting Help

- Stuck on a step? Re-read the lab carefully
- Check the troubleshooting guide
- Review the theory section
- Take a break and come back with fresh eyes

---

**Ready to start?** Open [01-theory.md](./01-theory.md) to begin! 🚀

**Estimated time to completion**: 6-8 hours
**Difficulty level**: Beginner → Intermediate
**Hardware needed**: 4GB RAM, 20GB disk space
