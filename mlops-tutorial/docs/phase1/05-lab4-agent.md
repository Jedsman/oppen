# Lab 1.4: Connect Agent to Kubernetes

**Estimated Time**: 2 hours
**Difficulty**: Hard
**Goal**: Build an AI agent that can query your Kubernetes cluster

## Before You Start

✅ Verify you completed Lab 1.3:
- [ ] nginx deployment running: `kubectl get deployment nginx`
- [ ] Cluster healthy: `kubectl get nodes` shows STATUS = Ready
- [ ] You understand Services and Deployments from previous labs

## Overview: What You're Building

```
┌─────────────────┐
│   Your Terminal │
│  (You type questions)
└────────┬────────┘
         │
    "What pods are running?"
         │
         ▼
┌─────────────────────────────┐
│   AI Agent (Python/LangGraph)│
│  + Ollama (LLM engine)      │
│  + MCP Servers (K8s tools)  │
└────────┬────────────────────┘
         │
    Uses kubectl tools
         │
         ▼
┌─────────────────────────────┐
│    Kubernetes Cluster       │
│  (Returns pod list)         │
└─────────────────────────────┘
```

## Part 1: Install Ollama (Local LLM)

Ollama runs a large language model locally on your machine. No API keys, no internet required, completely private.

### Install Ollama

**macOS:**
```bash
# Download from https://ollama.com (~600MB)
# Or install via Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
```
Download from https://ollama.com/download/OllamaSetup.exe
Run installer
```

### Start Ollama Service

After installation, start the Ollama service:

**macOS/Linux:**
```bash
# In background
ollama serve &

# Or in foreground (keep terminal open)
ollama serve
```

**Windows:**
```
Ollama starts automatically when installed
Or manually start from Start Menu → Ollama
```

**Verify it's running:**
```bash
curl http://localhost:11434/api/tags
```

**Expected output (even if empty):**
```json
{"models":[]}
```

✅ If you get a response, Ollama is running!

### Download a Model

Ollama downloads language models on demand. For this lab, use `llama3.2:3b` (lightweight):

```bash
ollama pull llama3.2:3b
```

**What happens:**
- Downloads ~2GB model file
- Stores in ~/.ollama/models/
- ⏱️ Takes 2-10 minutes depending on internet speed

**Expected output:**
```
pulling manifest
pulling 6a5b7d0e...
pulling 2e...
verifying sha256 digest
writing manifest
success
```

### Verify Model Downloaded

```bash
ollama list
```

**Expected output:**
```
NAME               ID              SIZE     MODIFIED
llama3.2:3b        6f4a5c4e9e5d   2.0 GB   2 hours ago
```

✅ Model is ready!

### Test Ollama Direct Chat

```bash
ollama run llama3.2:3b
```

Then ask a question:
```
>>> What is Kubernetes?

Kubernetes is a container orchestration platform that helps manage containerized
applications across multiple machines...

>>> /bye
```

✅ Ollama is working! (Press Ctrl+D or type /bye to exit)

## Part 2: Set Up Your Python Environment

Create a Python project for your agent:

```bash
# Create project directory
mkdir -p ~/oppen-agent
cd ~/oppen-agent

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Install Required Packages

```bash
cat > requirements.txt <<EOF
langchain-ollama==0.1.1
langgraph==0.0.44
langchain==0.1.20
anthropic==0.25.1
pydantic==2.6.0
EOF

pip install -r requirements.txt
```

**⏱️ This takes 1-2 minutes.**

### Verify Installation

```bash
python3 -c "from langchain_ollama import ChatOllama; print('✅ Installation successful!')"
```

✅ If no errors, packages installed correctly!

## Part 3: Create Kubernetes Tools

Your agent needs tools to query Kubernetes. Create a tools module:

```bash
cat > k8s_tools.py <<'EOF'
import subprocess
import json
from langchain_core.tools import tool

@tool
def list_pods(namespace: str = "default") -> str:
    """List all pods in a namespace"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        pods = json.loads(result.stdout)
        pod_names = [p["metadata"]["name"] for p in pods["items"]]

        if not pod_names:
            return f"No pods found in namespace '{namespace}'"

        return f"Pods in {namespace}:\n" + "\n".join(f"  - {name}" for name in pod_names)
    except Exception as e:
        return f"Error listing pods: {str(e)}"

@tool
def get_pod_details(pod_name: str, namespace: str = "default") -> str:
    """Get details about a specific pod"""
    try:
        result = subprocess.run(
            ["kubectl", "describe", "pod", pod_name, "-n", namespace],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return f"Pod not found: {result.stderr}"

        return result.stdout
    except Exception as e:
        return f"Error getting pod details: {str(e)}"

@tool
def list_deployments(namespace: str = "default") -> str:
    """List all deployments in a namespace"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "deployments", "-n", namespace, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        deployments = json.loads(result.stdout)
        dep_info = []
        for d in deployments["items"]:
            name = d["metadata"]["name"]
            replicas = d["spec"]["replicas"]
            ready = d["status"]["readyReplicas"] or 0
            dep_info.append(f"  - {name}: {ready}/{replicas} ready")

        if not dep_info:
            return f"No deployments found in namespace '{namespace}'"

        return f"Deployments in {namespace}:\n" + "\n".join(dep_info)
    except Exception as e:
        return f"Error listing deployments: {str(e)}"

@tool
def get_cluster_info() -> str:
    """Get cluster information (nodes, version)"""
    try:
        # Get nodes
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        nodes = json.loads(result.stdout)
        node_count = len(nodes["items"])

        # Get version
        version_result = subprocess.run(
            ["kubectl", "version", "--short"],
            capture_output=True,
            text=True,
            timeout=10
        )

        version = version_result.stdout if version_result.returncode == 0 else "unknown"

        return f"""Cluster Information:
  Nodes: {node_count}
  Version: {version}
  Ready to deploy applications!"""
    except Exception as e:
        return f"Error getting cluster info: {str(e)}"

# Export all tools
TOOLS = [list_pods, get_pod_details, list_deployments, get_cluster_info]
EOF
```

### Test the Tools

```bash
python3 << 'EOF'
import subprocess
import json

# Test kubectl
result = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ kubectl works")
else:
    print("❌ kubectl failed:", result.stderr)

# Test python import
try:
    from k8s_tools import TOOLS
    print(f"✅ Loaded {len(TOOLS)} Kubernetes tools")
except Exception as e:
    print(f"❌ Failed to load tools: {e}")
EOF
```

✅ Both should work!

## Part 4: Build the Agent

Create your AI agent that uses these tools:

```bash
cat > agent.py <<'EOF'
#!/usr/bin/env python3
"""
Kubernetes AI Agent
Connects to your local Kubernetes cluster via LangGraph + Ollama
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from k8s_tools import TOOLS

def main():
    print("🤖 Kubernetes AI Agent")
    print("=" * 50)
    print("Connected to: oppen-lab cluster")
    print("Model: llama3.2:3b (local)")
    print("=" * 50)
    print()

    # Initialize LLM
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        base_url="http://localhost:11434"
    )

    # Create agent with all K8s tools
    agent = create_react_agent(llm, TOOLS)

    print("Available commands:")
    print("  'What pods are running?'")
    print("  'List all deployments'")
    print("  'Tell me about my cluster'")
    print("  'Describe pod <pod-name>'")
    print("  'exit' or 'quit' to exit")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("\nAgent thinking...")

            # Run agent
            result = agent.invoke({
                "messages": [("user", user_input)]
            })

            # Extract final response
            final_message = result["messages"][-1]
            print(f"\nAgent: {final_message.content}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
EOF

chmod +x agent.py
```

## Part 5: Run the Agent

Make sure everything is ready:

```bash
# 1. Verify Ollama is running
curl http://localhost:11434/api/tags

# 2. Verify Kubernetes cluster is healthy
kubectl get nodes

# 3. Verify nginx deployment exists
kubectl get deployment nginx
```

All three should work!

### Start the Agent

```bash
python3 agent.py
```

**Expected output:**
```
🤖 Kubernetes AI Agent
==================================================
Connected to: oppen-lab cluster
Model: llama3.2:3b (local)
==================================================

Available commands:
  'What pods are running?'
  'List all deployments'
  'Tell me about my cluster'
  'Describe pod <pod-name>'
  'exit' or 'quit' to exit

You:
```

✅ Agent is running! Type a question!

## Part 6: Interact with Your Agent

### Example 1: List Pods

```
You: What pods are running in the default namespace?

Agent thinking...

Agent: There is 1 pod running in the default namespace:
  - nginx-748c667d99-abc12
```

✅ Agent queried Kubernetes!

### Example 2: List Deployments

```
You: Show me all deployments

Agent thinking...

Agent: Deployments in default:
  - nginx: 1/1 ready
```

✅ Agent can see your deployment!

### Example 3: Cluster Info

```
You: Tell me about my cluster

Agent thinking...

Agent: Cluster Information:
  Nodes: 1
  Version: Client Version: v1.28.0
  Ready to deploy applications!
```

✅ Agent understands your cluster state!

### Example 4: Pod Details

```
You: Describe the nginx pod

Agent thinking...

Agent: [Detailed pod information including image, status, ports, etc.]
```

✅ Agent can drill into details!

### Explore on Your Own

Try questions like:
- "How many nodes do I have?"
- "Is the nginx deployment healthy?"
- "What's running on my cluster?"
- "Tell me more about [pod-name]"

The agent will use tools to find answers!

## Part 7: Understanding What Happened

```
User question: "What pods are running?"
         ↓
  Ollama LLM receives question
         ↓
  LLM decides: "I should use list_pods() tool"
         ↓
  LangGraph runs: list_pods(namespace="default")
         ↓
  subprocess runs: kubectl get pods -n default
         ↓
  Tool returns: ["nginx-748c667d99-abc12"]
         ↓
  LLM receives tool result
         ↓
  LLM generates answer: "There is 1 pod running: nginx-748..."
         ↓
  Agent prints response to user
```

This is called **ReAct** (Reasoning + Acting):
1. **Reason**: LLM thinks about what to do
2. **Act**: LLM calls tools
3. **Observe**: LLM sees tool results
4. **Reason again**: LLM thinks about answer
5. **Respond**: LLM gives human-readable response

## Part 8: Add More Tools (Optional Advanced)

You can extend the agent with more tools:

```bash
cat >> k8s_tools.py <<'EOF'

@tool
def get_logs(pod_name: str, namespace: str = "default", lines: int = 20) -> str:
    """Get logs from a pod"""
    try:
        result = subprocess.run(
            ["kubectl", "logs", pod_name, "-n", namespace, f"--tail={lines}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Error getting logs: {str(e)}"

# Update TOOLS list
TOOLS = [list_pods, get_pod_details, list_deployments, get_cluster_info, get_logs]
EOF
```

Now your agent can also fetch logs!

```
You: Show me the logs from the nginx pod

Agent: [Recent nginx access logs]
```

## Validation Checkpoint ✅

Complete all of these checks:

- [ ] Ollama installed: `ollama list` shows llama3.2:3b
- [ ] Ollama running: `curl http://localhost:11434/api/tags` works
- [ ] Python packages installed: `pip list | grep langchain`
- [ ] Tools work: `kubectl get pods` works
- [ ] Agent starts: `python3 agent.py` shows "🤖 Kubernetes AI Agent"
- [ ] Agent can list pods: Ask "What pods are running?" → Gets response
- [ ] Agent can list deployments: Ask "Show deployments" → Gets response
- [ ] Agent understands cluster: Ask "Tell me about cluster" → Gets info

**If all boxes are checked**: You've completed Lab 1.4! 🎉

## Troubleshooting

### "Ollama is not running"

**Problem**: `curl http://localhost:11434/api/tags` fails.

**Solution:**
```bash
# Start Ollama
ollama serve &

# Wait 5 seconds
sleep 5

# Test again
curl http://localhost:11434/api/tags
```

### "Model not found"

**Problem**: `ollama list` doesn't show llama3.2:3b.

**Solution:**
```bash
# Download the model
ollama pull llama3.2:3b

# Wait for download (2-10 min)

# Verify
ollama list
```

### "Agent fails to find kubectl"

**Problem**: `python3 agent.py` shows "kubectl: command not found".

**Solution:**
```bash
# Verify kubectl is installed
which kubectl

# If not found, add to PATH
export PATH="/usr/local/bin:$PATH"

# Or reinstall kubectl (see Lab 1.1)
```

### "Agent hangs or is slow"

**Problem**: Agent takes >30 seconds to respond.

**Solution:**
1. Ollama might be slow on first request
2. First request loads model into memory (~30s)
3. Subsequent requests are faster (~5s)
4. If still slow, check:
   ```bash
   # Check Ollama status
   ps aux | grep ollama

   # Monitor resources
   top  # or Task Manager on Windows
   ```

### "CUDA out of memory" or similar

**Problem**: Ollama crashes with GPU error.

**Solution:**
- Ollama falls back to CPU automatically
- Or specify CPU-only: `OLLAMA_NUM_GPU=0 ollama serve`
- First run slower but memory safe

### Agent Responds Incorrectly

**Problem**: Agent misunderstands question or gives wrong answer.

**Solution:**
1. Ask more specific questions:
   - ✅ "List all pods in the default namespace"
   - ❌ "What's happening?" (too vague)
2. Use complete pod/deployment names
3. This is expected - small 3B model is smart but not perfect
4. Larger models (7B, 13B) more accurate but need more RAM

## Optional: Deployment Approaches

### Approach 1: Running Agent Locally (What you just did)

**Pros:**
- Easy to debug
- Full control
- Works on any machine

**Cons:**
- Terminal-based only
- Must run manually

### Approach 2: Agent in Kubernetes (Advanced)

You could run the agent as a Pod in your cluster:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: k8s-agent
spec:
  containers:
  - name: agent
    image: python:3.11
    command: ["python", "agent.py"]
    # ... mount code, set env vars ...
```

Then users interact via API instead of terminal.

This is what Phase 2 (Infrastructure as Code) will explore!

## What You've Learned

✅ **Installed Ollama**: Run LLMs locally, no API keys

✅ **Created Kubernetes Tools**: Custom Python functions that query K8s

✅ **Built AI Agent**: LangGraph agent using ReAct pattern

✅ **Connected Agent to Cluster**: Agent can query real cluster state

✅ **Demonstrated LLM Tool Use**: LLM decides which tools to call, interprets results

This is the foundation for Phase 2+ where agents become autonomous operators!

## Next Steps

You've completed Phase 1! What comes next:

**In Phase 2** (Infrastructure as Code):
- Deploy agent to Kubernetes as a Pod
- Connect agent to Terraform for infrastructure automation
- Build state machines for autonomous operations

**In Phase 3** (Chaos & Diagnostics):
- Agent monitors Prometheus metrics
- Agent diagnoses failures
- Agent recommends fixes

**In Phase 4+** (Self-Healing & ML):
- Agent executes fixes autonomously
- Agent trains ML models
- Agent optimizes costs

---

## Quick Reference: Commands

```bash
# Ollama
ollama serve              # Start Ollama
ollama list              # List downloaded models
ollama pull llama3.2:3b  # Download model
ollama run llama3.2:3b   # Chat with model

# Python
python3 -m venv venv    # Create virtual environment
source venv/bin/activate # Activate (macOS/Linux)
pip install -r requirements.txt  # Install packages

# Agent
python3 agent.py        # Run agent
python3 -i agent.py     # Run in interactive mode

# Testing
curl http://localhost:11434/api/tags  # Test Ollama
kubectl get pods        # Test kubectl
python3 -c "from k8s_tools import TOOLS"  # Test tools
```

---

**Completed Lab 1.4?** → You've finished Phase 1! Take the Quiz! 🏆
