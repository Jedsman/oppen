# Codebase Review Report

## 🚨 Critical Issues

### 1. Missing Dependency: `langchain-ollama`

**Severity: High**
The files `app/core/agent.py` and `app/core/memory.py` import from `langchain_ollama`:

```python
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
```

However, the `pyproject.toml` file does not list `langchain-ollama` in the `dependencies` section. It only lists `ollama` and `langchain`. This will cause `ImportError` at runtime.

### 2. Hardcoded Cluster DNS in Client Code

**Severity: High**
In `app/core/tools.py`, the `MLflowClient` is initialized with a cluster-internal DNS:

```python
client = MLflowClient("http://mlflow-server.ml-training.svc.cluster.local:5000")
```

This URL is only resolvable from **inside** the Kubernetes cluster. Since the agent CLI (`app/main.py`) allows local execution, this will fail on a developer machine unless `tools.py` logic detects the environment or uses a configurable URL (e.g., pointing to `localhost:5000` via the port-forward mentioned in user metadata).

## 🐛 Bugs & Logic Flaws

### 3. Fragile Distributed Job Status Logic

**Severity: Medium**
In `app/training/job_manager.py`, the method `get_pytorch_job_status` calculates success:

```python
if ... worker.get("succeeded", 0) == worker.get("replicas", 1) - 1:
```

If `replicas` key is missing, it defaults to `1`. `1 - 1 = 0`. This logic seems to depend on a specific "Master + Worker" pattern where one worker is the master, or implies `-1` logic. If `succeeded` is 0 and `replicas` is 1 (default), 0 == 0. It works by coincidence or magic numbers. It's better to explicitly handle missing keys or use the CRD status provided by Kubeflow.

### 4. Race Condition in `scale_app`

**Severity: Medium**
The `scale_app` tool in `app/core/tools.py` reads and writes `terraform/terraform.tfvars.json`:

```python
with open(tf_vars_file, 'r') as f: data = json.load(f)
...
with open(tf_vars_file, 'w') as f: json.dump(data, f, indent=2)
```

There is no file locking. If two agent threads (or multiple invocations) call this tool simultaneously, the file modification can be corrupted (Last-Write-Wins overwriting previous updates).

### 5. Inconsistent Cost Calculation

**Severity: Low**
`app/core/tools.py` defines `calculate_training_cost` with hardcoded rates:

```python
gpu_rate, cpu_rate = 0.25, 0.05
```

However, `app/training/cost_optimizer.py` implements a robust `CostCalculator` that reads these from env vars or config. The tool should import and use the central `CostCalculator` to avoid inconsistency.

## ⚠️ Improvements & Security

### 6. Hardcoded Model Names

`app/core/agent.py` and `app/core/memory.py` hardcode `model="llama3.2:3b"`. This makes the application rigid. These should be loaded from environment variables (e.g., `OLLAMA_MODEL`) to allow users to switch models without changing code.

### 7. Unsafe Shell Command Construction

`terraform_run` in `app/core/tools.py` splits the command string naively:

```python
cmd_parts = command.split()
base_cmd = cmd_parts[0]
if base_cmd not in allowed_commands: ...
```

While it checks the first token, it doesn't strictly validate the subsequent arguments, potentially allowing flag injection if not carefully monitored, although `subprocess.run` with a list is safer than `shell=True`.
