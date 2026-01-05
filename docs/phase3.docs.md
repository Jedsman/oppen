# Phase 3: Chaos & Detection

**Objective**: Introduce failures and build an Agent capable of diagnosing them.

## 1. The Target (`podinfo`)

- **Deployed via**: Terraform + Helm (stefanprodan/podinfo).
- **Location**: `apps` namespace.
- **Scale**: 3 replicas.

## 2. The Chaos Monkey (`scripts/chaos_monkey.py`)

- **Action**: Connects to K8s, finds pods labeled `app.kubernetes.io/name=podinfo`, and deletes one every 30s.
- **Safety**: Locked to `apps` namespace.

## 3. The Diagnostic Agent (`scripts/phase3_agent.py`)

- **Evolution**: Extends Phase 2 Agent.
- **New Tools**:
  - `list_pods(namespace)`: Wrapper for `kubectl get pods`.
  - `get_k8s_events(namespace)`: Wrapper for `kubectl get events`.
  - `query_prometheus(query)`: Direct access to PromQL via `kubectl get --raw`.
- **Capability**: Can correlate a "Pod Created" event with a recent "Pod Deleted" event and check `sum(up)` metrics.

## 4. Verification Workflow

1. **Start Chaos**: `uv run scripts/chaos_monkey.py`
2. **Observe**: Watch pods restart (`kubectl get pods -n apps -w`).
3. **Diagnose**:
   ```bash
   uv run scripts/phase3_agent.py
   > "Is the cluster healthy?"
   > "Check apps namespace for restarts"
   ```

---

**Status**: [COMPLETE]
**Next**: Phase 4 (Self-Healing & Remediation)
