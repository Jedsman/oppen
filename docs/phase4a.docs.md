# Phase 4a: Robust IaC (The Safety Upgrade)

**Objective**: Replace fragile Regex-based code editing with safe, standard Terraform Variables.

## 1. Infrastructure Changes

- **`terraform/variables.tf`**: Defined `podinfo_replicas` numeric variable.
- **`terraform/podinfo.tf`**: Updated to use `var.podinfo_replicas`.
- **`terraform/terraform.tfvars.json`**: New state file for variable persistence.

## 2. The Robust Healer (`scripts/phase4a_agent.py`)

- **Upgrade**: `scale_app` tool now reads/writes `terraform.tfvars.json`.
- **Logic**:
  1. Load JSON.
  2. Update key `podinfo_replicas`.
  3. Write JSON.
  4. Run `terraform apply`.
- **Benefit**: No risk of syntax errors or regex mismatches breaking the infrastructure code.

## 3. Verification Results

- **Instruction**: "Scale podinfo to 3 replicas."
- **Outcome**:
  - `terraform.tfvars.json` updated to `3`.
  - Terraform updated in-place.
  - Deployment stabilized at **3/3** replicas.

---

**Status**: [COMPLETE]
**Next**: Phase 4b (Event-Driven Architecture)
