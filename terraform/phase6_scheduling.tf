# Phase 6: GPU Scheduling & Cost Optimization
# Resource quotas, priority classes, and scheduling policies

# PriorityClasses for job prioritization
resource "kubernetes_priority_class" "urgent" {
  metadata {
    name = "urgent"
  }
  value       = 1000
  global_default = false
  description = "Urgent training jobs (e.g., production model updates, critical research)"
}

resource "kubernetes_priority_class" "normal" {
  metadata {
    name = "normal"
  }
  value       = 100
  global_default = true
  description = "Normal training jobs (default priority)"
}

resource "kubernetes_priority_class" "background" {
  metadata {
    name = "background"
  }
  value       = 10
  global_default = false
  description = "Background jobs (validation runs, hyperparameter sweeps)"
}

# ResourceQuota for ml-training namespace
# Limits total GPU usage, memory, CPU to prevent resource exhaustion
resource "kubernetes_resource_quota" "ml_training" {
  metadata {
    name      = "ml-training-quota"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  spec {
    hard = {
      "requests.nvidia.com/gpu"  = "2"          # Max 2 GPUs in namespace
      "limits.nvidia.com/gpu"    = "2"          # Hard limit 2 GPUs
      "requests.memory"          = "16Gi"       # Max 16GB memory requested
      "limits.memory"            = "32Gi"       # Max 32GB memory limit
      "requests.cpu"             = "8"          # Max 8 CPU cores requested
      "limits.cpu"               = "16"         # Max 16 CPU cores limit
      "pods"                     = "50"         # Max 50 pods
    }

    scope_selector {
      match_expression {
        operator       = "In"
        scope_name     = "PriorityClass"
        values         = [kubernetes_priority_class.urgent.metadata[0].name]
      }
    }
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# ResourceQuota for background jobs (stricter limits)
resource "kubernetes_resource_quota" "ml_training_background" {
  metadata {
    name      = "ml-training-background-quota"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  spec {
    hard = {
      "requests.nvidia.com/gpu"  = "1"          # Max 1 GPU for background jobs
      "limits.nvidia.com/gpu"    = "1"
      "requests.memory"          = "4Gi"        # Max 4GB memory for background
      "limits.memory"            = "8Gi"
    }

    scope_selector {
      match_expression {
        operator       = "In"
        scope_name     = "PriorityClass"
        values         = [kubernetes_priority_class.background.metadata[0].name]
      }
    }
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# LimitRange for default resource requests/limits per pod
resource "kubernetes_limit_range" "ml_training" {
  metadata {
    name      = "ml-training-limits"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  spec {
    limit {
      type = "Pod"
      max = {
        "cpu"              = "8"
        "memory"           = "16Gi"
        "nvidia.com/gpu"   = "1"
      }
      min = {
        "cpu"              = "100m"
        "memory"           = "64Mi"
      }
    }

    limit {
      type = "Container"
      max = {
        "cpu"              = "4"
        "memory"           = "8Gi"
        "nvidia.com/gpu"   = "1"
      }
      min = {
        "cpu"              = "50m"
        "memory"           = "32Mi"
      }
      default = {
        "cpu"              = "500m"
        "memory"           = "256Mi"
      }
      default_request = {
        "cpu"              = "250m"
        "memory"           = "128Mi"
      }
    }
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# ConfigMap for scheduling and cost policies
resource "kubernetes_config_map" "scheduling_policies" {
  metadata {
    name      = "scheduling-policies"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  data = {
    # Job scheduling thresholds
    "max_concurrent_jobs"           = "3"           # Max 3 jobs running simultaneously
    "max_jobs_per_priority_level"   = "2"           # Max 2 urgent, 2 normal, etc.
    "gpu_utilization_threshold_min" = "30"          # Alert if < 30% utilized
    "gpu_utilization_threshold_max" = "90"          # Alert if > 90% (throttling)

    # Cost policy
    "budget_enforcement"            = "warn"        # "warn" at 70%, "block" at 100%
    "peak_hours"                    = "06:00-22:00" # UTC, for time-based pricing
    "peak_rate_multiplier"          = "1.0"         # Full price during peak
    "offpeak_rate_multiplier"       = "0.5"         # 50% discount during off-peak

    # Job characteristics
    "normal_job_timeout_seconds"    = "3600"        # 1 hour normal timeout
    "long_job_timeout_seconds"      = "14400"       # 4 hours for long-running jobs
    "urgent_job_timeout_seconds"    = "1800"        # 30 min for urgent (get results fast)

    # Spot instance simulation (Phase 6+)
    "spot_instance_discount"        = "0.7"         # 70% cheaper
    "spot_instance_interruption_rate" = "0.05"      # 5% chance of interruption
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# ConfigMap for cost model parameters
resource "kubernetes_config_map" "cost_model" {
  metadata {
    name      = "cost-model"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  data = {
    # Resource costs (hourly rates)
    "gpu_hourly_rate"               = "0.25"       # $/hour for GPU
    "cpu_hourly_rate_per_core"      = "0.05"       # $/hour per CPU core
    "memory_hourly_rate_gb"         = "0.01"       # $/hour per GB memory

    # Cost efficiency metrics
    "target_gpu_utilization"        = "70"         # Target % utilization
    "target_training_time"          = "1800"       # Target seconds per epoch
    "cost_per_accuracy_point"       = "1.0"        # Cost budget per % improvement

    # Cost multipliers
    "urgent_job_cost_multiplier"    = "1.5"        # Urgent jobs cost 50% more (priority)
    "batch_job_discount"            = "0.9"        # Batch jobs 10% cheaper (better utilization)

    # Forecasting parameters
    "forecast_confidence_interval"  = "0.95"       # 95% confidence for forecasts
    "historical_data_window_days"   = "30"         # Use last 30 days for trends
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# Output for Terraform
output "priority_classes" {
  description = "Created priority classes for job scheduling"
  value = {
    urgent     = kubernetes_priority_class.urgent.metadata[0].name
    normal     = kubernetes_priority_class.normal.metadata[0].name
    background = kubernetes_priority_class.background.metadata[0].name
  }
}

output "quotas" {
  description = "Resource quotas applied to ml-training namespace"
  value = {
    total_gpu_limit = "2"
    total_memory_limit = "32Gi"
    total_cpu_limit = "16"
  }
}

output "scheduling_policies" {
  description = "Scheduling policies configuration"
  value = {
    max_concurrent_jobs = kubernetes_config_map.scheduling_policies.data["max_concurrent_jobs"]
    budget_enforcement = kubernetes_config_map.scheduling_policies.data["budget_enforcement"]
    peak_hours = kubernetes_config_map.scheduling_policies.data["peak_hours"]
  }
}
