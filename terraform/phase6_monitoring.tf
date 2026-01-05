# Phase 6 Tier 2: Monitoring Infrastructure
# Prometheus + Grafana for cost tracking and resource monitoring

# Namespace for monitoring (if needed separately)
# For simplicity, we'll use ml-training namespace

# Prometheus ConfigMap with scrape configs
resource "kubernetes_config_map" "prometheus_config" {
  metadata {
    name      = "prometheus-config"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  data = {
    "prometheus.yml" = <<-EOT
global:
  scrape_interval: 30s
  evaluation_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files:
  - '/etc/prometheus/rules/*.yml'

scrape_configs:
  # MLflow metrics endpoint
  - job_name: 'mlflow'
    metrics_path: '/api/2.0/metrics'
    static_configs:
      - targets: ['mlflow-server:5000']
    scrape_interval: 1m
    scrape_timeout: 10s

  # Kubernetes API server metrics
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https

  # Node metrics (kubelet)
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)

  # Pod metrics (cAdvisor)
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: "true"
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
EOT
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# Prometheus alert rules ConfigMap
resource "kubernetes_config_map" "prometheus_rules" {
  metadata {
    name      = "prometheus-rules"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  data = {
    "cost_alerts.yml" = <<-EOT
groups:
  - name: cost_alerts
    interval: 30s
    rules:
      # Alert when GPU utilization is low (waste)
      - alert: GPUUnderutilized
        expr: gpu_utilization < 30
        for: 5m
        annotations:
          summary: "GPU underutilized ({{ $value }}%)"
          description: "GPU utilization below 30% threshold"

      # Alert when approaching budget limit
      - alert: BudgetWarning
        expr: training_cost_usd > (training_budget_usd * 0.7)
        for: 1m
        annotations:
          summary: "Budget 70% utilized"
          description: "Training cost approaching limit"

      # Alert when exceeding budget
      - alert: BudgetExceeded
        expr: training_cost_usd > training_budget_usd
        for: 1m
        annotations:
          summary: "Budget exceeded!"
          description: "Training cost has exceeded budget"

      # Alert on high memory pressure
      - alert: MemoryPressure
        expr: node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.2
        for: 5m
        annotations:
          summary: "Memory pressure detected"
          description: "Available memory below 20%"

      # Alert on job failures
      - alert: JobFailure
        expr: increase(kubernetes_job_failures_total[5m]) > 0
        annotations:
          summary: "Training job failed"
          description: "A training job has failed"

      # Alert when job exceeds 2x expected duration
      - alert: JobSlowness
        expr: job_duration_seconds > (job_expected_duration * 2)
        for: 5m
        annotations:
          summary: "Job running slower than expected"
          description: "Job duration 2x above baseline"
EOT
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# Prometheus Deployment
resource "kubernetes_deployment" "prometheus" {
  metadata {
    name      = "prometheus"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "prometheus"
      }
    }

    template {
      metadata {
        labels = {
          app = "prometheus"
        }
      }

      spec {
        service_account_name = kubernetes_service_account.prometheus.metadata[0].name

        container {
          name  = "prometheus"
          image = "prom/prometheus:latest"
          args  = [
            "--config.file=/etc/prometheus/prometheus.yml",
            "--storage.tsdb.path=/prometheus",
            "--storage.tsdb.retention.time=15d",
            "--web.console.libraries=/usr/share/prometheus/console_libraries",
            "--web.console.templates=/usr/share/prometheus/consoles"
          ]

          port {
            name           = "web"
            container_port = 9090
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "1000m"
              memory = "1Gi"
            }
          }

          volume_mount {
            name       = "config"
            mount_path = "/etc/prometheus"
          }

          volume_mount {
            name       = "rules"
            mount_path = "/etc/prometheus/rules"
          }

          volume_mount {
            name       = "storage"
            mount_path = "/prometheus"
          }
        }

        volume {
          name = "config"
          config_map {
            name = kubernetes_config_map.prometheus_config.metadata[0].name
          }
        }

        volume {
          name = "rules"
          config_map {
            name = kubernetes_config_map.prometheus_rules.metadata[0].name
          }
        }

        volume {
          name = "storage"
          empty_dir {}
        }
      }
    }
  }

  depends_on = [kubernetes_namespace.ml_training, kubernetes_config_map.prometheus_config]
}

# Prometheus Service
resource "kubernetes_service" "prometheus" {
  metadata {
    name      = "prometheus"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  spec {
    selector = {
      app = "prometheus"
    }

    port {
      port        = 9090
      target_port = "web"
      name        = "web"
    }

    type = "ClusterIP"
  }

  depends_on = [kubernetes_deployment.prometheus]
}

# ServiceAccount for Prometheus (to read K8s metrics)
resource "kubernetes_service_account" "prometheus" {
  metadata {
    name      = "prometheus"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# ClusterRole for Prometheus
resource "kubernetes_cluster_role" "prometheus" {
  metadata {
    name = "prometheus"
  }

  rule {
    api_groups = [""]
    resources  = ["nodes", "nodes/proxy", "services", "endpoints", "pods"]
    verbs      = ["get", "list", "watch"]
  }

  rule {
    api_groups = ["extensions"]
    resources  = ["ingresses"]
    verbs      = ["get", "list", "watch"]
  }

  rule {
    api_groups = ["batch"]
    resources  = ["jobs"]
    verbs      = ["get", "list", "watch"]
  }
}

# ClusterRoleBinding for Prometheus
resource "kubernetes_cluster_role_binding" "prometheus" {
  metadata {
    name = "prometheus"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.prometheus.metadata[0].name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.prometheus.metadata[0].name
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }
}

# Grafana Deployment
resource "kubernetes_deployment" "grafana" {
  metadata {
    name      = "grafana"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "grafana"
      }
    }

    template {
      metadata {
        labels = {
          app = "grafana"
        }
      }

      spec {
        container {
          name  = "grafana"
          image = "grafana/grafana:latest"

          port {
            name           = "web"
            container_port = 3000
          }

          env {
            name  = "GF_SECURITY_ADMIN_PASSWORD"
            value = "admin"
          }

          env {
            name  = "GF_INSTALL_PLUGINS"
            value = "grafana-piechart-panel"
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "128Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
          }

          volume_mount {
            name       = "storage"
            mount_path = "/var/lib/grafana"
          }

          # Datasource provisioning
          volume_mount {
            name       = "datasources"
            mount_path = "/etc/grafana/provisioning/datasources"
          }
        }

        volume {
          name = "storage"
          empty_dir {}
        }

        volume {
          name = "datasources"
          config_map {
            name = kubernetes_config_map.grafana_datasources.metadata[0].name
          }
        }
      }
    }
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# Grafana Service
resource "kubernetes_service" "grafana" {
  metadata {
    name      = "grafana"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  spec {
    selector = {
      app = "grafana"
    }

    port {
      port        = 3000
      target_port = "web"
      name        = "web"
    }

    type = "ClusterIP"
  }

  depends_on = [kubernetes_deployment.grafana]
}

# Grafana Datasource Configuration
resource "kubernetes_config_map" "grafana_datasources" {
  metadata {
    name      = "grafana-datasources"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  data = {
    "prometheus.yaml" = <<-EOT
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOT
  }

  depends_on = [kubernetes_namespace.ml_training]
}

# Output for Terraform
output "prometheus_url" {
  description = "Prometheus UI access"
  value       = "kubectl port-forward -n ml-training svc/prometheus 9090:9090 &"
}

output "grafana_url" {
  description = "Grafana UI access (admin/admin)"
  value       = "kubectl port-forward -n ml-training svc/grafana 3000:3000 &"
}

output "monitoring_services" {
  description = "Monitoring services deployed"
  value = {
    prometheus = kubernetes_service.prometheus.metadata[0].name
    grafana    = kubernetes_service.grafana.metadata[0].name
  }
}
