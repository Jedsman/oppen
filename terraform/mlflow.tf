resource "kubernetes_namespace" "ml_training" {
  metadata {
    name = "ml-training"
  }
}

resource "kubernetes_deployment" "mlflow_server" {
  metadata {
    name      = "mlflow-server"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }

  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "mlflow-server"
      }
    }
    template {
      metadata {
        labels = {
          app = "mlflow-server"
        }
      }
      spec {
        container {
          name  = "mlflow"
          image = "ghcr.io/mlflow/mlflow:v2.9.2"

          command = [
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--backend-store-uri", "sqlite:////tmp/mlflow/mlflow.db",
            "--default-artifact-root", "/tmp/mlflow/artifacts"
          ]

          port {
            container_port = 5000
          }

          volume_mount {
            name       = "mlflow-storage"
            mount_path = "/tmp/mlflow"
          }

          env {
            name  = "MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"
            value = "3600"
          }
          env {
            name  = "GUNICORN_CMD_ARGS"
            value = "--workers=2 --timeout=120 --graceful-timeout=60"
          }

          resources {
            requests = {
              cpu    = "500m"
              memory = "1Gi"
            }
            limits = {
              cpu    = "2000m"
              memory = "2Gi"
            }
          }
        }

        volume {
          name = "mlflow-storage"
          empty_dir {}
        }
      }
    }
  }

  depends_on = [kubernetes_namespace.ml_training]
}

resource "kubernetes_service" "mlflow_server" {
  metadata {
    name      = "mlflow-server"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }
  spec {
    selector = {
      app = "mlflow-server"
    }
    port {
      port        = 5000
      target_port = 5000
    }
    type = "ClusterIP"
  }
}
