resource "kubernetes_config_map" "training_costs" {
  metadata {
    name      = "training-cost-rates"
    namespace = kubernetes_namespace.ml_training.metadata[0].name
  }
  data = {
    gpu_hourly_rate = var.gpu_hourly_rate
    cpu_hourly_rate = var.cpu_hourly_rate
  }
}
