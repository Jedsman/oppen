resource "helm_release" "podinfo" {
  name       = "podinfo"
  repository = "https://stefanprodan.github.io/podinfo"
  chart      = "podinfo"
  namespace  = kubernetes_namespace.apps.metadata[0].name

  set {
    name  = "replicaCount"
    value = var.podinfo_replicas
  }
}
