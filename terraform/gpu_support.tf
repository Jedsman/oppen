resource "helm_release" "nvidia_device_plugin" {
  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  namespace  = "kube-system"
  version    = "0.14.5"

  set {
    name  = "nodeSelector.nvidia\\.com/gpu"
    value = "present"
  }
}

resource "helm_release" "gpu_feature_discovery" {
  name       = "gpu-feature-discovery"
  repository = "https://nvidia.github.io/gpu-feature-discovery"
  chart      = "gpu-feature-discovery"
  namespace  = "kube-system"
  version    = "0.8.2"
}
