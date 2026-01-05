variable "podinfo_replicas" {
  description = "Number of replicas for podinfo app"
  type        = number
  default     = 3
}

variable "gpu_hourly_rate" {
  description = "Hourly cost of GPU in USD"
  type        = string
  default     = "0.25"
}

variable "cpu_hourly_rate" {
  description = "Hourly cost per CPU core in USD"
  type        = string
  default     = "0.05"
}
