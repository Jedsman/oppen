import subprocess, json
from typing import Dict

class TrainingJobManager:
    def __init__(self, namespace: str = "ml-training"):
        self.namespace = namespace

    def _build_env_list(self, extra_env: Dict = None):
        """Build environment variable list for container"""
        env = [
            {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow-server.ml-training.svc.cluster.local:5000"},
            {"name": "GPU_HOURLY_RATE", "valueFrom": {"configMapKeyRef": {"name": "training-cost-rates", "key": "gpu_hourly_rate"}}},
            {"name": "CPU_HOURLY_RATE", "valueFrom": {"configMapKeyRef": {"name": "training-cost-rates", "key": "cpu_hourly_rate"}}}
        ]
        # Phase 7: Add extra env vars (for Optuna trial tagging)
        if extra_env:
            for key, value in extra_env.items():
                env.append({"name": key, "value": str(value)})
        return env

    def create_job_manifest(self, job_name: str, image: str, training_args: Dict, gpu_enabled: bool = True, extra_env: Dict = None) -> Dict:
        args = []
        for key, value in training_args.items():
            args.extend([f"--{key}", str(value)])

        resources = {"requests": {"cpu": "4", "memory": "8Gi"}, "limits": {"cpu": "4", "memory": "8Gi"}}
        if gpu_enabled:
            resources["requests"]["nvidia.com/gpu"] = "1"
            resources["limits"]["nvidia.com/gpu"] = "1"

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": self.namespace},
            "spec": {
                "backoffLimit": 1,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "training",
                            "image": image,
                            "imagePullPolicy": "Never",
                            "args": args,
                            "env": self._build_env_list(extra_env),
                            "resources": resources
                        }]
                    }
                }
            }
        }

    def submit_job(self, manifest: Dict) -> str:
        result = subprocess.run(["kubectl", "apply", "-f", "-"], input=json.dumps(manifest), capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed: {result.stderr}")
        return manifest["metadata"]["name"]

    def get_job_status(self, job_name: str) -> Dict:
        result = subprocess.run(["kubectl", "get", "job", job_name, "-n", self.namespace, "-o", "json"], capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "not_found"}

        data = json.loads(result.stdout)
        status_obj = data.get("status", {})
        status = "pending"
        if status_obj.get("succeeded", 0) > 0:
            status = "completed"
        elif status_obj.get("failed", 0) > 0:
            status = "failed"
        elif status_obj.get("active", 0) > 0:
            status = "running"

        return {
            "name": job_name,
            "status": status,
            "active": status_obj.get("active", 0),
            "succeeded": status_obj.get("succeeded", 0),
            "failed": status_obj.get("failed", 0),
            "start_time": status_obj.get("startTime"),
            "completion_time": status_obj.get("completionTime")
        }

    def get_job_logs(self, job_name: str) -> str:
        result = subprocess.run(["kubectl", "get", "pods", "-n", self.namespace, "-l", f"job-name={job_name}", "-o", "json"], capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        pods = json.loads(result.stdout).get("items", [])
        if not pods:
            return "No pods found"

        pod_name = pods[0]["metadata"]["name"]
        result = subprocess.run(["kubectl", "logs", pod_name, "-n", self.namespace, "--tail=50"], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    # Phase 9: PyTorchJob Support for Distributed Training

    def create_pytorch_job_manifest(self, job_name: str, image: str, training_args: Dict, world_size: int = 2, gpu_per_replica: int = 1, extra_env: Dict = None) -> Dict:
        """Create Kubeflow PyTorchJob manifest for distributed training"""
        args = []
        for key, value in training_args.items():
            args.extend([f"--{key}", str(value)])

        container_spec = {
            "name": "pytorch",
            "image": image,
            "imagePullPolicy": "Never",
            "args": args,
            "env": self._build_env_list(extra_env),
            "resources": {"limits": {"nvidia.com/gpu": str(gpu_per_replica)}}
        }

        return {
            "apiVersion": "kubeflow.org/v1",
            "kind": "PyTorchJob",
            "metadata": {"name": job_name, "namespace": self.namespace},
            "spec": {
                "pytorchReplicaSpecs": {
                    "Master": {
                        "replicas": 1,
                        "restartPolicy": "OnFailure",
                        "template": {"spec": {"containers": [container_spec]}}
                    },
                    "Worker": {
                        "replicas": world_size - 1,
                        "restartPolicy": "OnFailure",
                        "template": {"spec": {"containers": [container_spec]}}
                    }
                }
            }
        }

    def submit_pytorch_job(self, manifest: Dict) -> str:
        """Submit PyTorchJob via kubectl apply"""
        result = subprocess.run(["kubectl", "apply", "-f", "-"], input=json.dumps(manifest), capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed: {result.stderr}")
        return manifest["metadata"]["name"]

    def get_pytorch_job_status(self, job_name: str) -> Dict:
        """Get PyTorchJob status"""
        result = subprocess.run(["kubectl", "get", "pytorchjob", job_name, "-n", self.namespace, "-o", "json"], capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "not_found"}

        data = json.loads(result.stdout)
        replica_statuses = data.get("status", {}).get("replicaStatuses", {})

        master = replica_statuses.get("Master", {})
        worker = replica_statuses.get("Worker", {})

        status = "pending"
        if master.get("succeeded", 0) > 0 and worker.get("succeeded", 0) == worker.get("replicas", 1) - 1:
            status = "succeeded"
        elif master.get("failed", 0) > 0 or worker.get("failed", 0) > 0:
            status = "failed"
        elif master.get("active", 0) > 0 or worker.get("active", 0) > 0:
            status = "running"

        return {
            "name": job_name,
            "status": status,
            "master_status": f"active={master.get('active', 0)} succeeded={master.get('succeeded', 0)} failed={master.get('failed', 0)}",
            "worker_count": worker.get("replicas", 0),
            "workers_succeeded": worker.get("succeeded", 0),
            "workers_failed": worker.get("failed", 0)
        }

    def get_pytorch_job_logs(self, job_name: str, tail_lines: int = 20) -> str:
        """Get logs from all PyTorchJob pods"""
        result = subprocess.run(["kubectl", "get", "pods", "-n", self.namespace, "-l", f"pytorch-job-name={job_name}", "-o", "json"], capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        pods = json.loads(result.stdout).get("items", [])
        if not pods:
            return "No pods found"

        output = []
        for pod in pods:
            pod_name = pod["metadata"]["name"]
            role = pod["metadata"]["labels"].get("pytorch-job-role", "unknown")

            log_result = subprocess.run(["kubectl", "logs", pod_name, "-n", self.namespace, f"--tail={tail_lines}"], capture_output=True, text=True)
            output.append(f"=== {role.capitalize()} ({pod_name}) ===")
            output.append(log_result.stdout if log_result.returncode == 0 else log_result.stderr)
            output.append("")

        return "\n".join(output)
