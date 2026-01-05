"""Phase 6 Tier 2: Lightweight Monitoring

Uses Kubernetes API and MLflow directly instead of Prometheus/Grafana
for quick deployment and low resource overhead.
"""

import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional
from app.training.mlflow_client import MLflowClient
from app.training.job_manager import TrainingJobManager


class TrainingMonitor:
    """Lightweight monitoring using K8s API and MLflow directly"""

    def __init__(self, namespace: str = "ml-training"):
        self.namespace = namespace
        self.mlflow_client = MLflowClient("http://mlflow-server.ml-training.svc.cluster.local:5000")
        self.job_manager = TrainingJobManager()

    def get_job_metrics(self, job_name: str) -> Dict:
        """Get metrics for a specific job from K8s and MLflow"""
        try:
            # Get K8s job status
            result = subprocess.run(
                ["kubectl", "get", "job", job_name, "-n", self.namespace, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {"error": f"Job not found: {job_name}"}

            k8s_job = json.loads(result.stdout)
            status = k8s_job["status"]

            # Try to find corresponding MLflow run
            mlflow_metrics = {}
            try:
                runs = self.mlflow_client.get_experiment_runs("mnist-training", max_results=10)
                # Match by job name or timestamp
                for run in runs:
                    if job_name.split('-')[2:4] in run.get('run_id', '')[:16]:  # Rough timestamp match
                        mlflow_metrics = run.get('metrics', {})
                        break
            except:
                pass

            return {
                "job_name": job_name,
                "status": "succeeded" if status.get("succeeded", 0) > 0 else "running",
                "completions": status.get("succeeded", 0),
                "active": status.get("active", 0),
                "failed": status.get("failed", 0),
                "start_time": k8s_job.get("status", {}).get("startTime", "N/A"),
                "completion_time": k8s_job.get("status", {}).get("completionTime", "N/A"),
                "mlflow_metrics": mlflow_metrics,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_cluster_metrics(self) -> Dict:
        """Get cluster-wide metrics"""
        try:
            # Get node metrics
            result = subprocess.run(
                ["kubectl", "top", "nodes"],
                capture_output=True,
                text=True,
                timeout=10
            )

            node_lines = result.stdout.strip().split('\n')[1:]  # Skip header
            nodes_data = []
            for line in node_lines:
                parts = line.split()
                if len(parts) >= 5:
                    nodes_data.append({
                        "name": parts[0],
                        "cpu_percent": int(parts[2].rstrip('%')),
                        "memory_percent": int(parts[4].rstrip('%')),
                    })

            # Get pod metrics in ml-training namespace
            result = subprocess.run(
                ["kubectl", "top", "pods", "-n", "ml-training"],
                capture_output=True,
                text=True,
                timeout=10
            )

            pod_lines = result.stdout.strip().split('\n')[1:]  # Skip header
            pods_data = []
            for line in pod_lines:
                parts = line.split()
                if len(parts) >= 3:
                    pods_data.append({
                        "name": parts[0],
                        "cpu_m": int(parts[1].rstrip('m')),
                        "memory_mi": int(parts[2].rstrip('Mi')),
                    })

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "nodes": nodes_data,
                "ml_training_pods": pods_data,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_cost_report(self) -> Dict:
        """Generate cost report from MLflow runs"""
        try:
            experiments = self.mlflow_client.list_experiments()
            total_cost = 0
            total_duration = 0
            run_count = 0

            for exp in experiments:
                runs = self.mlflow_client.get_experiment_runs(exp['name'], max_results=100)
                for run in runs:
                    metrics = run.get('metrics', {})
                    cost = metrics.get('total_cost_usd', 0)
                    duration = metrics.get('duration_seconds', 0)
                    total_cost += cost
                    total_duration += duration
                    run_count += 1

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_runs": run_count,
                "total_cost_usd": round(total_cost, 4),
                "total_duration_hours": round(total_duration / 3600, 2),
                "average_cost_per_run": round(total_cost / run_count, 4) if run_count > 0 else 0,
                "average_duration_minutes": round(total_duration / run_count / 60, 1) if run_count > 0 else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_health_status(self) -> Dict:
        """Get overall system health"""
        try:
            # Check if key services are running
            services = ["mlflow-server"]
            service_status = {}

            for svc in services:
                result = subprocess.run(
                    ["kubectl", "get", "pods", "-n", "ml-training", "-l", f"app={svc}", "-o", "json"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    pods = json.loads(result.stdout).get("items", [])
                    ready_count = sum(1 for p in pods if p["status"]["conditions"][-1]["status"] == "True")
                    service_status[svc] = {
                        "replicas": len(pods),
                        "ready": ready_count,
                        "healthy": ready_count > 0
                    }

            # Get resource quotas
            result = subprocess.run(
                ["kubectl", "get", "quota", "-n", "ml-training", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            quotas = {}
            if result.returncode == 0:
                quota_items = json.loads(result.stdout).get("items", [])
                for quota in quota_items:
                    name = quota["metadata"]["name"]
                    hard = quota.get("status", {}).get("hard", {})
                    used = quota.get("status", {}).get("used", {})
                    quotas[name] = {
                        "limits": hard,
                        "used": used
                    }

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "services": service_status,
                "quotas": quotas,
                "overall_healthy": all(s.get("healthy", False) for s in service_status.values()),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_dashboard_summary(self) -> str:
        """Get a text-based dashboard summary"""
        summary = "=" * 70 + "\n"
        summary += "📊 TRAINING SYSTEM DASHBOARD\n"
        summary += f"Generated: {datetime.utcnow().isoformat()}\n"
        summary += "=" * 70 + "\n\n"

        # Health status
        health = self.get_health_status()
        summary += "🏥 SYSTEM HEALTH:\n"
        if "error" not in health:
            for svc, status in health.get("services", {}).items():
                health_emoji = "✅" if status.get("healthy") else "❌"
                summary += f"  {health_emoji} {svc}: {status['ready']}/{status['replicas']} ready\n"
        summary += "\n"

        # Cost report
        cost = self.get_cost_report()
        summary += "💰 COST TRACKING:\n"
        if "error" not in cost:
            summary += f"  Total Runs: {cost.get('total_runs', 0)}\n"
            summary += f"  Total Cost: ${cost.get('total_cost_usd', 0):.4f}\n"
            summary += f"  Avg Cost/Run: ${cost.get('average_cost_per_run', 0):.4f}\n"
            summary += f"  Total Duration: {cost.get('total_duration_hours', 0):.1f}h\n"
        summary += "\n"

        # Resource metrics
        cluster = self.get_cluster_metrics()
        summary += "🖥️  CLUSTER RESOURCES:\n"
        if "error" not in cluster:
            for node in cluster.get("nodes", []):
                summary += f"  {node['name']}: CPU {node['cpu_percent']}%, Memory {node['memory_percent']}%\n"
            summary += "\n  ML-Training Pods:\n"
            for pod in cluster.get("ml_training_pods", [])[:5]:  # Top 5
                summary += f"    {pod['name']}: {pod['cpu_m']}m CPU, {pod['memory_mi']}Mi Memory\n"
        summary += "\n"

        summary += "=" * 70 + "\n"
        summary += "For Prometheus/Grafana dashboards, see phase6_monitoring.tf\n"
        summary += "(Deferred to Phase 6 follow-up due to image size constraints)\n"

        return summary
