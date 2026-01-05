import mlflow
from typing import List, Dict, Optional

class MLflowClient:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def list_experiments(self, experiment_name: Optional[str] = None) -> List[Dict]:
        experiments = self.client.search_experiments()
        if experiment_name:
            experiments = [e for e in experiments if experiment_name.lower() in e.name.lower()]
        return [{"id": e.experiment_id, "name": e.name} for e in experiments]

    def get_experiment_runs(self, experiment_name: str, max_results: int = 10) -> List[Dict]:
        exp = self.client.get_experiment_by_name(experiment_name)
        if not exp:
            return []
        runs = self.client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=max_results)
        return [{"id": r.info.run_id, "status": r.info.status, "metrics": dict(r.data.metrics), "params": dict(r.data.params), "tags": dict(r.data.tags)} for r in runs]

    def get_run_details(self, run_id: str) -> Dict:
        run = self.client.get_run(run_id)
        return {
            "id": run.info.run_id,
            "status": run.info.status,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags)
        }

    def get_run_by_trial_number(self, experiment_name: str, trial_number: int) -> Optional[Dict]:
        """Find MLflow run for a specific Optuna trial number"""
        exp = self.client.get_experiment_by_name(experiment_name)
        if not exp:
            return None

        # Search for run with matching trial_number tag
        runs = self.client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.trial_number = '{trial_number}'",
            max_results=1
        )

        if not runs:
            return None

        run = runs[0]
        return {
            "id": run.info.run_id,
            "status": run.info.status,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }
