"""Phase 7: AutoML/Hyperparameter Optimization using Optuna

Provides automated hyperparameter tuning with parallel trial execution,
cost-aware optimization, and MLflow integration.
"""

import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import os

from app.training.job_queue import JobQueue, JobPriority
from app.training.mlflow_client import MLflowClient

# Constants
STORAGE_PATH = "sqlite:///automl_studies.db"
MLFLOW_URI = "http://mlflow-server.ml-training.svc.cluster.local:5000"


@dataclass
class SearchSpace:
    """Hyperparameter search space definition"""
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    lr_log: bool = True

    batch_size_choices: List[int] = field(default_factory=lambda: [32, 64, 128, 256])

    epochs_min: int = 1
    epochs_max: int = 10

    # Additional params for LLM
    weight_decay_min: float = 0.0
    weight_decay_max: float = 0.1


@dataclass
class StudyConfig:
    """Optuna study configuration"""
    study_name: str
    model_type: str  # "mnist" or "llm"
    direction: str = "maximize"  # maximize val_accuracy
    n_trials: int = 20
    max_concurrent_trials: int = 3
    timeout_seconds: Optional[int] = None
    search_space: Optional[SearchSpace] = None
    cost_budget_usd: Optional[float] = None  # Optional cost constraint

    def __post_init__(self):
        if self.search_space is None:
            self.search_space = SearchSpace()


class OptunaStudyManager:
    """Manages Optuna studies for hyperparameter optimization"""

    def __init__(
        self,
        storage: str = STORAGE_PATH,
        mlflow_tracking_uri: str = MLFLOW_URI,
    ):
        self.storage = storage
        self.mlflow_client = MLflowClient(mlflow_tracking_uri)
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def create_or_load_study(
        self,
        config: StudyConfig
    ) -> optuna.Study:
        """
        Create a new study or load existing one.

        Args:
            config: Study configuration

        Returns:
            Optuna Study object
        """
        sampler = TPESampler(
            seed=42,  # Reproducibility
            n_startup_trials=5,  # Random sampling first
            multivariate=True,  # Model param interactions
        )

        pruner = MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,  # Start pruning after epoch 1
            interval_steps=1,
        )

        study = optuna.create_study(
            study_name=config.study_name,
            storage=self.storage,
            direction=config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,  # Resume automatically
        )

        return study

    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        search_space: SearchSpace,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial
            search_space: Search space definition
            model_type: "mnist" or "llm"

        Returns:
            Dict of hyperparameters
        """
        params = {}

        # Learning rate (log scale)
        params["lr"] = trial.suggest_float(
            "lr",
            search_space.lr_min,
            search_space.lr_max,
            log=search_space.lr_log
        )

        # Batch size (categorical)
        params["batch_size"] = trial.suggest_categorical(
            "batch_size",
            search_space.batch_size_choices
        )

        # Epochs (int)
        params["epochs"] = trial.suggest_int(
            "epochs",
            search_space.epochs_min,
            search_space.epochs_max
        )

        # Model-specific params
        if model_type == "llm":
            params["weight_decay"] = trial.suggest_float(
                "weight_decay",
                search_space.weight_decay_min,
                search_space.weight_decay_max
            )

        return params

    def run_trial_job(
        self,
        trial: optuna.Trial,
        model_type: str,
        hyperparams: Dict[str, Any],
        job_queue: JobQueue,
    ) -> Tuple[str, Dict]:
        """
        Submit a trial as a training job.

        Args:
            trial: Optuna trial
            model_type: "mnist" or "llm"
            hyperparams: Suggested hyperparameters
            job_queue: Job queue instance

        Returns:
            Tuple of (job_id, job_details)
        """
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"automl-{model_type}-trial{trial.number}-{ts}"
        job_name = f"{model_type}-trial-{trial.number}"

        # Submit to queue with normal priority
        success, msg, queue_pos = job_queue.submit_job(
            job_id=job_id,
            name=job_name,
            model_type=model_type,
            priority=JobPriority.NORMAL,
            epochs=int(hyperparams["epochs"]),
            batch_size=int(hyperparams["batch_size"]),
            lr=float(hyperparams["lr"]),
            gpu_enabled=True,
        )

        if not success:
            raise RuntimeError(f"Failed to queue trial: {msg}")

        return job_id, {"job_id": job_id, "queue_position": queue_pos}

    def wait_for_trial_completion(
        self,
        job_id: str,
        job_queue: JobQueue,
        timeout_seconds: int = 3600,
        poll_interval: int = 10,
    ) -> Dict:
        """
        Wait for a trial job to complete.

        Args:
            job_id: Job ID to wait for
            job_queue: Job queue instance
            timeout_seconds: Max wait time
            poll_interval: Polling interval

        Returns:
            Job details dict
        """
        start = time.time()

        while time.time() - start < timeout_seconds:
            # Get job status from queue
            try:
                job = job_queue.jobs.get(job_id)
                if job and job.status in ["completed", "failed"]:
                    return {
                        "job_id": job_id,
                        "status": job.status,
                        "duration": job.estimated_duration_seconds,
                        "error_message": job.error_message
                    }
            except:
                pass

            time.sleep(poll_interval)

        raise optuna.TrialPruned(f"Trial {job_id} timed out after {timeout_seconds}s")

    def get_trial_metrics_from_mlflow(
        self,
        model_type: str,
        trial_number: int
    ) -> Optional[Dict]:
        """
        Retrieve trial metrics from MLflow.

        Args:
            model_type: "mnist" or "llm"
            trial_number: Trial number

        Returns:
            Metrics dict or None if not found
        """
        try:
            experiment_name = f"{model_type}-training"
            runs = self.mlflow_client.get_experiment_runs(
                experiment_name,
                max_results=100
            )

            # Find run matching trial number (via tags or run name)
            for run in runs:
                # Check if trial_number tag matches
                if run.get("tags", {}).get("trial_number") == str(trial_number):
                    metrics = run.get("metrics", {})
                    if "val_accuracy" in metrics:
                        return metrics

            return None
        except:
            return None

    def objective_function(
        self,
        trial: optuna.Trial,
        config: StudyConfig,
        job_queue: JobQueue,
    ) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial
            config: Study configuration
            job_queue: Job queue instance

        Returns:
            Objective value (val_accuracy)
        """
        try:
            # 1. Suggest hyperparameters
            hyperparams = self.suggest_hyperparameters(
                trial,
                config.search_space,
                config.model_type
            )

            # 2. Submit trial job
            job_id, job_details = self.run_trial_job(
                trial,
                config.model_type,
                hyperparams,
                job_queue
            )

            # 3. Wait for completion
            try:
                final_details = self.wait_for_trial_completion(
                    job_id,
                    job_queue,
                    timeout_seconds=config.timeout_seconds or 3600
                )
            except optuna.TrialPruned as e:
                raise e

            # 4. Check job status
            if final_details["status"] == "failed":
                raise optuna.TrialPruned(f"Job failed: {final_details.get('error_message')}")

            # 5. Get metrics from MLflow
            metrics = self.get_trial_metrics_from_mlflow(
                config.model_type,
                trial.number
            )

            if not metrics or "val_accuracy" not in metrics:
                raise optuna.TrialPruned("No metrics found")

            val_accuracy = float(metrics["val_accuracy"])

            # 6. Report intermediate values (for pruning)
            trial.report(val_accuracy, step=int(hyperparams["epochs"]))

            # 7. Check cost budget if specified
            if config.cost_budget_usd:
                total_cost = float(metrics.get("total_cost_usd", 0))
                if total_cost > config.cost_budget_usd:
                    raise optuna.TrialPruned(f"Exceeded cost budget: ${total_cost:.4f}")

            return val_accuracy

        except optuna.TrialPruned:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Trial failed: {str(e)}")

    def run_optimization(
        self,
        config: StudyConfig,
        job_queue: JobQueue,
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            config: Study configuration
            job_queue: Job queue instance

        Returns:
            Completed study
        """
        study = self.create_or_load_study(config)

        # MLflow callback for tracking
        mlflow_callback = MLflowCallback(
            tracking_uri=self.mlflow_tracking_uri,
            metric_name="val_accuracy",
        )

        # Run optimization
        study.optimize(
            lambda trial: self.objective_function(trial, config, job_queue),
            n_trials=config.n_trials,
            timeout=config.timeout_seconds,
            callbacks=[mlflow_callback],
            n_jobs=1,  # Sequential (queue handles parallelism)
        )

        return study

    def get_study_summary(self, study_name: str) -> Dict:
        """Get summary of a study"""
        try:
            study = optuna.load_study(study_name, storage=self.storage)

            best_trial = study.best_trial

            return {
                "study_name": study_name,
                "n_trials": len(study.trials),
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial_number": best_trial.number if best_trial else None,
                "direction": study.direction.name,
                "state": "completed" if study.best_trial else "running",
            }
        except Exception as e:
            return {"error": str(e)}

    def list_studies(self) -> List[str]:
        """List all study names"""
        try:
            return optuna.get_all_study_names(storage=self.storage)
        except:
            return []
