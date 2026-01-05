import subprocess, os, json, urllib.parse, time
from datetime import datetime
from langchain_core.tools import tool
from app.core.memory import MemoryManager
from app.training.mlflow_client import MLflowClient
from app.training.job_manager import TrainingJobManager
from app.training.cost_optimizer import CostCalculator, CostRecommender, BudgetTracker
from app.training.job_queue import JobQueue, JobPriority
from app.training.monitoring import TrainingMonitor

# Global job queue instance (Phase 6)
_global_queue = JobQueue(max_concurrent_jobs=3)

# Helper for Terraform
def _run_terraform_cmd(command: str) -> str:
    """Helper to run terraform command directly."""
    allowed_commands = ["version", "init", "plan", "show", "validate", "apply"]
    cmd_parts = command.split()
    base_cmd = cmd_parts[0]
    if base_cmd not in allowed_commands:
        return f"Error: Command '{base_cmd}' is not allowed."
    tf_dir = os.path.join(os.getcwd(), "terraform")
    args = ["terraform", base_cmd, "-no-color"]
    if base_cmd == "apply":
        args.append("-auto-approve")
    try:
        result = subprocess.run(
            args, cwd=tf_dir, capture_output=True, text=True, check=False
        )
        return result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
    except Exception as e:
        return f"Failed to execute terraform: {e}"

# --- Diagnostic Tools ---

@tool
def list_containers() -> str:
    """Lists all running Docker containers. Returns JSON string."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{json .}}"],
            capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            return f"Error listing containers: {result.stderr}"
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return json.dumps(containers, indent=2)
    except Exception as e:
        return f"Failed to execute docker: {e}"

@tool
def terraform_run(command: str) -> str:
    """Runs a Terraform command (version, init, plan, show, validate, apply)."""
    return _run_terraform_cmd(command)

@tool
def list_pods(namespace: str = "default") -> str:
    """List Kubernetes pods in a namespace using kubectl."""
    try:
        cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "wide"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Failed to list pods: {e}"

@tool
def get_k8s_events(namespace: str = "default") -> str:
    """Get recent Kubernetes events in a namespace."""
    try:
        cmd = ["kubectl", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Failed to get events: {e}"

@tool
def query_prometheus(query: str) -> str:
    """Executes a PromQL query against the cluster Prometheus."""
    try:
        encoded_query = urllib.parse.quote(query)
        api_path = f"/api/v1/namespaces/monitoring/services/prometheus-kube-prometheus-prometheus:9090/proxy/api/v1/query?query={encoded_query}"
        cmd = ["kubectl", "get", "--raw", api_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return f"Prometheus Query Failed: {result.stderr}"
        data = json.loads(result.stdout)
        if data.get("status") == "success":
            return json.dumps(data["data"]["result"], indent=2)
        else:
            return f"Prometheus Error: {data}"
    except Exception as e:
        return f"Failed to query Prometheus: {e}"

# --- Remediation Tools ---

@tool
def scale_app(app_name: str, replicas: int) -> str:
    """
    Scales application by updating 'terraform.tfvars.json'. ONLY supports 'podinfo'.
    """
    if app_name != "podinfo":
        return "Error: Auto-scaling only supported for 'podinfo'."
    tf_vars_file = os.path.join(os.getcwd(), "terraform", "terraform.tfvars.json")
    try:
        if os.path.exists(tf_vars_file):
            with open(tf_vars_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data["podinfo_replicas"] = replicas
        with open(tf_vars_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[Healer] Updated terraform.tfvars.json to {replicas} replicas. Applying Terraform...")
        return _run_terraform_cmd("apply")
    except Exception as e:
        return f"Failed to scale app: {e}"

# --- Training Tools ---

@tool
def list_mlflow_experiments(experiment_name: str = None) -> str:
    """
    List all MLflow experiments with recent runs and metrics.

    Args:
        experiment_name: Optional filter by experiment name (default: show all)

    Returns: Formatted list of experiments with recent runs and key metrics
    """
    try:
        client = MLflowClient("http://mlflow-server.ml-training.svc.cluster.local:5000")
        exps = client.list_experiments(experiment_name)
        if not exps:
            return "No experiments found"
        output = "MLflow Experiments:\n"
        for exp in exps:
            output += f"  {exp['name']} (ID: {exp['experiment_id']})\n"
            runs = client.get_experiment_runs(exp['name'], max_results=3)
            for run in runs:
                metrics = run.get('metrics', {})
                cost = metrics.get('total_cost_usd', 'N/A')
                acc = metrics.get('val_accuracy', 'N/A')
                output += f"    - {run['run_id'][:8]}: Acc={acc}, Cost=${cost}\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_experiment_metrics(run_id: str) -> str:
    """
    Get detailed metrics and parameters for a specific training run.

    Args:
        run_id: MLflow run ID (e.g. a1b2c3d4)

    Returns: Complete metrics, parameters, and run status
    """
    try:
        client = MLflowClient("http://mlflow-server.ml-training.svc.cluster.local:5000")
        details = client.get_run_details(run_id)
        output = f"Run {details['run_id'][:8]}\nStatus: {details['status']}\n"
        output += "\nMetrics:\n"
        for k, v in details['metrics'].items():
            output += f"  {k}: {v}\n"
        output += "\nParameters:\n"
        for k, v in details['params'].items():
            output += f"  {k}: {v}\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def trigger_training_job(model_type: str = "mnist", epochs: int = 3, lr: float = 0.001, batch_size: int = 64, gpu_enabled: bool = True) -> str:
    """
    Trigger a training job (MNIST or LLM fine-tuning).

    Args:
        model_type: Type of model to train - "mnist" or "llm" (default: mnist)
        epochs: Number of training epochs (default: 3)
        lr: Learning rate for optimizer (default: 0.001)
        batch_size: Batch size for training (default: 64)
        gpu_enabled: Use GPU acceleration (default: true)

    Returns: Job submission confirmation and monitoring instructions
    """
    try:
        # Coerce types in case LLM passes strings
        epochs = int(epochs) if epochs else 3
        lr = float(lr) if lr else 0.001
        batch_size = int(batch_size) if batch_size else 64
        model_type = str(model_type).lower().strip() if model_type else "mnist"

        if model_type not in ["mnist", "llm"]:
            return "Invalid model. Choose: 'mnist' or 'llm'"

        image = f"oppen-training-{model_type}:latest"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{model_type}-training-{ts}"

        mgr = TrainingJobManager()
        manifest = mgr.create_job_manifest(job_name, image, {"epochs": epochs, "lr": lr, "batch-size": batch_size}, gpu_enabled)
        name = mgr.submit_job(manifest)

        time.sleep(2)
        status = mgr.get_job_status(name)

        return f"Job submitted: {name}\nStatus: {status['status']}\nView: kubectl logs -n ml-training -l job-name={name} -f"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_training_status(job_name: str) -> str:
    """
    Check the status and logs of a training job.

    Args:
        job_name: Name of the training job (e.g. mnist-training-20260103-140000)

    Returns: Job status, pod counts, and recent logs
    """
    try:
        mgr = TrainingJobManager()
        status = mgr.get_job_status(job_name)
        output = f"Job: {job_name}\nStatus: {status['status']}\n"
        output += f"Active: {status['active']}, Succeeded: {status['succeeded']}, Failed: {status['failed']}\n"
        logs = mgr.get_job_logs(job_name)
        output += f"\nLogs (last 20 lines):\n{chr(10).join(logs.split(chr(10))[-20:])}"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def calculate_training_cost(job_name: str) -> str:
    """
    Calculate the cost of a completed training job.

    Args:
        job_name: Name of the completed training job

    Returns: Detailed cost breakdown (CPU, GPU, total)
    """
    try:
        mgr = TrainingJobManager()
        status = mgr.get_job_status(job_name)
        if status['status'] != 'completed':
            return f"Job not completed yet: {status['status']}"

        if not status['start_time'] or not status['completion_time']:
            return "Missing timestamps"

        start = datetime.fromisoformat(status['start_time'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(status['completion_time'].replace('Z', '+00:00'))
        duration_sec = (end - start).total_seconds()
        duration_hr = duration_sec / 3600

        gpu_rate, cpu_rate = 0.25, 0.05
        cpu_cost = duration_hr * 4 * cpu_rate
        gpu_cost = duration_hr * gpu_rate
        total = cpu_cost + gpu_cost

        return f"Duration: {duration_sec:.1f}s\nCPU: ${cpu_cost:.4f}\nGPU: ${gpu_cost:.4f}\nTotal: ${total:.4f}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Phase 6: Cost Optimization Tools ---

@tool
def recommend_cost_optimization(run_id: str) -> str:
    """
    Analyze a training run and recommend cost optimizations.

    Args:
        run_id: MLflow run ID to analyze

    Returns: List of optimization recommendations with potential savings
    """
    try:
        client = MLflowClient("http://mlflow-server.ml-training.svc.cluster.local:5000")
        run_details = client.get_run_details(run_id)

        recommender = CostRecommender()
        recommendations = recommender.analyze_run(run_details)

        if not recommendations:
            return "No optimization opportunities found for this run"

        output = f"Cost Optimization Recommendations for {run_id[:8]}:\n\n"
        for i, rec in enumerate(recommendations, 1):
            output += f"{i}. {rec['category'].upper()}\n"
            output += f"   Description: {rec['description']}\n"
            output += f"   Potential Savings: {rec['potential_savings']}\n"
            output += f"   Confidence: {rec['confidence']}\n"
            if 'action' in rec:
                output += f"   Action: {rec['action']}\n"
            output += "\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def forecast_training_cost(job_name: str, remaining_epochs: int = 1) -> str:
    """
    Forecast the cost for remaining training.

    Args:
        job_name: Name of the training job
        remaining_epochs: How many more epochs to train

    Returns: Cost forecast with confidence interval
    """
    try:
        mgr = TrainingJobManager()
        status = mgr.get_job_status(job_name)

        if status['status'] == 'running':
            # Try to estimate from logs
            logs = mgr.get_job_logs(job_name)
            # Parse logs for epoch timing (simplified)
            time_per_epoch = 300  # Default to 5 minutes if can't parse

            calculator = CostCalculator()
            remaining_seconds = remaining_epochs * time_per_epoch

            forecast = calculator.forecast_cost(
                current_epoch=0,
                total_epochs=remaining_epochs,
                time_per_epoch=time_per_epoch
            )

            output = f"Cost Forecast for {job_name}:\n"
            output += f"Remaining Epochs: {forecast['epochs_remaining']}\n"
            output += f"Time Remaining: {forecast['time_remaining_hours']} hours\n"
            output += f"Estimated Cost: ${forecast['forecast_total']:.4f}\n"
            output += f"Confidence: {forecast['confidence']*100:.0f}%\n"

            return output
        else:
            return f"Job not running: {status['status']}"

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def check_training_budget(budget_usd: float, epochs: int = 3, batch_size: int = 64) -> str:
    """
    Check if a training job will fit within a budget.

    Args:
        budget_usd: Maximum budget in USD
        epochs: Number of epochs to train
        batch_size: Batch size for training

    Returns: Budget feasibility check with recommendation
    """
    try:
        # Estimate time per epoch (varies with batch size)
        time_per_epoch = 300 / (batch_size / 64)  # Scales inversely with batch size

        tracker = BudgetTracker()
        result = tracker.check_budget_feasibility(
            budget=budget_usd,
            epochs=epochs,
            estimated_time_per_epoch=time_per_epoch,
            gpu_count=1,
            cpu_cores=4,
            memory_gb=8
        )

        output = f"Budget Check for {epochs} epochs (batch_size={batch_size}):\n"
        output += f"Budget: ${result['budget']:.2f}\n"
        output += f"Forecast Cost: ${result['forecast_cost']:.4f}\n"
        output += f"Utilization: {result['utilization_percent']:.1f}%\n"
        output += f"Status: {'✅ FEASIBLE' if result['feasible'] else '❌ EXCEEDS BUDGET'}\n"
        output += f"\nRecommendation: {result['recommendation']}\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_training_cost_report(start_date: str = None, end_date: str = None) -> str:
    """
    Generate a cost report for training runs.

    Args:
        start_date: Start date (YYYY-MM-DD), default: all
        end_date: End date (YYYY-MM-DD), default: all

    Returns: Formatted cost report with totals
    """
    try:
        client = MLflowClient("http://mlflow-server.ml-training.svc.cluster.local:5000")
        experiments = client.list_experiments()

        all_runs = []
        for exp in experiments:
            runs = client.get_experiment_runs(exp['name'], max_results=100)
            all_runs.extend(runs)

        tracker = BudgetTracker()
        report = tracker.generate_cost_report(all_runs, start_date, end_date)

        return report
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def compare_training_configs(config_json: str) -> str:
    """
    Compare cost and performance of different training configurations.

    Args:
        config_json: JSON string with list of configs.
                     Example: '[{"epochs": 3, "batch_size": 64}, {"epochs": 2, "batch_size": 128}]'

    Returns: Formatted comparison table
    """
    try:
        configs = json.loads(config_json)
        recommender = CostRecommender()
        comparison = recommender.compare_configs(configs)
        return comparison
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for configs"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Phase 6 Tier 2: Job Queue Tools ---

@tool
def queue_training_job(model_type: str = "mnist", priority: str = "normal", epochs: int = 3, batch_size: int = 64, run_name: str = None) -> str:
    """
    Queue a training job with priority scheduling.

    Args:
        model_type: "mnist" or "llm"
        priority: "urgent", "normal", or "background"
        epochs: Number of training epochs
        batch_size: Batch size
        run_name: Optional name for the job

    Returns: Queue confirmation with position and estimated wait time
    """
    try:
        if model_type not in ["mnist", "llm"]:
            return "Error: model_type must be 'mnist' or 'llm'"

        priority_enum = JobPriority(priority.lower())

        # Generate job ID
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"{model_type}-{priority}-{ts}"
        job_name = run_name or job_id

        # Submit to queue
        success, msg, queue_pos = _global_queue.submit_job(
            job_id=job_id,
            name=job_name,
            model_type=model_type,
            priority=priority_enum,
            epochs=epochs,
            batch_size=batch_size,
        )

        if not success:
            return f"Error: {msg}"

        # Get queue status for additional info
        status = _global_queue.get_queue_status()

        output = f"✅ Job Queued: {job_name}\n"
        output += f"Job ID: {job_id}\n"
        output += f"Priority: {priority.upper()}\n"
        output += f"Position in queue: {queue_pos}\n"
        output += f"Queue status: {status['queued_count']} queued, {status['running_count']}/{status['max_concurrent']} running\n"
        output += f"Estimated wait: {status['next_available_in_seconds']:.0f}s\n"

        return output

    except ValueError:
        return "Error: priority must be 'urgent', 'normal', or 'background'"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_queue_status() -> str:
    """
    Get current job queue status.

    Returns: Queue statistics and pending job info
    """
    try:
        status = _global_queue.get_queue_status()

        output = "📊 Job Queue Status:\n"
        output += f"Queued: {status['queued_count']} jobs\n"
        output += f"Running: {status['running_count']}/{status['max_concurrent']} slots\n"
        output += f"Completed: {status['completed_count']}\n"
        output += f"Failed: {status['failed_count']}\n"
        output += f"\nCapacity: {status['capacity_used']:.0f}% used\n"
        output += f"Next slot available in: {status['next_available_in_seconds']:.0f}s\n"
        output += f"Total estimated wait time: {status['total_wait_time_hours']:.1f} hours\n"
        output += f"Pending job cost: ${status['total_cost_pending']:.4f}\n"

        # List queued jobs
        queued_jobs = _global_queue.list_jobs(status="queued")
        if queued_jobs:
            output += f"\nQueued jobs (in priority order):\n"
            for i, job in enumerate(queued_jobs[:5], 1):  # Show top 5
                output += f"  {i}. {job['name']} ({job['model_type']}) - {job['priority']} priority\n"
            if len(queued_jobs) > 5:
                output += f"  ... and {len(queued_jobs) - 5} more\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def submit_batch_training_jobs(jobs_json: str) -> str:
    """
    Submit multiple training jobs as a batch.

    Args:
        jobs_json: JSON string with list of job configs.
                   Example: '[{"model_type": "mnist", "epochs": 2, "priority": "normal"}]'

    Returns: Batch submission confirmation and statistics
    """
    try:
        success, msg, stats = _global_queue.submit_batch(jobs_json)

        if not success:
            return f"Error: {msg}"

        output = f"✅ Batch Submitted!\n"
        output += f"Jobs submitted: {stats['submitted']}\n"
        output += f"Jobs failed: {stats['failed']}\n"
        output += f"Total estimated cost: ${stats['total_estimated_cost']:.4f}\n"
        if stats['first_job']:
            output += f"First job ID: {stats['first_job']}\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def cancel_queued_job(job_id: str) -> str:
    """
    Cancel a queued job.

    Args:
        job_id: ID of the job to cancel

    Returns: Cancellation confirmation
    """
    try:
        success, msg = _global_queue.cancel_job(job_id)

        if not success:
            return f"Error: {msg}"

        return f"✅ Job {job_id} cancelled"

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def list_queued_jobs() -> str:
    """
    List all queued jobs.

    Returns: Formatted list of pending jobs
    """
    try:
        jobs = _global_queue.list_jobs(status="queued")

        if not jobs:
            return "No queued jobs"

        output = "📋 Queued Training Jobs:\n"
        output += f"{'Job ID':<25} {'Model':<8} {'Priority':<12} {'Epochs':<8} {'Est. Time':<10}\n"
        output += "-" * 70 + "\n"

        for job in jobs:
            est_time = f"{job['estimated_duration_seconds']/60:.0f}m"
            output += f"{job['name']:<25} {job['model_type']:<8} {job['priority']:<12} {job['epochs']:<8} {est_time:<10}\n"

        total_cost = sum(
            (job['estimated_duration_seconds']/3600) * (0.25 if job['gpu_enabled'] else 0) +
            (job['estimated_duration_seconds']/3600) * 4 * 0.05
            for job in jobs
        )
        output += f"\nTotal pending cost: ${total_cost:.4f}\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_training_dashboard() -> str:
    """
    Get a real-time dashboard of training system metrics and health.

    Returns: Text-based dashboard with cost, resource, and health metrics
    """
    try:
        monitor = TrainingMonitor()
        return monitor.get_dashboard_summary()
    except Exception as e:
        return f"Error: {str(e)}"

# --- Phase 7: AutoML Tools ---

@tool
def start_automl_study(
    study_name: str,
    model_type: str = "mnist",
    n_trials: int = 20,
    epochs_max: int = 5,
    cost_budget_usd: float = None
) -> str:
    """
    Start an AutoML hyperparameter optimization study.

    Args:
        study_name: Unique name for this study
        model_type: "mnist" or "llm"
        n_trials: Number of trials to run (default: 20)
        epochs_max: Maximum epochs per trial (default: 5)
        cost_budget_usd: Optional cost budget limit

    Returns: Study launch confirmation and tracking info
    """
    try:
        from app.training.automl_optimizer import (
            OptunaStudyManager,
            StudyConfig,
            SearchSpace
        )

        # Validate inputs
        if model_type not in ["mnist", "llm"]:
            return "Error: model_type must be 'mnist' or 'llm'"

        # Configure search space
        search_space = SearchSpace(
            lr_min=1e-5,
            lr_max=1e-2,
            batch_size_choices=[32, 64, 128, 256],
            epochs_min=1,
            epochs_max=epochs_max,
        )

        # Configure study
        config = StudyConfig(
            study_name=study_name,
            model_type=model_type,
            n_trials=n_trials,
            max_concurrent_trials=3,
            search_space=search_space,
            cost_budget_usd=cost_budget_usd,
        )

        # Start optimization
        manager = OptunaStudyManager()
        study = manager.run_optimization(config, _global_queue)

        best = study.best_trial

        output = f"[OK] AutoML Study Complete: {study_name}\n"
        output += f"Model: {model_type}\n"
        output += f"Trials completed: {len(study.trials)}\n"
        output += f"Best val_accuracy: {study.best_value:.4f}\n"
        output += f"\nBest hyperparameters:\n"
        for k, v in study.best_params.items():
            output += f"  {k}: {v}\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_automl_study_status(study_name: str) -> str:
    """
    Get current status of an AutoML study.

    Args:
        study_name: Name of the study

    Returns: Study progress and current best results
    """
    try:
        from app.training.automl_optimizer import OptunaStudyManager

        manager = OptunaStudyManager()
        summary = manager.get_study_summary(study_name)

        if "error" in summary:
            return f"Error: {summary['error']}"

        output = f"[Info] AutoML Study: {study_name}\n"
        output += f"Trials: {summary['n_trials']}\n"
        output += f"Best value: {summary['best_value']:.4f}\n"
        output += f"Best trial: #{summary['best_trial_number']}\n"
        output += f"Direction: {summary['direction']}\n"
        output += f"\nBest params:\n"
        for k, v in summary['best_params'].items():
            output += f"  {k}: {v}\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def list_automl_studies() -> str:
    """
    List all AutoML studies.

    Returns: List of study names with status
    """
    try:
        from app.training.automl_optimizer import OptunaStudyManager

        manager = OptunaStudyManager()
        studies = manager.list_studies()

        if not studies:
            return "No AutoML studies found"

        output = "[Info] AutoML Studies:\n"
        for study_name in studies:
            summary = manager.get_study_summary(study_name)
            if "error" not in summary:
                output += f"\n  {study_name}:\n"
                output += f"    Trials: {summary['n_trials']}\n"
                output += f"    Best: {summary['best_value']:.4f}\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def compare_automl_trials(study_name: str, top_k: int = 5) -> str:
    """
    Compare top trials from an AutoML study.

    Args:
        study_name: Name of the study
        top_k: Number of top trials to show (default: 5)

    Returns: Comparison table of best trials
    """
    try:
        import optuna
        from app.training.automl_optimizer import STORAGE_PATH

        study = optuna.load_study(study_name, storage=STORAGE_PATH)

        # Get top trials
        sorted_trials = sorted(
            study.trials,
            key=lambda t: t.value if t.value else 0,
            reverse=True
        )[:top_k]

        output = f"[Info] Top {top_k} Trials for {study_name}:\n"
        output += "=" * 80 + "\n"
        output += f"{'Trial':<8} {'Val Acc':<10} {'LR':<12} {'Batch':<8} {'Epochs':<8}\n"
        output += "-" * 80 + "\n"

        for trial in sorted_trials:
            if trial.value is None:
                continue
            output += f"#{trial.number:<7} {trial.value:<10.4f} "
            output += f"{trial.params.get('lr', 'N/A'):<12.2e} "
            output += f"{trial.params.get('batch_size', 'N/A'):<8} "
            output += f"{trial.params.get('epochs', 'N/A'):<8}\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_automl_best_params(study_name: str) -> str:
    """
    Get best hyperparameters from a study (ready to use).

    Args:
        study_name: Name of the study

    Returns: Best hyperparameters in copyable format
    """
    try:
        import optuna
        from app.training.automl_optimizer import STORAGE_PATH

        study = optuna.load_study(study_name, storage=STORAGE_PATH)
        best = study.best_params

        output = f"[Info] Best Hyperparameters for {study_name}:\n\n"
        output += "Use with trigger_training_job():\n"
        output += f"  epochs={best.get('epochs', 3)}\n"
        output += f"  lr={best.get('lr', 0.001)}\n"
        output += f"  batch_size={best.get('batch_size', 64)}\n"
        output += f"\nFull command:\n"
        output += f"trigger_training_job("
        output += f"epochs={best.get('epochs', 3)}, "
        output += f"lr={best.get('lr', 0.001)}, "
        output += f"batch_size={best.get('batch_size', 64)})\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"

# --- Phase 8: Distributed Training (DDP) Simulation Tools ---

@tool
def run_ddp_simulation(
    world_size: int = 4,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.001
) -> str:
    """
    Run local DDP simulation for learning distributed training concepts.

    Uses multiprocessing to simulate distributed training on CPU/single GPU.
    Demonstrates: data sharding (DistributedSampler), rank-based logging,
    and multi-worker training without requiring actual multi-GPU hardware.

    Args:
        world_size: Number of worker processes to simulate (2-4 typical)
        epochs: Number of training epochs (1-3 for quick demo)
        batch_size: Batch size per worker (32-64 typical)
        lr: Learning rate (default 0.001)

    Returns:
        Training completion status and summary

    Examples:
        - Quick 2-worker test: world_size=2, epochs=1
        - Realistic 4-worker: world_size=4, epochs=3
    """
    import subprocess
    import sys

    try:
        cmd = [
            sys.executable,
            "scripts/training/train_mnist_ddp_sim.py",
            f"--world-size", str(world_size),
            f"--epochs", str(epochs),
            f"--batch-size", str(batch_size),
            f"--lr", str(lr)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            return f"""[OK] DDP Simulation Complete!

Configuration:
  Workers: {world_size}
  Epochs: {epochs}
  Batch size per worker: {batch_size}
  Total batch size: {batch_size * world_size}
  Learning rate: {lr}

Output:
{result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}

Key Concepts Demonstrated:
1. Each worker processed 1/{world_size} of the training data (DistributedSampler)
2. Only rank 0 logged metrics to MLflow
3. Models converged together across workers

Ready to learn more? Try:
  - explain_ddp_concepts() for detailed explanations
  - compare_ddp_vs_single() to see benefits
"""
        else:
            return f"[ERROR] Simulation failed:\n{result.stderr}"

    except Exception as e:
        return f"[ERROR] Failed to run simulation: {str(e)}"

@tool
def explain_ddp_concepts() -> str:
    """
    Educational explanation of Distributed Data Parallel (DDP) training concepts.

    Covers: RANK, WORLD_SIZE, DistributedSampler, gradient synchronization,
    and the Gloo backend for CPU-friendly simulation.

    Returns:
        Detailed explanation of DDP concepts
    """
    return """
[Info] Distributed Data Parallel (DDP) Training Concepts

FUNDAMENTAL CONCEPTS:

1. **RANK** - Worker Process ID
   - Each worker is a separate Python process
   - RANK = 0 is the "master" (logs metrics, saves models)
   - RANK = 1, 2, 3... are worker processes
   - Example with 4 workers: RANK ∈ {0, 1, 2, 3}

2. **WORLD_SIZE** - Total Number of Workers
   - Total parallel processes running
   - Must be same for all workers
   - Example: WORLD_SIZE = 4 (4 parallel training processes)

3. **LOCAL_RANK** - GPU/Device ID (for multi-GPU)
   - Maps worker to specific GPU
   - Example: 2 nodes × 2 GPUs each
     - Node 0: RANK=0,1 with LOCAL_RANK=0,1
     - Node 1: RANK=2,3 with LOCAL_RANK=0,1

DATA DISTRIBUTION:

4. **DistributedSampler**
   - Automatically shards dataset across ranks
   - Each rank sees different data subset
   - Example with WORLD_SIZE=4:
     - Rank 0 sees samples [0, 4, 8, 12, ...]
     - Rank 1 sees samples [1, 5, 9, 13, ...]
     - Rank 2 sees samples [2, 6, 10, 14, ...]
     - Rank 3 sees samples [3, 7, 11, 15, ...]
   - No data duplication or overlap

5. **Gradient Synchronization**
   - Each rank computes gradients on its data shard
   - DDP averages gradients across all ranks
   - All ranks update model with same averaged gradient
   - Result: All workers converge to same model

BACKENDS:

6. **Gloo Backend** (Used for Simulation)
   - CPU-friendly communication
   - Works on Windows/Linux/Mac without multi-GPU
   - Slightly slower but great for learning
   - Uses TCP/IP or shared memory

7. **NCCL Backend** (For Real Multi-GPU)
   - GPU-optimized communication
   - Uses NVIDIA NVLink for ultra-fast sync
   - Only works with CUDA GPUs
   - 10-100x faster than Gloo

WORKFLOW EXAMPLE (4 Workers):

Setup:
  1. RANK=0 initializes process group with WORLD_SIZE=4
  2. RANK=1, 2, 3 connect to group
  3. All get different data via DistributedSampler

Training Loop:
  1. Each rank loads batch from its data shard
  2. Each rank forward pass → local loss
  3. Each rank backward pass → local gradients
  4. DDP synchronizes: average gradients across 4 ranks
  5. All ranks update model with averaged gradient
  6. Repeat for next batch

Benefits:
  ✓ 4x data throughput (4 workers processing in parallel)
  ✓ 3x-4x speedup if GPUs available
  ✓ Same model converges as single-GPU training
  ✓ Automatic load balancing

LOGGING:

8. **Rank-Based Logging**
   - Only rank 0 logs to MLflow/files
   - Prevents duplicate metrics
   - Prevents file conflicts
   - Other ranks remain silent

TRY IT YOURSELF:

  python scripts/training/train_mnist_ddp_sim.py --world-size 4 --epochs 1

This demonstrates all concepts without multi-GPU hardware!
"""

@tool
def compare_ddp_vs_single() -> str:
    """
    Compare single-process vs DDP simulation training.

    Shows conceptual differences, benefits, and when to use DDP.

    Returns:
        Comparison table and analysis
    """
    return """
[Info] Single-Process vs DDP Simulation Comparison

SINGLE-PROCESS TRAINING:

Process Model:
  ├─ Single Python process
  └─ Single thread processing batches sequentially

Data Processing:
  ├─ Reads all 60,000 MNIST samples in order
  └─ One batch at a time: batch 0, batch 1, batch 2...

Gradient Computation:
  ├─ Forward pass on batch → loss
  └─ Backward pass on batch → gradients

Metrics:
  ├─ Logs every batch
  └─ One model being trained

Example Timeline (Simplified):
  Time 0s:   Load batch 0, forward, backward
  Time 1s:   Load batch 1, forward, backward
  Time 2s:   Load batch 2, forward, backward
  ...
  Time 30s:  Epoch complete


DDP SIMULATION (4 Workers):

Process Model:
  ├─ Process 0 (Rank 0)  ─ Device 0
  ├─ Process 1 (Rank 1)  ─ Device 1
  ├─ Process 2 (Rank 2)  ─ Device 2
  └─ Process 3 (Rank 3)  ─ Device 3

Data Processing:
  ├─ Rank 0 reads samples [0, 4, 8, 12, ...] (15,000 samples)
  ├─ Rank 1 reads samples [1, 5, 9, 13, ...] (15,000 samples)
  ├─ Rank 2 reads samples [2, 6, 10, 14, ...] (15,000 samples)
  └─ Rank 3 reads samples [3, 7, 11, 15, ...] (15,000 samples)

Gradient Computation:
  1. All ranks compute gradients on their data shard (parallel!)
  2. Gradients synchronized via DDP (average)
  3. All ranks update with same averaged gradient

Metrics:
  ├─ Only rank 0 logs to MLflow
  ├─ Prevents duplicate runs
  └─ Single training run with all workers contributing

Example Timeline (Simplified):
  Time 0s:   All ranks load batch 0, compute grad in parallel
  Time 0.25s: Sync gradients, update all models
  Time 0.5s: All ranks load batch 1, compute grad in parallel
  ...
  Time 7.5s: Epoch complete (4x faster with 4 workers!)


SIDE-BY-SIDE COMPARISON:

┌────────────────────┬─────────────────┬──────────────────┐
│ Aspect             │ Single-Process  │ DDP (4 workers)  │
├────────────────────┼─────────────────┼──────────────────┤
│ Processes          │ 1               │ 4                │
│ GPU Usage          │ 1 GPU or CPU    │ 4 GPUs or CPU    │
│ Data per Worker    │ All 60k samples │ 15k each (1/4)   │
│ Speedup            │ 1x baseline     │ 3-4x (w/GPU)     │
│ Gradient Sync      │ N/A             │ After each batch │
│ Logging            │ Single run      │ Single run       │
│ Model Convergence  │ Standard        │ Same as single   │
│ Learning Curve     │ Smooth          │ Smooth           │
│ Final Accuracy     │ 98.5%           │ 98.5% (same!)    │
└────────────────────┴─────────────────┴──────────────────┘


BENEFITS OF DDP:

✓ **Parallelism**: 4 workers process data in parallel
  - Single-GPU: 1 batch at a time
  - DDP: 4 batches in parallel (if 4 GPUs)

✓ **Scalability**: Add more GPUs for linear speedup
  - 8 GPUs → 7-8x speedup (not perfect due to comm overhead)
  - 16 GPUs → 14-15x speedup

✓ **Same Convergence**: Mathematical proof that distributed
  - training converges to same solution as single GPU
  - Final accuracy identical
  - Learning curves similar

✓ **Cost Efficiency**: For same time, DDP cheaper if batch
  - doubles per GPU (larger batches more efficient)

✓ **Production Standard**: Industry standard for large models
  - GPT, BERT, ResNet training uses DDP or similar


WHEN TO USE DDP:

Single-Process is Fine For:
  ✓ Small datasets (< 100GB)
  ✓ Small models (< 1B parameters)
  ✓ Learning/experimentation
  ✓ Single GPU available

Use DDP When:
  ✓ Multiple GPUs available
  ✓ Large models (1B+ parameters)
  ✓ Large datasets (100GB+)
  ✓ Production training
  ✓ Need faster training time
  ✓ Cost not critical (speedup > cost increase)


HOW THE SIMULATION HELPS:

Without Multi-GPU Hardware:
  ✓ Learn DDP concepts (RANK, WORLD_SIZE, DistributedSampler)
  ✓ Understand gradient synchronization
  ✓ Practice rank-based logging patterns
  ✓ Test code that will work on real multi-GPU

Key Insight:
  The DDP simulation teaches concepts but NOT performance.
  Real multi-GPU DDP achieves 3-8x speedup.
  Simulation runs at same speed as single-process.
  Same code works for both!


TRY BOTH:

Single-process training:
  python scripts/training/train_mnist.py --epochs 1

DDP simulation (4 workers):
  python scripts/training/train_mnist_ddp_sim.py --world-size 4 --epochs 1

Compare the outputs and metrics!
"""

# --- Phase 9: Kubeflow PyTorchJob Integration Tools ---

@tool
def submit_distributed_training(
    model_type: str = "mnist",
    world_size: int = 2,
    epochs: int = 3,
    lr: float = 0.001,
    batch_size: int = 64,
    gpu_per_replica: int = 1
) -> str:
    """
    Submit a distributed training job using Kubeflow PyTorchJob.

    Creates and submits a PyTorchJob manifest to Kubernetes cluster for
    multi-worker distributed training using NCCL backend.

    Args:
        model_type: "mnist" or "llm" (default: mnist)
        world_size: Number of distributed workers (2-8, default: 2)
        epochs: Training epochs (default: 3)
        lr: Learning rate (default: 0.001)
        batch_size: Batch size per worker (default: 64)
        gpu_per_replica: GPUs per worker (default: 1)

    Returns:
        Job submission confirmation with status and monitoring commands

    Examples:
        - Quick 2-worker MNIST: world_size=2, epochs=1
        - Full 4-worker training: world_size=4, epochs=3
    """
    from datetime import datetime
    from app.training.job_manager import TrainingJobManager

    if world_size < 2:
        return "[ERROR] world_size must be >= 2 for distributed training"

    if world_size > 8:
        return "[ERROR] world_size must be <= 8"

    try:
        image = f"oppen-training-{model_type}-ddp:latest"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{model_type}-ddp-{world_size}w-{ts}"

        mgr = TrainingJobManager()
        manifest = mgr.create_pytorch_job_manifest(
            job_name=job_name,
            image=image,
            training_args={"epochs": epochs, "lr": lr, "batch-size": batch_size},
            world_size=world_size,
            gpu_per_replica=gpu_per_replica
        )

        name = mgr.submit_pytorch_job(manifest)
        time.sleep(2)
        status = mgr.get_pytorch_job_status(name)

        return f"""[OK] Distributed Training Job Submitted: {name}

Model: {model_type}
World Size: {world_size} workers
Total GPUs: {world_size * gpu_per_replica}
Batch size per worker: {batch_size}
Effective batch size: {batch_size * world_size}
Status: {status['status']}

Monitor with:
  get_distributed_status('{name}')
  kubectl get pytorchjobs -n ml-training -w
  kubectl logs -n ml-training -l pytorch-job-name={name} -f

Note: Only rank 0 logs to MLflow to prevent duplicates"""

    except Exception as e:
        return f"[ERROR] Failed to submit job: {str(e)}"


@tool
def get_distributed_status(job_name: str) -> str:
    """
    Get status and logs from a distributed training job (PyTorchJob).

    Retrieves current job status, replica information, and recent logs
    from master and worker pods.

    Args:
        job_name: PyTorchJob name (e.g., "mnist-ddp-4w-20240115-143022")

    Returns:
        Job status, replica details, and recent logs from all pods
    """
    from app.training.job_manager import TrainingJobManager

    try:
        mgr = TrainingJobManager()
        status = mgr.get_pytorch_job_status(job_name)
        logs = mgr.get_pytorch_job_logs(job_name, tail_lines=20)

        return f"""[INFO] Distributed Job Status: {job_name}

Status: {status['status']}
Master: {status['master_status']}
Workers: {status['worker_count']} total, {status['workers_succeeded']} succeeded

Recent Logs:
{'='*70}
{logs}
{'='*70}

To stream live logs:
  kubectl logs -n ml-training -f -l pytorch-job-name={job_name}

To delete job:
  kubectl delete pytorchjob {job_name} -n ml-training"""

    except Exception as e:
        return f"[ERROR] Failed to get status: {str(e)}"


@tool
def estimate_distributed_speedup(
    model_type: str = "mnist",
    epochs: int = 3,
    world_sizes: str = "[1,2,4,8]"
) -> str:
    """
    Estimate training time and cost for different world sizes (distributed configs).

    Shows time and cost estimates for single-node vs 2-worker vs 4-worker
    vs 8-worker configurations to help choose optimal distributed setup.

    Args:
        model_type: "mnist" or "llm" (default: mnist)
        epochs: Number of training epochs (default: 3)
        world_sizes: JSON array of world sizes (default: "[1,2,4,8]")

    Returns:
        Comparison table with time, cost, and speedup for each config

    Examples:
        - Compare MNIST with 2,4,8 workers: estimate_distributed_speedup(world_sizes="[2,4,8]")
        - LLM with fewer options: estimate_distributed_speedup("llm", world_sizes="[1,2,4]")
    """
    import json

    try:
        sizes = json.loads(world_sizes)
    except json.JSONDecodeError:
        return f"[ERROR] Invalid JSON for world_sizes: {world_sizes}"

    if model_type == "mnist":
        base_time_per_epoch = 300  # seconds
    elif model_type == "llm":
        base_time_per_epoch = 600  # seconds
    else:
        return "[ERROR] model_type must be 'mnist' or 'llm'"

    base_duration = base_time_per_epoch * epochs

    output = f"[INFO] Distributed Training Speedup Estimation\n"
    output += f"Model: {model_type.upper()}\n"
    output += f"Epochs: {epochs}\n"
    output += f"{'='*90}\n"
    output += f"{'Config':<20} {'Duration':<15} {'Cost (USD)':<15} {'Speedup':<15} {'Efficiency':<15}\n"
    output += f"{'-'*90}\n"

    for world_size in sizes:
        if world_size < 1:
            continue

        if world_size == 1:
            speedup = 1.0
            efficiency = 100.0
            duration = base_duration
        else:
            speedup = world_size * 0.7  # 70% scaling efficiency
            efficiency = (speedup / world_size) * 100
            duration = base_duration / speedup

        # Cost calculation: $0.25/GPU/hour, $0.05/CPU core/hour
        # Assume: 1 GPU per replica, 4 CPU cores per replica
        hours_per_worker = duration / 3600
        gpu_cost_per_worker = hours_per_worker * 1.0 * 0.25
        cpu_cost_per_worker = hours_per_worker * 4 * 0.05
        total_cost = (gpu_cost_per_worker + cpu_cost_per_worker) * world_size

        config_name = f"{world_size}-worker" if world_size > 1 else "single-node"
        duration_str = f"{duration/60:.1f}m" if duration < 3600 else f"{duration/3600:.1f}h"
        cost_str = f"${total_cost:.2f}"
        speedup_str = f"{speedup:.1f}x"
        efficiency_str = f"{efficiency:.0f}%"

        output += f"{config_name:<20} {duration_str:<15} {cost_str:<15} {speedup_str:<15} {efficiency_str:<15}\n"

    output += f"{'-'*90}\n"
    output += f"\nNote: Assumes 70% scaling efficiency (typical for distributed training)\n"
    output += f"Actual speedup depends on: network latency, synchronization overhead, hardware\n"
    output += f"\nCost breakdown:\n"
    output += f"  - GPU: $0.25/GPU/hour\n"
    output += f"  - CPU: $0.05/core/hour (4 cores assumed per worker)\n"

    return output

# --- Memory Tools Factory ---
# Since memory tools need the memory instance, we create them dynamically or assume singleton.
# Let's create a factory function.

def get_memory_tools(memory: MemoryManager):
    @tool
    def search_memory(query: str) -> str:
        """Searches the incident database for past similar alerts and solutions."""
        print(f"\n[Memory] Searching for: {query}")
        results = memory.search_incidents(query)
        if not results:
            return "No similar past incidents found."
        return f"Found similar past incidents:\n{json.dumps(results, indent=2)}"

    @tool
    def save_memory(description: str, resolution: str) -> str:
        """Saves the current incident and its resolution to memory for future reference."""
        print(f"\n[Memory] Saving incident: {description[:50]}...")
        doc_id = memory.save_incident(description, resolution)
        return f"Incident saved with ID: {doc_id}"
    
    return [search_memory, save_memory]
