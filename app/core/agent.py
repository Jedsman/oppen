from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from app.core.memory import MemoryManager
from app.core.tools import (
    list_containers, terraform_run, list_pods,
    get_k8s_events, query_prometheus, scale_app,
    list_mlflow_experiments, get_experiment_metrics,
    trigger_training_job, get_training_status, calculate_training_cost,
    recommend_cost_optimization, forecast_training_cost, check_training_budget,
    get_training_cost_report, compare_training_configs,
    queue_training_job, get_queue_status, submit_batch_training_jobs,
    cancel_queued_job, list_queued_jobs,
    get_training_dashboard,
    start_automl_study, get_automl_study_status, list_automl_studies,
    compare_automl_trials, get_automl_best_params,
    run_ddp_simulation, explain_ddp_concepts, compare_ddp_vs_single,
    submit_distributed_training, get_distributed_status, estimate_distributed_speedup,
    get_memory_tools
)

# Singleton memory instance for the application lifetime
memory_instance = MemoryManager()

def get_tools():
    """Aggregates all tools including dynamic memory tools."""
    basic_tools = [
        # Diagnostic & remediation
        list_containers, terraform_run, list_pods,
        get_k8s_events, query_prometheus, scale_app,
        # Training management (Phase 5)
        list_mlflow_experiments, get_experiment_metrics,
        trigger_training_job, get_training_status, calculate_training_cost,
        # Cost optimization (Phase 6 Tier 1)
        recommend_cost_optimization, forecast_training_cost, check_training_budget,
        get_training_cost_report, compare_training_configs,
        # Job queue & batch scheduling (Phase 6 Tier 2)
        queue_training_job, get_queue_status, submit_batch_training_jobs,
        cancel_queued_job, list_queued_jobs,
        # Monitoring (Phase 6 Tier 2)
        get_training_dashboard,
        # AutoML (Phase 7)
        start_automl_study, get_automl_study_status, list_automl_studies,
        compare_automl_trials, get_automl_best_params,
        # Distributed Training (Phase 8)
        run_ddp_simulation, explain_ddp_concepts, compare_ddp_vs_single,
        # Kubeflow PyTorchJob (Phase 9)
        submit_distributed_training, get_distributed_status, estimate_distributed_speedup
    ]
    mem_tools = get_memory_tools(memory_instance)
    return basic_tools + mem_tools

def get_agent():
    """Creates the ReAct agent with all capabilities."""
    tools = get_tools()
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    return create_react_agent(llm, tools)
