"""Phase 6: Cost Optimization Engine

Provides dynamic cost calculation, cost recommendations, and budget forecasting
for training jobs with time-based pricing, efficiency metrics, and optimization.
"""

import os
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class CostConfig:
    """Cost model configuration"""
    gpu_hourly_rate: float = 0.25           # $/hour
    cpu_hourly_rate_per_core: float = 0.05 # $/hour per core
    memory_hourly_rate_gb: float = 0.01     # $/hour per GB

    # Time-based pricing
    peak_hours_start: int = 6               # UTC hour when peak starts
    peak_hours_end: int = 22                # UTC hour when peak ends
    peak_rate_multiplier: float = 1.0       # Full price during peak
    offpeak_rate_multiplier: float = 0.5    # 50% discount off-peak

    # Job priority multipliers
    urgent_cost_multiplier: float = 1.5     # Urgent jobs cost 50% more (priority)
    batch_cost_discount: float = 0.9        # Batch jobs 10% cheaper

    # Spot instance simulation
    spot_discount: float = 0.7              # 70% cheaper
    spot_interruption_rate: float = 0.05    # 5% chance of interruption

    # Forecasting
    forecast_confidence: float = 0.95       # 95% confidence interval

    @classmethod
    def from_env(cls) -> "CostConfig":
        """Load config from environment variables"""
        return cls(
            gpu_hourly_rate=float(os.getenv("GPU_HOURLY_RATE", "0.25")),
            cpu_hourly_rate_per_core=float(os.getenv("CPU_HOURLY_RATE", "0.05")),
            memory_hourly_rate_gb=float(os.getenv("MEMORY_HOURLY_RATE", "0.01")),
            peak_hours_start=int(os.getenv("PEAK_HOURS_START", "6")),
            peak_hours_end=int(os.getenv("PEAK_HOURS_END", "22")),
            peak_rate_multiplier=float(os.getenv("PEAK_RATE_MULTIPLIER", "1.0")),
            offpeak_rate_multiplier=float(os.getenv("OFFPEAK_RATE_MULTIPLIER", "0.5")),
        )


class CostCalculator:
    """Calculate training costs with dynamic pricing"""

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig.from_env()

    def _get_time_multiplier(self, dt: datetime) -> float:
        """Get price multiplier based on time of day (UTC)"""
        hour = dt.hour

        # Check if in peak hours
        if self.config.peak_hours_start <= hour < self.config.peak_hours_end:
            return self.config.peak_rate_multiplier
        else:
            return self.config.offpeak_rate_multiplier

    def calculate_duration_cost(
        self,
        duration_seconds: float,
        gpu_count: int = 1,
        cpu_cores: int = 4,
        memory_gb: int = 8,
        start_time: Optional[datetime] = None,
        priority: str = "normal",
        use_spot: bool = False
    ) -> Dict[str, float]:
        """
        Calculate cost for a training job.

        Args:
            duration_seconds: How long the job ran
            gpu_count: Number of GPUs used
            cpu_cores: Number of CPU cores used
            memory_gb: Memory in GB used
            start_time: When job started (for time-based pricing). Default: now
            priority: Job priority (urgent/normal/background)
            use_spot: Whether using spot instances

        Returns:
            Dict with gpu_cost, cpu_cost, memory_cost, total_cost, effective_rate
        """
        if start_time is None:
            start_time = datetime.utcnow()

        duration_hours = duration_seconds / 3600

        # Get time-based multiplier
        time_multiplier = self._get_time_multiplier(start_time)

        # Calculate individual costs
        gpu_cost = gpu_count * duration_hours * self.config.gpu_hourly_rate * time_multiplier
        cpu_cost = cpu_cores * duration_hours * self.config.cpu_hourly_rate_per_core * time_multiplier
        memory_cost = memory_gb * duration_hours * self.config.memory_hourly_rate_gb * time_multiplier

        total_cost = gpu_cost + cpu_cost + memory_cost

        # Apply priority multiplier
        if priority == "urgent":
            total_cost *= self.config.urgent_cost_multiplier
        elif priority == "background":
            total_cost *= 0.8  # Background jobs get 20% discount

        # Apply spot discount if using spot instances
        if use_spot:
            total_cost *= self.config.spot_discount

        # Determine effective hourly rate
        effective_rate = total_cost / duration_hours if duration_hours > 0 else 0

        return {
            "gpu_cost": round(gpu_cost, 4),
            "cpu_cost": round(cpu_cost, 4),
            "memory_cost": round(memory_cost, 4),
            "total_cost": round(total_cost, 4),
            "effective_hourly_rate": round(effective_rate, 4),
            "duration_hours": round(duration_hours, 2),
            "time_multiplier": time_multiplier,
            "priority": priority,
            "use_spot": use_spot,
        }

    def forecast_cost(
        self,
        current_epoch: int,
        total_epochs: int,
        time_per_epoch: float,
        gpu_count: int = 1,
        cpu_cores: int = 4,
        memory_gb: int = 8,
        start_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Forecast the total cost for remaining training.

        Args:
            current_epoch: Current epoch (0-indexed)
            total_epochs: Total epochs to train
            time_per_epoch: Average seconds per epoch
            gpu_count: Number of GPUs
            cpu_cores: Number of CPU cores
            memory_gb: Memory in GB
            start_time: When training started

        Returns:
            Dict with forecast_completed, forecast_total, forecast_remaining, confidence
        """
        remaining_epochs = total_epochs - current_epoch
        remaining_seconds = remaining_epochs * time_per_epoch

        # Calculate cost for completed portion
        completed_seconds = current_epoch * time_per_epoch
        completed_cost = self.calculate_duration_cost(
            completed_seconds, gpu_count, cpu_cores, memory_gb, start_time
        )["total_cost"]

        # Estimate cost for remaining portion (use average multiplier)
        remaining_cost = self.calculate_duration_cost(
            remaining_seconds, gpu_count, cpu_cores, memory_gb, start_time
        )["total_cost"]

        total_forecast = completed_cost + remaining_cost

        return {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "epochs_remaining": remaining_epochs,
            "cost_completed": round(completed_cost, 4),
            "cost_remaining": round(remaining_cost, 4),
            "forecast_total": round(total_forecast, 4),
            "time_remaining_hours": round(remaining_seconds / 3600, 2),
            "confidence": self.config.forecast_confidence,
        }


class CostRecommender:
    """Generate cost optimization recommendations"""

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig.from_env()
        self.calculator = CostCalculator(config)

    def analyze_run(self, run_metrics: Dict) -> List[Dict[str, str]]:
        """
        Analyze a completed training run and generate recommendations.

        Args:
            run_metrics: MLflow metrics dict with keys:
                - duration_seconds: How long training took
                - train_loss, val_accuracy, etc.
                - Parameters: epochs, batch_size, lr

        Returns:
            List of recommendations with description and potential_savings
        """
        recommendations = []

        duration = run_metrics.get("duration_seconds", 0)
        epochs = run_metrics.get("params", {}).get("epochs", 1)
        batch_size = run_metrics.get("params", {}).get("batch_size", 64)

        time_per_epoch = duration / epochs if epochs > 0 else 0

        # Recommendation 1: Batch size efficiency
        if batch_size < 128:
            recommendations.append({
                "category": "batch_size",
                "description": f"Current batch size: {batch_size}. Try 128-256 for better GPU utilization.",
                "potential_savings": "15-20%",
                "confidence": "high",
                "action": f"Retry with --batch-size 128"
            })

        # Recommendation 2: Training time efficiency
        if time_per_epoch > 300:  # > 5 minutes per epoch
            recommendations.append({
                "category": "efficiency",
                "description": f"Each epoch takes {time_per_epoch:.1f}s. Consider data augmentation or model simplification.",
                "potential_savings": "10-30%",
                "confidence": "medium",
                "action": "Profile training loop for bottlenecks"
            })

        # Recommendation 3: Schedule optimization
        recommendations.append({
            "category": "scheduling",
            "description": "Run non-urgent jobs during off-peak hours (22:00-06:00 UTC) for 50% discount.",
            "potential_savings": "50%",
            "confidence": "high",
            "action": "Schedule with --priority background --start-after 22:00"
        })

        # Recommendation 4: Spot instances
        recommendations.append({
            "category": "spot_instances",
            "description": f"Use spot instances for {epochs} epochs (70% cheaper, {self.config.spot_interruption_rate*100:.0f}% interruption risk).",
            "potential_savings": "70%",
            "confidence": "high",
            "risk": "interruption",
            "action": "Retry with --use-spot true"
        })

        # Recommendation 5: Model optimization
        if "val_accuracy" in run_metrics:
            val_acc = run_metrics.get("val_accuracy", 0)
            if val_acc > 95:  # Already good accuracy
                recommendations.append({
                    "category": "early_stopping",
                    "description": f"Accuracy already {val_acc}%. Could stop early or use smaller model.",
                    "potential_savings": "20-40%",
                    "confidence": "medium",
                    "action": "Implement early stopping callback"
                })

        return recommendations

    def compare_configs(self, configs: List[Dict]) -> str:
        """
        Compare cost/performance of multiple training configurations.

        Args:
            configs: List of config dicts with epochs, batch_size, lr, etc.

        Returns:
            Formatted comparison table
        """
        result = "Configuration Comparison:\n"
        result += "=" * 80 + "\n"
        result += f"{'Config':<15} {'Epochs':<8} {'Batch':<8} {'Est.Time':<12} {'Est.Cost':<10}\n"
        result += "-" * 80 + "\n"

        for i, config in enumerate(configs):
            epochs = config.get("epochs", 3)
            batch_size = config.get("batch_size", 64)
            lr = config.get("lr", 0.001)

            # Estimate: batch size and lr affect training speed
            time_per_epoch = 300 / (batch_size / 64) * (0.001 / lr if lr != 0 else 1)
            total_seconds = time_per_epoch * epochs

            cost = self.calculator.calculate_duration_cost(total_seconds)["total_cost"]

            result += f"Config {i+1:<8} {epochs:<8} {batch_size:<8} {total_seconds/60:>8.1f}min  ${cost:<9.2f}\n"

        return result


class BudgetTracker:
    """Track and enforce training budgets"""

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig.from_env()
        self.calculator = CostCalculator(config)

    def check_budget_feasibility(
        self,
        budget: float,
        epochs: int,
        estimated_time_per_epoch: float,
        gpu_count: int = 1,
        cpu_cores: int = 4,
        memory_gb: int = 8
    ) -> Dict:
        """
        Check if training fits within budget.

        Returns:
            Dict with feasible, cost_forecast, utilization_percent, recommendation
        """
        total_seconds = epochs * estimated_time_per_epoch
        cost_info = self.calculator.calculate_duration_cost(
            total_seconds, gpu_count, cpu_cores, memory_gb
        )
        cost_forecast = cost_info["total_cost"]
        utilization = (cost_forecast / budget * 100) if budget > 0 else 0

        feasible = cost_forecast <= budget

        recommendation = ""
        if utilization > 100:
            reduction_factor = cost_forecast / budget
            new_epochs = int(epochs / reduction_factor)
            recommendation = f"Reduce to {new_epochs} epochs (${cost_forecast/reduction_factor:.2f}) to fit budget"
        elif utilization > 70:
            recommendation = f"At {utilization:.0f}% of budget. Consider off-peak scheduling (50% savings)"
        else:
            recommendation = f"Fits comfortably in budget ({utilization:.0f}% utilization)"

        return {
            "feasible": feasible,
            "budget": budget,
            "forecast_cost": round(cost_forecast, 4),
            "utilization_percent": round(utilization, 1),
            "recommendation": recommendation,
        }

    def generate_cost_report(
        self,
        runs: List[Dict],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        """
        Generate a cost report for training runs.

        Args:
            runs: List of run dicts with duration_seconds, params, etc.
            start_date: Filter runs after this date (YYYY-MM-DD)
            end_date: Filter runs before this date (YYYY-MM-DD)

        Returns:
            Formatted cost report
        """
        if not runs:
            return "No runs to report"

        total_cost = 0
        total_duration = 0
        report = f"Cost Report ({start_date or '(all)'} to {end_date or '(all)'})\n"
        report += "=" * 80 + "\n"
        report += f"{'Run ID':<20} {'Model':<10} {'Duration':<12} {'Cost':<10} {'Accuracy':<10}\n"
        report += "-" * 80 + "\n"

        for run in runs:
            run_id = run.get("run_id", "unknown")[:16]
            model = run.get("params", {}).get("model_type", "unknown")
            duration = run.get("duration_seconds", 0)
            cost_info = self.calculator.calculate_duration_cost(duration)
            cost = cost_info["total_cost"]
            accuracy = run.get("metrics", {}).get("val_accuracy", 0)

            total_cost += cost
            total_duration += duration

            report += f"{run_id:<20} {model:<10} {duration/60:>8.1f}min  ${cost:<9.2f} {accuracy:>8.1f}%\n"

        report += "-" * 80 + "\n"
        report += f"{'TOTAL':<20} {'':<10} {total_duration/3600:>7.1f}hrs  ${total_cost:<9.2f}\n"
        report += f"\nAverage cost per run: ${total_cost/len(runs):.2f}\n"
        report += f"Average duration: {total_duration/len(runs)/60:.1f} minutes\n"

        return report
