"""Phase 6 Tier 2: Job Queue Manager

Manages training job queuing with priority scheduling, resource awareness,
and batch job support.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os


class JobPriority(str, Enum):
    """Job priority levels"""
    URGENT = "urgent"
    NORMAL = "normal"
    BACKGROUND = "background"


@dataclass
class QueuedJob:
    """A job in the queue"""
    job_id: str
    name: str
    model_type: str
    priority: JobPriority
    status: str  # queued, scheduled, running, completed, failed
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    epochs: int = 3
    batch_size: int = 64
    lr: float = 0.001
    gpu_enabled: bool = True
    estimated_duration_seconds: float = 300.0
    error_message: Optional[str] = None
    world_size: int = 1
    distributed: bool = False
    job_type: str = "batch/v1/Job"
    gpu_per_replica: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @property
    def queue_position(self) -> int:
        """Position when queued (would be set by queue manager)"""
        return 0

    @property
    def estimated_wait_seconds(self) -> float:
        """Estimated seconds until this job will run"""
        # This would be calculated by queue manager based on jobs ahead
        return 0


class JobQueue:
    """Manages a prioritized job queue"""

    def __init__(self, max_concurrent_jobs: int = 3):
        """
        Initialize job queue.

        Args:
            max_concurrent_jobs: Maximum jobs running simultaneously
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: Dict[str, QueuedJob] = {}
        self.submission_order: List[str] = []  # Track insertion order
        self.running_jobs: List[str] = []

    def submit_job(
        self,
        job_id: str,
        name: str,
        model_type: str,
        priority: JobPriority = JobPriority.NORMAL,
        epochs: int = 3,
        batch_size: int = 64,
        lr: float = 0.001,
        gpu_enabled: bool = True,
        world_size: int = 1,
        distributed: bool = False,
        gpu_per_replica: int = 1,
    ) -> Tuple[bool, str, int]:
        """
        Submit a job to the queue.

        Args:
            job_id: Unique job identifier
            name: Human-readable job name
            model_type: "mnist" or "llm"
            priority: Job priority level
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            gpu_enabled: Whether to use GPU
            world_size: Number of distributed workers (default: 1 for single-node)
            distributed: Whether this is a distributed job (PyTorchJob)
            gpu_per_replica: GPUs per worker in distributed training

        Returns:
            Tuple of (success, message, queue_position)
        """
        if job_id in self.jobs:
            return False, f"Job {job_id} already exists", -1

        # Estimate duration based on model and parameters
        if model_type == "mnist":
            time_per_epoch = 300 / (batch_size / 64)  # ~5 min base for 64 batch
        elif model_type == "llm":
            time_per_epoch = 600 / (batch_size / 64)  # ~10 min base for 64 batch
        else:
            time_per_epoch = 300

        estimated_duration = time_per_epoch * epochs

        # Adjust duration estimate for distributed training (assume 70% scaling efficiency)
        if distributed and world_size > 1:
            speedup_factor = world_size * 0.7  # 70% scaling efficiency
            estimated_duration = estimated_duration / speedup_factor

        job_type = "kubeflow.org/v1/PyTorchJob" if distributed else "batch/v1/Job"

        job = QueuedJob(
            job_id=job_id,
            name=name,
            model_type=model_type,
            priority=priority,
            status="queued",
            submitted_at=datetime.utcnow().isoformat(),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            gpu_enabled=gpu_enabled,
            estimated_duration_seconds=estimated_duration,
            world_size=world_size,
            distributed=distributed,
            job_type=job_type,
            gpu_per_replica=gpu_per_replica,
        )

        self.jobs[job_id] = job
        self.submission_order.append(job_id)

        queue_pos = self._calculate_queue_position(job_id)

        return True, f"Job {job_id} queued at position {queue_pos}", queue_pos

    def _calculate_queue_position(self, job_id: str) -> int:
        """Calculate position in queue for a job"""
        queued_jobs = [
            jid for jid in self.submission_order
            if jid in self.jobs and self.jobs[jid].status == "queued"
        ]
        return queued_jobs.index(job_id) + 1 if job_id in queued_jobs else -1

    def get_next_job(self) -> Optional[QueuedJob]:
        """
        Get the next job to run based on priority and FIFO within priority.

        Returns:
            Next job to schedule, or None if queue empty or max capacity reached
        """
        # Check capacity
        if len(self.running_jobs) >= self.max_concurrent_jobs:
            return None

        # Find queued jobs by priority (urgent > normal > background)
        queued_by_priority = {
            JobPriority.URGENT: [],
            JobPriority.NORMAL: [],
            JobPriority.BACKGROUND: []
        }

        for job_id in self.submission_order:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == "queued":
                    queued_by_priority[job.priority].append((self.submission_order.index(job_id), job_id))

        # Return highest priority queued job (oldest if same priority)
        for priority in [JobPriority.URGENT, JobPriority.NORMAL, JobPriority.BACKGROUND]:
            if queued_by_priority[priority]:
                # Sort by submission order (earliest first)
                queued_by_priority[priority].sort(key=lambda x: x[0])
                _, job_id = queued_by_priority[priority][0]
                return self.jobs[job_id]

        return None

    def schedule_job(self, job_id: str) -> Tuple[bool, str]:
        """
        Schedule a queued job to run.

        Args:
            job_id: Job to schedule

        Returns:
            Tuple of (success, message)
        """
        if job_id not in self.jobs:
            return False, f"Job {job_id} not found"

        job = self.jobs[job_id]
        if job.status != "queued":
            return False, f"Job {job_id} status is {job.status}, not queued"

        if len(self.running_jobs) >= self.max_concurrent_jobs:
            return False, f"Queue at capacity ({len(self.running_jobs)}/{self.max_concurrent_jobs})"

        job.status = "running"
        job.started_at = datetime.utcnow().isoformat()
        self.running_jobs.append(job_id)

        return True, f"Job {job_id} scheduled to run"

    def complete_job(self, job_id: str, success: bool = True, error: Optional[str] = None) -> Tuple[bool, str]:
        """
        Mark a job as completed.

        Args:
            job_id: Job to complete
            success: Whether job succeeded
            error: Error message if failed

        Returns:
            Tuple of (success, message)
        """
        if job_id not in self.jobs:
            return False, f"Job {job_id} not found"

        job = self.jobs[job_id]
        if job.status != "running":
            return False, f"Job {job_id} status is {job.status}, not running"

        job.status = "completed" if success else "failed"
        job.completed_at = datetime.utcnow().isoformat()
        job.error_message = error

        if job_id in self.running_jobs:
            self.running_jobs.remove(job_id)

        return True, f"Job {job_id} marked as {'completed' if success else 'failed'}"

    def get_queue_status(self) -> Dict:
        """
        Get current queue status.

        Returns:
            Dict with queue stats
        """
        queued_count = sum(1 for j in self.jobs.values() if j.status == "queued")
        running_count = len(self.running_jobs)
        completed_count = sum(1 for j in self.jobs.values() if j.status == "completed")
        failed_count = sum(1 for j in self.jobs.values() if j.status == "failed")

        # Calculate estimated wait times
        next_available = datetime.utcnow()
        if self.running_jobs:
            # Find job that will finish soonest
            finish_times = []
            for job_id in self.running_jobs:
                job = self.jobs[job_id]
                if job.started_at:
                    started = datetime.fromisoformat(job.started_at)
                    finish = started.timestamp() + job.estimated_duration_seconds
                    finish_times.append(finish)
            if finish_times:
                next_available = datetime.fromtimestamp(min(finish_times))

        total_wait = 0
        for job_id in self.submission_order:
            if job_id in self.jobs and self.jobs[job_id].status == "queued":
                total_wait += self.jobs[job_id].estimated_duration_seconds

        return {
            "queued_count": queued_count,
            "running_count": running_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "max_concurrent": self.max_concurrent_jobs,
            "capacity_used": running_count / self.max_concurrent_jobs * 100,
            "total_wait_time_hours": total_wait / 3600,
            "next_available_in_seconds": max(0, next_available.timestamp() - datetime.utcnow().timestamp()),
            "total_cost_pending": self._calculate_pending_cost(),
        }

    def _calculate_pending_cost(self) -> float:
        """Calculate estimated cost for all pending jobs"""
        total_cost = 0
        for job in self.jobs.values():
            if job.status == "queued":
                # Rough cost estimate: $0.25/GPU hour, $0.05/CPU core/hour
                hours = job.estimated_duration_seconds / 3600

                # For distributed jobs, multiply GPU cost by world_size
                if job.distributed and job.world_size > 1:
                    gpu_cost = hours * job.gpu_per_replica * job.world_size * 0.25 if job.gpu_enabled else 0
                    cpu_cost = hours * 4 * job.world_size * 0.05  # 4 cores per replica
                else:
                    gpu_cost = hours * 0.25 if job.gpu_enabled else 0
                    cpu_cost = hours * 4 * 0.05  # 4 cores @ $0.05/core/hour

                total_cost += gpu_cost + cpu_cost
        return round(total_cost, 4)

    def get_job_details(self, job_id: str) -> Optional[Dict]:
        """Get detailed information about a job"""
        if job_id not in self.jobs:
            return None
        return self.jobs[job_id].to_dict()

    def list_jobs(self, status: Optional[str] = None) -> List[Dict]:
        """
        List jobs, optionally filtered by status.

        Args:
            status: Filter by status (queued, running, completed, failed)

        Returns:
            List of job dicts
        """
        result = []
        for job_id in self.submission_order:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if status is None or job.status == status:
                    result.append(job.to_dict())
        return result

    def cancel_job(self, job_id: str) -> Tuple[bool, str]:
        """
        Cancel a queued job.

        Args:
            job_id: Job to cancel

        Returns:
            Tuple of (success, message)
        """
        if job_id not in self.jobs:
            return False, f"Job {job_id} not found"

        job = self.jobs[job_id]
        if job.status != "queued":
            return False, f"Cannot cancel job in {job.status} status"

        job.status = "cancelled"
        self.submission_order.remove(job_id)

        return True, f"Job {job_id} cancelled"

    def submit_batch(self, jobs_json: str) -> Tuple[bool, str, Dict]:
        """
        Submit multiple jobs as a batch.

        Args:
            jobs_json: JSON string with list of job configs

        Returns:
            Tuple of (success, message, stats)
        """
        try:
            jobs = json.loads(jobs_json)
            if not isinstance(jobs, list):
                return False, "Jobs must be a list", {}

            stats = {
                "submitted": 0,
                "failed": 0,
                "first_job": None,
                "total_estimated_cost": 0,
            }

            for i, job_config in enumerate(jobs):
                job_id = job_config.get("job_id") or f"batch-job-{int(time.time())}-{i}"
                name = job_config.get("name", job_id)
                model_type = job_config.get("model_type", "mnist")
                priority = JobPriority(job_config.get("priority", "normal"))

                success, msg, _ = self.submit_job(
                    job_id=job_id,
                    name=name,
                    model_type=model_type,
                    priority=priority,
                    epochs=job_config.get("epochs", 3),
                    batch_size=job_config.get("batch_size", 64),
                    lr=job_config.get("lr", 0.001),
                    gpu_enabled=job_config.get("gpu_enabled", True),
                    world_size=job_config.get("world_size", 1),
                    distributed=job_config.get("distributed", False),
                    gpu_per_replica=job_config.get("gpu_per_replica", 1),
                )

                if success:
                    stats["submitted"] += 1
                    if stats["first_job"] is None:
                        stats["first_job"] = job_id
                    # Add to total cost
                    job = self.jobs[job_id]
                    hours = job.estimated_duration_seconds / 3600
                    if job.distributed and job.world_size > 1:
                        cost = hours * job.gpu_per_replica * job.world_size * (0.25 if job.gpu_enabled else 0) + hours * 4 * job.world_size * 0.05
                    else:
                        cost = hours * (0.25 if job.gpu_enabled else 0) + hours * 4 * 0.05
                    stats["total_estimated_cost"] += cost
                else:
                    stats["failed"] += 1

            stats["total_estimated_cost"] = round(stats["total_estimated_cost"], 4)

            return True, f"Batch submitted: {stats['submitted']} jobs queued", stats

        except json.JSONDecodeError:
            return False, "Invalid JSON format", {}
        except Exception as e:
            return False, f"Error: {str(e)}", {}
