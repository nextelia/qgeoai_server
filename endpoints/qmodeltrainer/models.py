"""
Training job state management

This module manages the state of training jobs in memory.
Only one training job can run at a time.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """Represents a single training job"""
    
    job_id: str
    status: str  # "running", "completed", "failed", "stopped"
    config: Dict[str, Any]
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Progress
    current_epoch: int = 0
    total_epochs: int = 0
    progress_percent: float = 0.0
    
    # Results
    checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    report_path: Optional[str] = None
    
    # Metrics history
    metrics_history: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    
    # Error info
    error: Optional[str] = None
    traceback: Optional[str] = None
    
    # Training worker reference (for stopping)
    worker: Optional[Any] = None


class TrainingJobManager:
    """
    Manages training jobs in memory.
    
    Singleton pattern - only one training at a time.
    Thread-safe with locks.
    """
    
    def __init__(self):
        self._current_job: Optional[TrainingJob] = None
        self._lock = Lock()
    
    def create_job(self, config: Dict[str, Any]) -> str:
        """
        Create a new training job.
        
        Args:
            config: Training configuration
            
        Returns:
            job_id: UUID of the created job
            
        Raises:
            RuntimeError: If a job is already running
        """
        with self._lock:
            if self._current_job and self._current_job.status == "running":
                raise RuntimeError(
                    f"Training job {self._current_job.job_id} is already running. "
                    "Only one training at a time is supported."
                )
            
            job_id = str(uuid.uuid4())
            
            self._current_job = TrainingJob(
                job_id=job_id,
                status="running",
                config=config,
                total_epochs=config.get('epochs', 0)
            )
            
            logger.info(f"Created training job {job_id}")
            return job_id
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                return self._current_job
            return None
    
    def get_current_job(self) -> Optional[TrainingJob]:
        """Get the current job (if any)"""
        with self._lock:
            return self._current_job
    
    def update_progress(self, job_id: str, epoch: int, progress_percent: float):
        """Update job progress"""
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                self._current_job.current_epoch = epoch
                self._current_job.progress_percent = progress_percent
    
    def add_metrics(self, job_id: str, metrics: Dict[str, Any]):
        """Add epoch metrics to job history"""
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                self._current_job.metrics_history.append(metrics)
    
    def add_log(self, job_id: str, message: str, level: str = "info"):
        """Add log message to job"""
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                self._current_job.logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "level": level
                })
    
    def set_worker(self, job_id: str, worker: Any):
        """Set worker reference for job (to allow stopping)"""
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                self._current_job.worker = worker
    
    def complete_job(
        self,
        job_id: str,
        success: bool,
        checkpoint_path: Optional[str] = None,
        best_checkpoint_path: Optional[str] = None,
        report_path: Optional[str] = None,
        error: Optional[str] = None,
        traceback_str: Optional[str] = None
    ):
        """Mark job as completed or failed"""
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                self._current_job.status = "completed" if success else "failed"
                self._current_job.end_time = datetime.now()
                self._current_job.checkpoint_path = checkpoint_path
                self._current_job.best_checkpoint_path = best_checkpoint_path
                self._current_job.report_path = report_path
                self._current_job.error = error
                self._current_job.traceback = traceback_str
                self._current_job.worker = None  # Clear worker reference
                
                logger.info(f"Job {job_id} completed with status: {self._current_job.status}")
    
    def stop_job(self, job_id: str) -> bool:
        """
        Stop a running job.
        
        Args:
            job_id: Job to stop
            
        Returns:
            True if job was stopped, False if not found or not running
        """
        with self._lock:
            if not self._current_job or self._current_job.job_id != job_id:
                return False
            
            if self._current_job.status != "running":
                return False
            
            # Signal worker to stop
            if self._current_job.worker:
                self._current_job.worker.should_stop = True
            
            self._current_job.status = "stopped"
            self._current_job.end_time = datetime.now()
            
            logger.info(f"Stopped job {job_id}")
            return True
    
    def clear_job(self, job_id: str):
        """Clear a completed/failed/stopped job"""
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                if self._current_job.status == "running":
                    logger.warning(f"Cannot clear running job {job_id}")
                    return
                
                logger.info(f"Cleared job {job_id}")
                self._current_job = None


# Global singleton instance
job_manager = TrainingJobManager()