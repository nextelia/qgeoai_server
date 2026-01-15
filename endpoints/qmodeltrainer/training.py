"""
Training endpoints for QModelTrainer

Provides REST API endpoints for:
- Starting training jobs
- Monitoring progress
- Stopping training
- Streaming logs/metrics (SSE)
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse

from .schemas import TrainingStartResponse, TrainingStatus, TrainingResults
from .models import job_manager
from .worker import TrainingWorker
from core.training.utils.validators import validate_full_dataset

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qmodeltrainer", tags=["training"])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _run_training_worker(job_id: str, config: dict, dataset_info: dict):
    """
    Run training worker in background.
    
    Called by BackgroundTasks.
    """
    try:
        worker = TrainingWorker(job_id, config, dataset_info)
        worker.run()
    except Exception as e:
        logger.error(f"Worker error for job {job_id}: {e}")
        job_manager.complete_job(job_id, success=False, error=str(e))


async def _event_stream_generator(job_id: str):
    """
    Generator for Server-Sent Events (SSE) streaming.
    
    Streams logs and metrics in real-time to the client.
    """
    last_log_index = 0
    last_metrics_index = 0
    
    while True:
        job = job_manager.get_job(job_id)
        
        if not job:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Job not found"})
            }
            break
        
        # Send new logs
        if len(job.logs) > last_log_index:
            for log in job.logs[last_log_index:]:
                yield {
                    "event": "log",
                    "data": json.dumps(log)
                }
            last_log_index = len(job.logs)
        
        # Send new metrics
        if len(job.metrics_history) > last_metrics_index:
            for metrics in job.metrics_history[last_metrics_index:]:
                yield {
                    "event": "metrics",
                    "data": json.dumps(metrics)
                }
            last_metrics_index = len(job.metrics_history)
        
        # Send status update
        yield {
            "event": "status",
            "data": json.dumps({
                "status": job.status,
                "current_epoch": job.current_epoch,
                "total_epochs": job.total_epochs,
                "progress_percent": job.progress_percent
            })
        }
        
        # Stop streaming if job is finished
        if job.status in ["completed", "failed", "stopped"]:
            yield {
                "event": "complete",
                "data": json.dumps({
                    "status": job.status,
                    "message": "Training finished" if job.status == "completed" else job.error
                })
            }
            break
        
        # Wait before next update
        await asyncio.sleep(1)


# =============================================================================
# TRAINING ENDPOINTS
# =============================================================================

@router.post("/training/start", response_model=TrainingStartResponse)
async def start_training(request: dict, background_tasks: BackgroundTasks):
    """
    Start a new training job.
    
    Accepts RAW config from plugin (no Pydantic validation yet).
    
    Steps:
    1. Validate dataset
    2. Create job
    3. Start worker in background
    4. Return job_id
    """
    try:
        # Check if training already running
        current_job = job_manager.get_current_job()
        if current_job and current_job.status == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Training job {current_job.job_id} already running"
            )
        
        # Validate dataset path exists
        dataset_path = Path(request.get('dataset_path', ''))
        if not dataset_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset path does not exist: {dataset_path}"
            )
        
        # Validate dataset format
        valid, error, dataset_info = validate_full_dataset(str(dataset_path))
        if not valid:
            raise HTTPException(status_code=400, detail=f"Invalid dataset: {error}")
        
        # Create output directory
        output_dir = Path(request.get('output_dir', ''))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log received config
        logger.info("\n" + "="*80)
        logger.info("🔍 SERVER RECEIVED CONFIG (RAW)")
        logger.info("="*80)
        logger.info(json.dumps(request, indent=2))
        logger.info("="*80 + "\n")
        
        # Create job
        job_id = job_manager.create_job(request)
        
        logger.info(f"Starting training job {job_id}")
        
        # Start worker in background
        background_tasks.add_task(_run_training_worker, job_id, request, dataset_info)
        
        return TrainingStartResponse(
            job_id=job_id,
            status="started",
            message="Training job started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a running training job."""
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job.status != "running":
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not running (status: {job.status})"
            )
        
        # Log for debugging
        logger.info(f"Stop requested for job {job_id}")
        logger.info(f"Job has worker: {job.worker is not None}")
        if job.worker:
            logger.info(f"Worker has trainer: {job.worker.trainer is not None}")
        
        # Signal worker to stop
        if job.worker:
            job.worker.should_stop = True
            
            # Also signal trainer if available
            if job.worker.trainer:
                if hasattr(job.worker.trainer, 'stop'):
                    logger.info("Calling trainer.stop() (YOLO)")
                    job.worker.trainer.stop()
                elif hasattr(job.worker.trainer, 'should_stop'):
                    logger.info("Setting trainer.should_stop (PyTorch)")
                    job.worker.trainer.should_stop = True
        
        job_manager.add_log(job_id, "⏸️  Stop signal received, will stop after current epoch...")
        
        return {"status": "stopping", "message": "Stop signal sent to trainer"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error stopping training: {str(e)}")


@router.get("/training/{job_id}/status", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get current status of a training job."""
    # Not used yet
    pass


@router.get("/training/{job_id}/results", response_model=TrainingResults)
async def get_training_results(job_id: str):
    """Get final results of a completed training job."""
    # Not used yet
    pass


@router.get("/training/{job_id}/stream")
async def stream_training_progress(job_id: str):
    """
    Stream training progress via Server-Sent Events (SSE).
    
    This endpoint provides real-time updates on:
    - Log messages
    - Epoch metrics
    - Training status
    
    The stream closes automatically when training completes or fails.
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return EventSourceResponse(_event_stream_generator(job_id))


@router.get("/info")
async def get_training_info():
    """Get training capabilities and current status."""
    # Not used yet
    pass