"""
Pydantic schemas for QModelTrainer endpoints

These schemas define the contract between the QGIS plugin and the server.
All ML-related configuration is sent via these schemas.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, validator
from pathlib import Path


# =============================================================================
# TRAINING REQUEST
# =============================================================================

class AugmentationConfig(BaseModel):
    """Data augmentation configuration"""
    # Geometric
    hflip: float = 0.0
    vflip: float = 0.0
    rotate90: float = 0.0
    # Radiometric
    brightness: float = 0.0
    contrast: float = 0.0
    hue: float = 0.0          
    saturation: float = 0.0   
    blur: float = 0.0         
    noise: float = 0.0        


class TrainingRequest(BaseModel):
    """Request to start a training job"""
    
    # Dataset
    dataset_path: str = Field(..., description="Absolute path to dataset folder on server")
    task: Literal["semantic", "instance"] = Field(..., description="Training task type")
    
    # Model architecture
    architecture: str = Field(..., description="Model architecture (e.g., 'unet', 'deeplabv3+')")
    backbone: str = Field(..., description="Encoder backbone (e.g., 'resnet50')")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    encoder_weights: Optional[str] = Field(None, description="Encoder weights (e.g., 'imagenet', 'ssl')")  # ← AJOUTER
    
    # Training parameters
    epochs: int = Field(..., ge=1, description="Number of training epochs")
    batch_size: int = Field(..., ge=1, description="Batch size")
    image_size: int = Field(..., ge=32, description="Input image size")
    val_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Validation split ratio")
    
    # Optimizer
    optimizer: Literal["adam", "adamw", "sgd"] = Field(default="adam")  # ← LOWERCASE DÉJÀ
    learning_rate: float = Field(..., gt=0.0, description="Initial learning rate")
    weight_decay: float = Field(default=0.0, ge=0.0, description="Weight decay (L2 regularization)")
    
    # Learning rate scheduler
    scheduler: Optional[Literal["plateau", "step", "onecycle", "cosine"]] = None  # ← LOWERCASE
    
    # Device
    device: Literal["cuda", "cpu"] = Field(default="cuda")
    cuda_device_id: int = Field(default=0, ge=0, description="CUDA device ID if using GPU")
    
    # Data augmentation
    augmentation_config: Optional[AugmentationConfig] = None
    
    # Advanced options
    resume_training: bool = Field(default=False, description="Resume from checkpoint")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint file")
    use_lr_finder: bool = Field(default=False, description="Run LR finder before training")
    generate_report: bool = Field(default=True, description="Generate HTML report after training")
    
    # Output
    output_dir: str = Field(..., description="Output directory for checkpoints and reports")
    experiment_name: Optional[str] = Field(None, description="Experiment name for outputs")
    
    @validator('dataset_path', 'output_dir')
    def validate_path_exists(cls, v):
        """Validate that path exists on server"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist on server: {v}")
        return str(path.absolute())
    
    @validator('checkpoint_path')
    def validate_checkpoint(cls, v, values):
        """Validate checkpoint path if resume_training is enabled"""
        if values.get('resume_training') and v:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Checkpoint file not found: {v}")
        return v


class YOLOTrainingRequest(BaseModel):
    """Request to start a YOLO training job"""
    
    # Dataset
    dataset_path: str = Field(..., description="Absolute path to YOLO dataset folder")
    task: Literal["detect", "segment", "obb"] = Field(..., description="YOLO task type")
    
    # Model
    model_size: Literal["n", "s", "m", "l", "x"] = Field(default="s", description="YOLO model size")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    
    # Training parameters
    epochs: int = Field(..., ge=1, description="Number of training epochs")
    batch_size: int = Field(..., ge=1, description="Batch size")
    image_size: int = Field(..., ge=32, description="Input image size")
    
    # Optimizer
    optimizer: Literal["auto", "SGD", "Adam", "AdamW"] = Field(default="auto")
    learning_rate: Optional[float] = Field(None, gt=0.0, description="Learning rate (auto if None)")
    
    # Device
    device: Literal["cuda", "cpu"] = Field(default="cuda")
    cuda_device_id: int = Field(default=0, ge=0)
    
    # Advanced
    resume_training: bool = Field(default=False)
    checkpoint_path: Optional[str] = None
    
    # Output
    output_dir: str = Field(..., description="Output directory")
    experiment_name: Optional[str] = Field(None)


# =============================================================================
# TRAINING RESPONSE
# =============================================================================

class TrainingStartResponse(BaseModel):
    """Response when training starts"""
    job_id: str = Field(..., description="Unique job identifier (UUID)")
    status: Literal["started"] = "started"
    message: str = "Training job started successfully"


# =============================================================================
# STATUS & METRICS
# =============================================================================

class TrainingStatus(BaseModel):
    """Current status of a training job"""
    job_id: str
    status: Literal["running", "completed", "failed", "stopped"]
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    progress_percent: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class EpochMetrics(BaseModel):
    """Metrics for a single epoch"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_metrics: Dict[str, float] = Field(default_factory=dict)
    val_metrics: Dict[str, float] = Field(default_factory=dict)
    learning_rate: float
    time_elapsed: float


class LogMessage(BaseModel):
    """Log message from training"""
    timestamp: str
    message: str
    level: Literal["info", "warning", "error"] = "info"


# =============================================================================
# TRAINING RESULTS
# =============================================================================

class TrainingResults(BaseModel):
    """Final results after training completes"""
    job_id: str
    status: Literal["completed", "failed", "stopped"]
    message: str
    
    # Paths to outputs (on server)
    checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    report_path: Optional[str] = None
    
    # Final metrics
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    final_metrics: Optional[Dict[str, float]] = None
    
    # Training info
    total_epochs_completed: Optional[int] = None
    total_time: Optional[str] = None
    
    # Error info (if failed)
    error: Optional[str] = None
    traceback: Optional[str] = None


# =============================================================================
# CONTROL REQUESTS
# =============================================================================

class StopTrainingRequest(BaseModel):
    """Request to stop a training job"""
    job_id: str


class StopTrainingResponse(BaseModel):
    """Response when stopping training"""
    job_id: str
    status: Literal["stopping", "not_found"]
    message: str


# =============================================================================
# DATASET VALIDATION (for pre-flight checks)
# =============================================================================

class DatasetValidationRequest(BaseModel):
    """Request to validate dataset before training"""
    dataset_path: str
    task: Literal["semantic", "instance", "yolo"]
    
    @validator('dataset_path')
    def validate_path(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        return str(path.absolute())


class DatasetValidationResponse(BaseModel):
    """Response with dataset validation results"""
    valid: bool
    message: str
    
    # Dataset info (if valid)
    num_classes: Optional[int] = None
    num_bands: Optional[int] = None
    class_names: Optional[List[str]] = None
    num_train_images: Optional[int] = None
    num_val_images: Optional[int] = None
    image_sizes: Optional[List[tuple]] = None
    
    # Errors (if invalid)
    errors: Optional[List[str]] = None