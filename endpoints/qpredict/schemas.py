"""
Pydantic schemas for QPredict API

Request and response models for type validation and documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


# ============================================================================
# Model Loading
# ============================================================================

class LoadModelRequest(BaseModel):
    """Request to load a model from .qmtp file."""
    device: Literal['cpu', 'cuda'] = Field(
        default='cpu',
        description="Device to load model on"
    )
    cuda_device_id: int = Field(
        default=0,
        ge=0,
        le=7,
        description="CUDA device ID if using GPU"
    )


class LoadModelResponse(BaseModel):
    """Response after loading a model."""
    model_id: str = Field(description="Unique identifier for the loaded model")
    task: str = Field(description="Model task type")
    architecture: str = Field(description="Model architecture name")
    num_classes: int = Field(description="Number of classes")
    classes: List[Dict[str, Any]] = Field(description="Class definitions")
    device: str = Field(description="Device model is loaded on")
    input_channels: int = Field(description="Number of input channels")
    divisibility: int = Field(description="Divisibility constraint for tile dimensions")
    loaded_at: datetime = Field(description="Timestamp when model was loaded")


# ============================================================================
# Prediction
# ============================================================================

class TileMetadata(BaseModel):
    """Metadata for a single tile."""
    tile_width: int
    tile_height: int
    pad_left: int
    pad_top: int
    out_x: int
    out_y: int
    # YOLO-specific (optional)
    x_min: Optional[int] = None
    y_min: Optional[int] = None
    x_max: Optional[int] = None
    y_max: Optional[int] = None


class PredictRequest(BaseModel):
    """Request for prediction on tiles."""
    model_id: str = Field(description="Model UUID from load-model")
    tiles: List[str] = Field(description="Base64-encoded numpy arrays (tiles)")
    tile_metadata: List[TileMetadata] = Field(description="Metadata for each tile")
    output_shape: tuple[int, int] = Field(description="(height, width) of output")
    batch_size: int = Field(default=4, ge=1, description="Batch size for inference")
    
    # Task-specific parameters
    output_format: Optional[str] = Field(
        default='class_raster',
        description="'class_raster', 'reliability_map', or 'vector'"
    )
    confidence_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold (instance/YOLO)"
    )
    iou_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS (instance/YOLO)"
    )
    nms_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="NMS threshold (instance)"
    )
    overlap: Optional[int] = Field(
        default=0,
        ge=0,
        description="Overlap size for centroid filtering"
    )
    simplify_tolerance: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        description="Simplification tolerance (YOLO segment)"
    )
    reliability_type: Optional[str] = Field(
        default=None,
        description="'Confidence' or 'Uncertainty' (semantic)"
    )
    apply_colors: Optional[bool] = Field(
        default=True,
        description="Include color information (semantic)"
    )


class PredictResponse(BaseModel):
    """Response from prediction."""
    prediction: str = Field(description="Base64-encoded numpy array or JSON data")
    output_format: str = Field(description="Type of output")
    classes: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Class definitions"
    )
    num_instances: Optional[int] = Field(
        default=None,
        description="Number of instances (instance segmentation)"
    )
    num_detections: Optional[int] = Field(
        default=None,
        description="Number of detections (YOLO)"
    )
    yolo_task: Optional[str] = Field(
        default=None,
        description="YOLO task type"
    )
    reliability_type: Optional[str] = Field(
        default=None,
        description="Type of reliability map"
    )


# ============================================================================
# Model Management
# ============================================================================

class UnloadModelRequest(BaseModel):
    """Request to unload a model."""
    model_id: str = Field(description="Model UUID to unload")


class UnloadModelResponse(BaseModel):
    """Response after unloading a model."""
    success: bool
    message: str


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    model_id: str
    task: str
    architecture: str
    device: str
    loaded_at: datetime


class ListModelsResponse(BaseModel):
    """Response listing all loaded models."""
    models: List[ModelInfo] = Field(description="List of currently loaded models")
    total: int = Field(description="Total number of loaded models")