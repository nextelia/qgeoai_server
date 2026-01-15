"""
Annotation endpoints for SAM2 interactive segmentation
Handles model loading, prediction, and session management for incremental refinement
"""

import logging
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Global state
# ============================================================================

_sam2_state = {
    "model": None,
    "predictor": None,
    "device": None,
    "model_type": None,
}

_session_state = {
    "active": False,
    "image_path": None,
    "crop_bounds": None,  # (x, y, width, height)
    "points_coords": [],  # List of [x, y] in crop coordinates
    "points_labels": [],  # List of 1 (positive) or 0 (negative)
    "mask_logits": None,  # Logits from last prediction for refinement
    "simplify_tolerance": 2.0,
    "rgb_bands": None,
}


# ============================================================================
# Pydantic models
# ============================================================================

class SAM2LoadRequest(BaseModel):
    model_type: str = Field(default="sam2.1_hiera_tiny")
    checkpoint_path: Optional[str] = Field(default=None)
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        valid = ['sam2.1_hiera_tiny', 'sam2.1_hiera_small', 
                 'sam2.1_hiera_base_plus', 'sam2.1_hiera_large']
        if v not in valid:
            raise ValueError(f"model_type must be one of {valid}")
        return v


class SessionStartRequest(BaseModel):
    image_path: str = Field(..., description="Path to raster image")
    crop_x: int = Field(..., description="Crop left pixel coordinate")
    crop_y: int = Field(..., description="Crop top pixel coordinate")
    crop_width: int = Field(..., description="Crop width in pixels")
    crop_height: int = Field(..., description="Crop height in pixels")
    simplify_tolerance: float = Field(default=2.0)
    rgb_bands: Optional[List[int]] = Field(default=None)
    
    @field_validator('image_path')
    @classmethod
    def validate_path(cls, v):
        if not Path(v).is_absolute():
            raise ValueError("Path must be absolute")
        return v


class AddPointRequest(BaseModel):
    point_x: int = Field(..., description="Point X in crop coordinates")
    point_y: int = Field(..., description="Point Y in crop coordinates")
    is_positive: bool = Field(default=True, description="True for positive, False for negative")


class PreviewPointRequest(BaseModel):
    point_x: int = Field(..., description="Point X in crop coordinates")
    point_y: int = Field(..., description="Point Y in crop coordinates")
    is_positive: bool = Field(default=True)


class UpdateSimplificationRequest(BaseModel):
    tolerance: float = Field(..., description="New simplification tolerance")


class PredictPointRequest(BaseModel):
    """Single point prediction without session"""
    image_path: str
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    point_x: int
    point_y: int
    point_label: int = Field(default=1)
    simplify_tolerance: float = Field(default=2.0)
    rgb_bands: Optional[List[int]] = Field(default=None)


class PredictBoxRequest(BaseModel):
    """Bounding box prediction"""
    image_path: str
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    box: List[int] = Field(..., description="[x_min, y_min, x_max, y_max] in crop coords")
    simplify_tolerance: float = Field(default=2.0)
    rgb_bands: Optional[List[int]] = Field(default=None)


class PolygonResponse(BaseModel):
    status: str
    polygon: Optional[List[List[float]]] = None
    point_count: int = 0
    score: Optional[float] = None
    message: Optional[str] = None


class StatusResponse(BaseModel):
    model_loaded: bool
    model_type: Optional[str] = None
    device: Optional[str] = None
    cuda_available: bool = False
    session_active: bool = False
    session_point_count: int = 0


# ============================================================================
# Helper functions
# ============================================================================

def _get_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda', torch.cuda.get_device_name(0)
    return 'cpu', 'CPU'


def _read_crop(image_path: str, x: int, y: int, width: int, height: int,
               rgb_bands: Optional[List[int]] = None):
    """Read crop from raster using rasterio"""
    import numpy as np
    import rasterio
    import rasterio.windows
    
    with rasterio.open(image_path) as src:
        window = rasterio.windows.Window(x, y, width, height)
        band_count = src.count
        
        if rgb_bands is not None:
            bands_to_read = rgb_bands
        elif band_count >= 3:
            bands_to_read = [1, 2, 3]
        else:
            bands_to_read = [1, 1, 1]
        
        arr = src.read(bands_to_read, window=window)
        arr = np.moveaxis(arr, 0, -1)  # (C,H,W) -> (H,W,C)
        
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            for c in range(arr.shape[2]):
                channel = arr[:, :, c]
                if channel.max() > channel.min():
                    arr[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min()) * 255
            arr = arr.astype(np.uint8)
        
        return arr


def _mask_to_polygon(mask, simplify_tolerance: float = 2.0):
    """Convert binary mask to polygon coordinates"""
    import cv2
    import numpy as np
    
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = (mask * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    
    if simplify_tolerance > 0:
        simplified = cv2.approxPolyDP(largest, simplify_tolerance, True)
    else:
        simplified = largest
    
    return [[float(pt[0][0]), float(pt[0][1])] for pt in simplified]


def _predict_with_session():
    """Run prediction with current session state, returns (polygon, score)"""
    import torch
    import numpy as np
    
    if not _session_state["active"]:
        return None, None
    
    if not _session_state["points_coords"]:
        return None, None
    
    predictor = _sam2_state["predictor"]
    
    with torch.no_grad():
        if _session_state["mask_logits"] is not None:
            masks, scores, logits = predictor.predict(
                point_coords=np.array(_session_state["points_coords"]),
                point_labels=np.array(_session_state["points_labels"]),
                mask_input=_session_state["mask_logits"],
                multimask_output=False
            )
        else:
            masks, scores, logits = predictor.predict(
                point_coords=np.array(_session_state["points_coords"]),
                point_labels=np.array(_session_state["points_labels"]),
                multimask_output=False
            )
    
    _session_state["mask_logits"] = logits
    
    if masks is None or len(masks) == 0:
        return None, None
    
    polygon = _mask_to_polygon(masks[0], _session_state["simplify_tolerance"])
    score = float(scores[0]) if scores is not None else None
    
    return polygon, score


# ============================================================================
# Model endpoints
# ============================================================================

@router.get("/sam2/status", response_model=StatusResponse)
def sam2_status():
    import torch
    return StatusResponse(
        model_loaded=_sam2_state["predictor"] is not None,
        model_type=_sam2_state["model_type"],
        device=_sam2_state["device"],
        cuda_available=torch.cuda.is_available(),
        session_active=_session_state["active"],
        session_point_count=len(_session_state["points_coords"])
    )


@router.post("/sam2/load")
def sam2_load(request: SAM2LoadRequest):
    import torch
    import sys
    
    if (_sam2_state["predictor"] is not None and 
        _sam2_state["model_type"] == request.model_type):
        return {"status": "already_loaded", "model_type": request.model_type, 
                "device": _sam2_state["device"]}
    
    if _sam2_state["model"] is not None:
        _sam2_state["model"] = None
        _sam2_state["predictor"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
        logger.info(f"Loading SAM2 model: {request.model_type}")
        device, device_name = _get_device()
        logger.info(f"Device: {device_name}")
        
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        server_dir = Path(__file__).parent.parent
        sam2_path = server_dir / "sam2"
        
        if not sam2_path.exists():
            raise HTTPException(status_code=503, detail=f"SAM2 not found at {sam2_path}")
        
        if str(server_dir) not in sys.path:
            sys.path.insert(0, str(server_dir))
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        config_map = {
            'sam2.1_hiera_tiny': 'sam2.1/sam2.1_hiera_t',
            'sam2.1_hiera_small': 'sam2.1/sam2.1_hiera_s',
            'sam2.1_hiera_base_plus': 'sam2.1/sam2.1_hiera_b+',
            'sam2.1_hiera_large': 'sam2.1/sam2.1_hiera_l'
        }
        config_name = config_map.get(request.model_type, 'sam2.1/sam2.1_hiera_t')
        
        checkpoint_dir = Path.home() / '.qgeoai' / 'sam2_checkpoints'
        checkpoint_map = {
            'sam2.1_hiera_tiny': 'sam2.1_hiera_tiny.pt',
            'sam2.1_hiera_small': 'sam2.1_hiera_small.pt',
            'sam2.1_hiera_base_plus': 'sam2.1_hiera_base_plus.pt',
            'sam2.1_hiera_large': 'sam2.1_hiera_large.pt'
        }
        checkpoint_path = str(checkpoint_dir / checkpoint_map.get(request.model_type))
        
        if not Path(checkpoint_path).exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
        
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        config_dir = str(sam2_path / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            model = build_sam2(config_name, checkpoint_path, device=device)
        
        model.eval()
        predictor = SAM2ImagePredictor(model)
        
        _sam2_state["model"] = model
        _sam2_state["predictor"] = predictor
        _sam2_state["device"] = device_name
        _sam2_state["model_type"] = request.model_type
        
        logger.info(f"SAM2 loaded on {device_name}")
        return {"status": "success", "model_type": request.model_type, "device": device_name}
        
    except Exception as e:
        logger.error(f"Failed to load SAM2: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load SAM2: {str(e)}")


@router.post("/sam2/unload")
def sam2_unload():
    import torch
    
    # End any active session first
    _session_state["active"] = False
    _session_state["points_coords"] = []
    _session_state["points_labels"] = []
    _session_state["mask_logits"] = None
    
    if _sam2_state["model"] is None:
        return {"status": "not_loaded"}
    
    _sam2_state["model"] = None
    _sam2_state["predictor"] = None
    _sam2_state["model_type"] = None
    _sam2_state["device"] = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"status": "success"}


# ============================================================================
# Session endpoints (incremental refinement)
# ============================================================================

@router.post("/sam2/session/start")
def session_start(request: SessionStartRequest):
    """Start an interactive session - encodes the crop once"""
    import torch
    
    if _sam2_state["predictor"] is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    if not Path(request.image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    
    try:
        # Read crop
        arr = _read_crop(
            request.image_path,
            request.crop_x, request.crop_y,
            request.crop_width, request.crop_height,
            request.rgb_bands
        )
        
        logger.info(f"Starting session - encoding crop: {arr.shape}")
        
        # Encode image
        with torch.no_grad():
            _sam2_state["predictor"].set_image(arr)
        
        # Initialize session state
        _session_state["active"] = True
        _session_state["image_path"] = request.image_path
        _session_state["crop_bounds"] = (request.crop_x, request.crop_y, 
                                          request.crop_width, request.crop_height)
        _session_state["points_coords"] = []
        _session_state["points_labels"] = []
        _session_state["mask_logits"] = None
        _session_state["simplify_tolerance"] = request.simplify_tolerance
        _session_state["rgb_bands"] = request.rgb_bands
        
        return {"status": "success", "message": "Session started, crop encoded"}
        
    except Exception as e:
        logger.error(f"Session start failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sam2/session/add_point", response_model=PolygonResponse)
def session_add_point(request: AddPointRequest):
    """Add a point to session and get refined mask"""
    if not _session_state["active"]:
        raise HTTPException(status_code=400, detail="No active session")
    
    try:
        # Add point
        _session_state["points_coords"].append([request.point_x, request.point_y])
        _session_state["points_labels"].append(1 if request.is_positive else 0)
        
        # Predict
        polygon, score = _predict_with_session()
        
        return PolygonResponse(
            status="success" if polygon else "no_mask",
            polygon=polygon,
            point_count=len(_session_state["points_coords"]),
            score=score
        )
        
    except Exception as e:
        logger.error(f"Add point failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sam2/session/preview", response_model=PolygonResponse)
def session_preview(request: PreviewPointRequest):
    """Preview mask with additional point WITHOUT committing"""
    import torch
    import numpy as np
    
    if not _session_state["active"]:
        raise HTTPException(status_code=400, detail="No active session")
    
    try:
        # Create temporary point lists
        preview_coords = _session_state["points_coords"] + [[request.point_x, request.point_y]]
        preview_labels = _session_state["points_labels"] + [1 if request.is_positive else 0]
        
        predictor = _sam2_state["predictor"]
        
        with torch.no_grad():
            if _session_state["mask_logits"] is not None:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(preview_coords),
                    point_labels=np.array(preview_labels),
                    mask_input=_session_state["mask_logits"],
                    multimask_output=False
                )
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(preview_coords),
                    point_labels=np.array(preview_labels),
                    multimask_output=False
                )
        
        if masks is None or len(masks) == 0:
            return PolygonResponse(status="no_mask", point_count=len(_session_state["points_coords"]))
        
        polygon = _mask_to_polygon(masks[0], _session_state["simplify_tolerance"])
        
        return PolygonResponse(
            status="success" if polygon else "no_contour",
            polygon=polygon,
            point_count=len(_session_state["points_coords"]),
            score=float(scores[0]) if scores is not None else None
        )
        
    except Exception as e:
        logger.error(f"Preview failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sam2/session/undo", response_model=PolygonResponse)
def session_undo():
    """Remove last point and recompute mask"""
    import torch
    import numpy as np
    
    if not _session_state["active"]:
        raise HTTPException(status_code=400, detail="No active session")
    
    if not _session_state["points_coords"]:
        return PolygonResponse(status="no_points", point_count=0)
    
    try:
        # Remove last point
        _session_state["points_coords"].pop()
        _session_state["points_labels"].pop()
        
        # If no points left, clear logits
        if not _session_state["points_coords"]:
            _session_state["mask_logits"] = None
            return PolygonResponse(status="cleared", point_count=0)
        
        # Recompute from scratch (no logits for clean undo)
        predictor = _sam2_state["predictor"]
        
        with torch.no_grad():
            masks, scores, logits = predictor.predict(
                point_coords=np.array(_session_state["points_coords"]),
                point_labels=np.array(_session_state["points_labels"]),
                multimask_output=False
            )
        
        _session_state["mask_logits"] = logits
        
        if masks is None or len(masks) == 0:
            return PolygonResponse(status="no_mask", point_count=len(_session_state["points_coords"]))
        
        polygon = _mask_to_polygon(masks[0], _session_state["simplify_tolerance"])
        
        return PolygonResponse(
            status="success" if polygon else "no_contour",
            polygon=polygon,
            point_count=len(_session_state["points_coords"]),
            score=float(scores[0]) if scores is not None else None
        )
        
    except Exception as e:
        logger.error(f"Undo failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sam2/session/update_simplification", response_model=PolygonResponse)
def session_update_simplification(request: UpdateSimplificationRequest):
    """Update simplification tolerance and return recomputed polygon"""
    if not _session_state["active"]:
        raise HTTPException(status_code=400, detail="No active session")
    
    _session_state["simplify_tolerance"] = request.tolerance
    
    if not _session_state["points_coords"]:
        return PolygonResponse(status="no_points", point_count=0)
    
    try:
        polygon, score = _predict_with_session()
        
        return PolygonResponse(
            status="success" if polygon else "no_mask",
            polygon=polygon,
            point_count=len(_session_state["points_coords"]),
            score=score
        )
        
    except Exception as e:
        logger.error(f"Update simplification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sam2/session/end")
def session_end():
    """End the current session"""
    _session_state["active"] = False
    _session_state["image_path"] = None
    _session_state["crop_bounds"] = None
    _session_state["points_coords"] = []
    _session_state["points_labels"] = []
    _session_state["mask_logits"] = None
    
    return {"status": "success", "message": "Session ended"}


# ============================================================================
# Direct prediction endpoints (no session)
# ============================================================================

@router.post("/sam2/predict_point", response_model=PolygonResponse)
def predict_point(request: PredictPointRequest):
    """Single point prediction without session state"""
    import torch
    import numpy as np
    
    if _sam2_state["predictor"] is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    if not Path(request.image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found")
    
    try:
        arr = _read_crop(
            request.image_path,
            request.crop_x, request.crop_y,
            request.crop_width, request.crop_height,
            request.rgb_bands
        )
        
        predictor = _sam2_state["predictor"]
        
        with torch.no_grad():
            predictor.set_image(arr)
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[request.point_x, request.point_y]]),
                point_labels=np.array([request.point_label]),
                multimask_output=False
            )
        
        if masks is None or len(masks) == 0:
            return PolygonResponse(status="no_mask")
        
        polygon = _mask_to_polygon(masks[0], request.simplify_tolerance)
        
        return PolygonResponse(
            status="success" if polygon else "no_contour",
            polygon=polygon,
            score=float(scores[0]) if scores is not None else None
        )
        
    except Exception as e:
        logger.error(f"Point prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sam2/predict_box", response_model=PolygonResponse)
def predict_box(request: PredictBoxRequest):
    """Bounding box prediction"""
    import torch
    import numpy as np
    
    if _sam2_state["predictor"] is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    if not Path(request.image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found")
    
    try:
        arr = _read_crop(
            request.image_path,
            request.crop_x, request.crop_y,
            request.crop_width, request.crop_height,
            request.rgb_bands
        )
        
        predictor = _sam2_state["predictor"]
        
        with torch.no_grad():
            predictor.set_image(arr)
            masks, scores, _ = predictor.predict(
                box=np.array(request.box),
                multimask_output=False
            )
        
        if masks is None or len(masks) == 0:
            return PolygonResponse(status="no_mask")
        
        polygon = _mask_to_polygon(masks[0], request.simplify_tolerance)
        
        return PolygonResponse(
            status="success" if polygon else "no_contour",
            polygon=polygon,
            score=float(scores[0]) if scores is not None else None
        )
        
    except Exception as e:
        logger.error(f"Box prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
def annotation_info():
    return {
        "status": "implemented",
        "endpoints": {
            "model": ["/sam2/status", "/sam2/load", "/sam2/unload"],
            "session": ["/sam2/session/start", "/sam2/session/add_point", 
                       "/sam2/session/preview", "/sam2/session/undo",
                       "/sam2/session/update_simplification", "/sam2/session/end"],
            "direct": ["/sam2/predict_point", "/sam2/predict_box"]
        }
    }