"""
Mask export endpoints for semantic segmentation
Handles PNG mask creation using PIL without OpenCV dependency
"""

import logging
import base64
from io import BytesIO
from typing import List, Dict
import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Check PIL availability
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - mask export will not work")


class PolygonFeature(BaseModel):
    """Single polygon feature with class ID"""
    class_id: int = Field(..., description="Class ID for this polygon")
    exterior_points: List[List[float]] = Field(..., description="Exterior ring points [[x, y], ...]")
    holes: List[List[List[float]]] = Field(default=[], description="Interior rings (holes) [[[x, y], ...], ...]")


class CreateMaskRequest(BaseModel):
    """Request to create a PNG mask from vector features"""
    width: int = Field(..., description="Mask width in pixels")
    height: int = Field(..., description="Mask height in pixels")
    features: List[PolygonFeature] = Field(..., description="List of polygon features to rasterize")
    x_offset: float = Field(..., description="Geographic X offset (left edge)")
    y_offset: float = Field(..., description="Geographic Y offset (top edge)")
    pixel_width: float = Field(..., description="Pixel width in geographic units")
    pixel_height: float = Field(..., description="Pixel height in geographic units")


@router.post("/create_png_mask")
async def create_png_mask(request: CreateMaskRequest):
    """Create a PNG mask from vector polygons.
    
    Returns:
        base64-encoded PNG mask
    """
    if not PIL_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="PIL not available on server - cannot create PNG masks"
        )
    
    try:
        # Create empty grayscale image (8-bit, value 0 = background)
        mask_image = Image.new('L', (request.width, request.height), 0)
        draw = ImageDraw.Draw(mask_image)
        
        # Sort features by class to ensure consistent rasterization order
        sorted_features = sorted(request.features, key=lambda f: f.class_id)
        
        for feature in sorted_features:
            class_id = feature.class_id
            
            # Convert exterior ring to pixel coordinates
            exterior_pixels = []
            for point in feature.exterior_points:
                px = (point[0] - request.x_offset) / request.pixel_width
                py = (request.y_offset - point[1]) / request.pixel_height
                
                # Clamp to bounds
                px = max(0, min(px, request.width - 1))
                py = max(0, min(py, request.height - 1))
                
                exterior_pixels.append((round(px), round(py)))
            
            # Draw exterior polygon
            if len(exterior_pixels) >= 3:
                draw.polygon(exterior_pixels, fill=class_id, outline=None)
            
            # Draw holes (interior rings) in black (background = 0)
            for hole in feature.holes:
                hole_pixels = []
                for point in hole:
                    px = (point[0] - request.x_offset) / request.pixel_width
                    py = (request.y_offset - point[1]) / request.pixel_height
                    
                    px = max(0, min(px, request.width - 1))
                    py = max(0, min(py, request.height - 1))
                    
                    hole_pixels.append((round(px), round(py)))
                
                if len(hole_pixels) >= 3:
                    draw.polygon(hole_pixels, fill=0, outline=None)
        
        # Convert to numpy array
        mask_array = np.array(mask_image, dtype=np.uint8)
        
        # Save to BytesIO buffer as PNG
        buffer = BytesIO()
        mask_image.save(buffer, format='PNG', optimize=False, compress_level=0)
        buffer.seek(0)
        
        # Encode to base64
        mask_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Calculate statistics
        unique_values = np.unique(mask_array)
        has_annotations = len(unique_values) > 1 or (len(unique_values) == 1 and unique_values[0] != 0)
        
        return {
            "success": True,
            "mask_base64": mask_base64,
            "has_annotations": has_annotations,
            "unique_classes": unique_values.tolist(),
            "shape": [request.height, request.width]
        }
    
    except Exception as e:
        logger.error(f"Error creating PNG mask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mask creation failed: {str(e)}")


@router.get("/mask_export/status")
async def mask_export_status():
    """Check if mask export functionality is available."""
    return {
        "available": PIL_AVAILABLE,
        "capabilities": {
            "png_export": PIL_AVAILABLE,
            "formats": ["PNG"] if PIL_AVAILABLE else []
        }
    }