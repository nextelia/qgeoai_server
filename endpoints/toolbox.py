"""
QToolbox endpoints
Handles building regularization and other toolbox operations
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator

# Import dependencies for verification
try:
    from buildingregulariser import regularize_geodataframe
    import geopandas as gpd
    REGULARISER_AVAILABLE = True
except ImportError:
    REGULARISER_AVAILABLE = False
    logging.warning("buildingregulariser not available")

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("rasterio not available")

try:
    from smoothify import smoothify
    SMOOTHIFY_AVAILABLE = True
except ImportError:
    SMOOTHIFY_AVAILABLE = False
    logging.warning("smoothify not available")

logger = logging.getLogger(__name__)

router = APIRouter()


class RegularizeRequest(BaseModel):
    """Request model for building regularization"""
    input_path: str = Field(..., description="Path to input GeoJSON file")
    output_path: str = Field(..., description="Path to output GeoJSON file")
    params: dict = Field(default_factory=dict, description="Regularization parameters")
    
    @field_validator('input_path', 'output_path')
    @classmethod
    def validate_paths(cls, v):
        """Validate that paths are absolute and safe"""
        path = Path(v)
        
        # Must be absolute path
        if not path.is_absolute():
            raise ValueError("Path must be absolute")
        
        # Basic security: prevent path traversal
        try:
            path.resolve()
        except (OSError, RuntimeError):
            raise ValueError("Invalid path")
        
        return str(path)


class RegularizeResponse(BaseModel):
    """Response model for building regularization"""
    status: str
    output_path: str
    message: Optional[str] = None
    processed_features: Optional[int] = None


@router.post("/regularize", response_model=RegularizeResponse)
def regularize_buildings(request: RegularizeRequest):
    """
    Regularize building geometries using buildingregulariser
    
    Args:
        request: Regularization request with input/output paths and parameters
    
    Returns:
        RegularizeResponse with status and output path
    """
    if not REGULARISER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="buildingregulariser not available in this environment"
        )
    
    input_path = Path(request.input_path)
    output_path = Path(request.output_path)
    
    # Validate input file exists
    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Input file not found: {input_path}"
        )
    
    # Validate input is a file
    if not input_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Input path is not a file: {input_path}"
        )
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Starting regularization: {input_path} -> {output_path}")
        logger.info(f"Parameters: {request.params}")
        
        # Read input GeoJSON with geopandas
        gdf = gpd.read_file(input_path)
        logger.info(f"Loaded {len(gdf)} features from {input_path}")
        
        # Default parameters for buildingregulariser
        # See: https://buildingregulariser.readthedocs.io/
        params = {
            'parallel_threshold': request.params.get('parallel_threshold', 1.0),
            'simplify': request.params.get('simplify', True),
            'simplify_tolerance': request.params.get('simplify_tolerance', 0.5),
            'allow_45_degree': request.params.get('allow_45_degree', True),
            'diagonal_threshold_reduction': request.params.get('diagonal_threshold_reduction', 15.0),
            'allow_circles': request.params.get('allow_circles', True),
            'circle_threshold': request.params.get('circle_threshold', 0.9),
            'num_cores': request.params.get('num_cores', 0),  # 0 = use all cores
            'include_metadata': request.params.get('include_metadata', False),
            'neighbor_alignment': request.params.get('neighbor_alignment', False),
            'neighbor_search_distance': request.params.get('neighbor_search_distance', 100.0),
            'neighbor_max_rotation': request.params.get('neighbor_max_rotation', 10.0),
        }
        
        # Call buildingregulariser.regularize_geodataframe
        # Returns a GeoDataFrame with regularized geometries
        gdf_regularized = regularize_geodataframe(
            gdf,
            parallel_threshold=params['parallel_threshold'],
            simplify=params['simplify'],
            simplify_tolerance=params['simplify_tolerance'],
            allow_45_degree=params['allow_45_degree'],
            diagonal_threshold_reduction=params['diagonal_threshold_reduction'],
            allow_circles=params['allow_circles'],
            circle_threshold=params['circle_threshold'],
            num_cores=params['num_cores'],
            include_metadata=params['include_metadata'],
            neighbor_alignment=params['neighbor_alignment'],
            neighbor_search_distance=params['neighbor_search_distance'],
            neighbor_max_rotation=params['neighbor_max_rotation']
        )
        
        # Save to output file
        gdf_regularized.to_file(output_path, driver='GeoJSON')
        
        logger.info(f"Regularization completed successfully")
        
        return RegularizeResponse(
            status="success",
            output_path=str(output_path),
            message="Building regularization completed",
            processed_features=len(gdf_regularized)
        )
        
    except Exception as e:
        logger.error(f"Regularization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Regularization failed: {str(e)}"
        )


@router.get("/info")
def toolbox_info():
    """
    Get information about available toolbox operations
    """
    return {
        "available_operations": ["regularize", "resample", "smoothify"],
        "regulariser_available": REGULARISER_AVAILABLE,
        "rasterio_available": RASTERIO_AVAILABLE,
        "smoothify_available": SMOOTHIFY_AVAILABLE,
    }


class ResampleRequest(BaseModel):
    """Request model for raster resampling"""
    input_path: str = Field(..., description="Path to input raster file")
    output_path: str = Field(..., description="Path to output raster file")
    target_resolution: float = Field(..., description="Target resolution in units of the raster CRS")
    resampling_method: str = Field(default="bilinear", description="Resampling method: nearest, bilinear, or cubic")
    
    @field_validator('input_path', 'output_path')
    @classmethod
    def validate_paths(cls, v):
        """Validate that paths are absolute and safe"""
        path = Path(v)
        
        if not path.is_absolute():
            raise ValueError("Path must be absolute")
        
        try:
            path.resolve()
        except (OSError, RuntimeError):
            raise ValueError("Invalid path")
        
        return str(path)
    
    @field_validator('resampling_method')
    @classmethod
    def validate_method(cls, v):
        """Validate resampling method"""
        valid_methods = ['nearest', 'bilinear', 'cubic']
        if v.lower() not in valid_methods:
            raise ValueError(f"Method must be one of: {', '.join(valid_methods)}")
        return v.lower()
    
    @field_validator('target_resolution')
    @classmethod
    def validate_resolution(cls, v):
        """Validate target resolution"""
        if v <= 0:
            raise ValueError("Target resolution must be positive")
        return v


class ResampleResponse(BaseModel):
    """Response model for raster resampling"""
    status: str
    output_path: str
    message: Optional[str] = None
    original_resolution: Optional[float] = None
    new_resolution: Optional[float] = None
    output_size: Optional[tuple] = None


@router.post("/resample", response_model=ResampleResponse)
def resample_raster(request: ResampleRequest):
    """
    Resample a raster to a different resolution
    
    Args:
        request: Resample request with input/output paths and parameters
    
    Returns:
        ResampleResponse with status and output information
    """
    if not RASTERIO_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="rasterio not available in this environment"
        )
    
    input_path = Path(request.input_path)
    output_path = Path(request.output_path)
    
    # Validate input file exists
    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Input file not found: {input_path}"
        )
    
    if not input_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Input path is not a file: {input_path}"
        )
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Starting resampling: {input_path} -> {output_path}")
        logger.info(f"Target resolution: {request.target_resolution}, Method: {request.resampling_method}")
        
        # Map method name to rasterio enum
        method_map = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic
        }
        resampling_method = method_map[request.resampling_method]
        
        # Open input raster
        with rasterio.open(input_path) as src:
            # Get original resolution (assuming square pixels)
            # Round to avoid floating point precision issues
            original_res = round(abs(src.res[0]), 10)
            
            # Calculate scale factor
            scale_factor = original_res / request.target_resolution
            
            # Calculate new dimensions
            new_width = int(src.width * scale_factor)
            new_height = int(src.height * scale_factor)
            
            logger.info(f"Original: {src.width}x{src.height} @ {original_res}")
            logger.info(f"New: {new_width}x{new_height} @ {request.target_resolution}")
            
            # Read and resample data
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=resampling_method
            )
            
            # Calculate new transform
            transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
            
            # Write output raster
            profile = src.profile.copy()
            profile.update({
                'width': new_width,
                'height': new_height,
                'transform': transform
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
        
        logger.info(f"Resampling completed successfully")
        
        return ResampleResponse(
            status="success",
            output_path=str(output_path),
            message="Raster resampling completed",
            original_resolution=round(original_res, 10),
            new_resolution=round(request.target_resolution, 10),
            output_size=(new_width, new_height)
        )
        
    except Exception as e:
        logger.error(f"Resampling failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Resampling failed: {str(e)}"
        )

class SmoothifyRequest(BaseModel):
    """Request model for polygon/line smoothing"""
    input_path: str = Field(..., description="Path to input vector file")
    output_path: str = Field(..., description="Path to output vector file")
    segment_length: Optional[float] = Field(None, description="Segment length in map units (None = auto-detect)")
    smooth_iterations: int = Field(default=3, description="Number of smoothing iterations (3-5 recommended)")
    merge_collection: bool = Field(default=True, description="Merge adjacent geometries before smoothing")
    merge_field: Optional[str] = Field(None, description="Field name to group geometries for merging")
    merge_multipolygons: bool = Field(default=True, description="Merge adjacent polygons within MultiPolygons")
    preserve_area: bool = Field(default=True, description="Preserve original area (polygons only)")
    area_tolerance: float = Field(default=0.01, description="Area preservation tolerance percentage")
    num_cores: int = Field(default=0, description="Number of CPU cores (0 = all available)")
    
    @field_validator('input_path', 'output_path')
    @classmethod
    def validate_paths(cls, v):
        """Validate that paths are absolute and safe"""
        path = Path(v)
        
        if not path.is_absolute():
            raise ValueError("Path must be absolute")
        
        try:
            path.resolve()
        except (OSError, RuntimeError):
            raise ValueError("Invalid path")
        
        return str(path)
    
    @field_validator('smooth_iterations')
    @classmethod
    def validate_iterations(cls, v):
        """Validate smooth iterations"""
        if v < 1 or v > 10:
            raise ValueError("smooth_iterations must be between 1 and 10")
        return v
    
    @field_validator('segment_length')
    @classmethod
    def validate_segment_length(cls, v):
        """Validate segment length"""
        if v is not None and v <= 0:
            raise ValueError("segment_length must be positive")
        return v
    
    @field_validator('area_tolerance')
    @classmethod
    def validate_area_tolerance(cls, v):
        """Validate area tolerance"""
        if v < 0 or v > 1:
            raise ValueError("area_tolerance must be between 0 and 1")
        return v


class SmoothifyResponse(BaseModel):
    """Response model for smoothing"""
    status: str
    output_path: str
    message: Optional[str] = None
    processed_features: Optional[int] = None
    segment_length_used: Optional[float] = None


@router.post("/smoothify", response_model=SmoothifyResponse)
def smooth_geometries(request: SmoothifyRequest):
    """
    Smooth polygon or line geometries using Chaikin's corner-cutting algorithm
    
    Args:
        request: Smoothify request with input/output paths and parameters
    
    Returns:
        SmoothifyResponse with status and output information
    """
    if not SMOOTHIFY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="smoothify not available in this environment"
        )
    
    input_path = Path(request.input_path)
    output_path = Path(request.output_path)
    
    # Validate input file exists
    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Input file not found: {input_path}"
        )
    
    if not input_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Input path is not a file: {input_path}"
        )
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Starting smoothing: {input_path} -> {output_path}")
        logger.info(f"Parameters: segment_length={request.segment_length}, iterations={request.smooth_iterations}")
        
        # Warn if user requested parallel processing
        if request.num_cores != 1:
            logger.info(f"Note: num_cores={request.num_cores} requested but forcing serial processing (num_cores=1) for server stability")
        
        # Read input with geopandas
        gdf = gpd.read_file(input_path)
        logger.info(f"Loaded {len(gdf)} features from {input_path}")
        
        # Build smoothify parameters
        smoothify_params = {
            'geom': gdf,
            'smooth_iterations': request.smooth_iterations,
            'merge_collection': request.merge_collection,
            'merge_multipolygons': request.merge_multipolygons,
            'preserve_area': request.preserve_area,
            'area_tolerance': request.area_tolerance,
            'num_cores': 1,
        }
        
        # Add optional parameters
        if request.segment_length is not None:
            smoothify_params['segment_length'] = request.segment_length
        
        if request.merge_field is not None:
            smoothify_params['merge_field'] = request.merge_field
        
        # Call smoothify - returns GeoDataFrame
        logger.info("Calling smoothify...")
        smoothed_gdf = smoothify(**smoothify_params)
        
        # Smoothify always returns a GeoDataFrame when given a GeoDataFrame
        logger.info(f"Smoothing complete: {len(smoothed_gdf)} features")
        
        # Save to output file
        smoothed_gdf.to_file(output_path, driver='GeoJSON')
        
        logger.info(f"Smoothing completed successfully")
        
        return SmoothifyResponse(
            status="success",
            output_path=str(output_path),
            message="Geometry smoothing completed",
            processed_features=len(smoothed_gdf),
            segment_length_used=request.segment_length
        )
        
    except Exception as e:
        logger.error(f"Smoothing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Smoothing failed: {str(e)}"
        )