"""
QPredict Models Endpoint

Handles model loading, unloading, and listing.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional

from .schemas import (
    LoadModelRequest,
    LoadModelResponse,
    UnloadModelRequest,
    UnloadModelResponse,
    ListModelsResponse,
    ModelInfo
)
from .model_manager import model_manager

router = APIRouter(prefix="/qpredict", tags=["qpredict-models"])


@router.post("/load-model", response_model=LoadModelResponse)
async def load_model(
    qmtp_file: UploadFile = File(..., description=".qmtp model file"),
    device: str = Form(default='cpu', description="Device: 'cpu' or 'cuda'"),
    cuda_device_id: int = Form(default=0, description="CUDA device ID (0-7)")
):
    """
    Load a PyTorch model from .qmtp file.
    
    The model is loaded into memory and assigned a unique UUID.
    Use this UUID for subsequent prediction requests.
    
    Returns:
        Model information including UUID, task type, architecture, and classes.
        
    Raises:
        HTTPException 400: If model loading fails
    """
    try:
        # Validate and normalize device
        device = device.lower()
        if device not in ['cpu', 'cuda']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid device '{device}'. Must be 'cpu' or 'cuda'."
            )
        # Validate cuda_device_id
        if not (0 <= cuda_device_id <= 7):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cuda_device_id {cuda_device_id}. Must be 0-7."
            )
        
        # Read file content
        content = await qmtp_file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        # Load model
        loaded_model = model_manager.load_model(
            qmtp_file_content=content,
            device=device,
            cuda_device_id=cuda_device_id
        )
        
        # Return response
        return LoadModelResponse(**loaded_model.get_info())
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/unload-model", response_model=UnloadModelResponse)
async def unload_model(request: UnloadModelRequest):
    """
    Unload a model from memory.
    
    Frees GPU/CPU memory and removes the model from the server.
    
    Returns:
        Success status and message.
        
    Raises:
        HTTPException 404: If model_id not found
    """
    try:
        success = model_manager.unload_model(request.model_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} not found"
            )
        
        return UnloadModelResponse(
            success=True,
            message=f"Model {request.model_id} unloaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/models", response_model=ListModelsResponse)
async def list_models():
    """
    List all currently loaded models.
    
    Returns:
        List of loaded models with their metadata.
    """
    try:
        models = model_manager.list_models()
        
        return ListModelsResponse(
            models=[ModelInfo(**m) for m in models],
            total=len(models)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )