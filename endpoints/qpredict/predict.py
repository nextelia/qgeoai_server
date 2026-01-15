"""
QPredict Prediction Endpoint

Handles inference on tiles using loaded models.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
import torch
import base64
import io
from .schemas import PredictRequest, PredictResponse
from .model_manager import model_manager
from core.prediction.predictors import create_predictor

router = APIRouter(prefix="/qpredict", tags=["qpredict-prediction"])

def decode_tile(encoded_tile: str) -> np.ndarray:
    """
    Decode a base64-encoded numpy array.
    
    Args:
        encoded_tile: Base64-encoded numpy array
        
    Returns:
        Numpy array (C, H, W)
    """
    # Decode base64
    tile_bytes = base64.b64decode(encoded_tile)
    
    # Load numpy array
    buffer = io.BytesIO(tile_bytes)
    tile_array = np.load(buffer, allow_pickle=False)
    
    return tile_array


def encode_prediction(prediction: np.ndarray) -> str:
    """
    Encode a numpy array to base64.
    
    Args:
        prediction: Numpy array
        
    Returns:
        Base64-encoded string
    """
    buffer = io.BytesIO()
    np.save(buffer, prediction)
    buffer.seek(0)
    
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    return encoded


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run inference on tiles using a loaded model.
    
    Process:
    1. Decode tiles from base64
    2. Convert to PyTorch tensors
    3. Run inference with appropriate predictor
    4. Encode results to base64
    
    Returns:
        Prediction results (format depends on task type)
        
    Raises:
        HTTPException 404: If model not found
        HTTPException 400: If request is invalid
        HTTPException 500: If inference fails
    """
    try:
        # Get loaded model
        loaded_model = model_manager.get_model(request.model_id)
        
        if loaded_model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} not found. Load it first with /load-model"
            )
        
        # Validate tiles count matches metadata count
        if len(request.tiles) != len(request.tile_metadata):
            raise HTTPException(
                status_code=400,
                detail=f"Tiles count ({len(request.tiles)}) != metadata count ({len(request.tile_metadata)})"
            )
        
        # Decode tiles
        print(f"📦 Decoding {len(request.tiles)} tiles...")
        tile_arrays = []
        for i, encoded_tile in enumerate(request.tiles):
            try:
                tile_array = decode_tile(encoded_tile)
                tile_arrays.append(tile_array)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode tile {i}: {str(e)}"
                )
        
        # Convert to PyTorch tensors
        device_str = loaded_model.loader.device_type
        device = torch.device(device_str)
        
        tile_tensors = [
            torch.from_numpy(arr).float()
            for arr in tile_arrays
        ]
        
        # Prepare metadata dicts
        tile_metadata_dicts = [
            meta.model_dump()
            for meta in request.tile_metadata
        ]
        
        # Create predictor
        print(f"🎯 Creating predictor for task: {loaded_model.task}")
        
        predictor_kwargs = {
            'batch_size': request.batch_size,
            'output_format': request.output_format,
        }
        
        # Add task-specific parameters
        if loaded_model.task == 'semantic_segmentation':
            predictor_kwargs['apply_colors'] = request.apply_colors
            predictor_kwargs['reliability_type'] = request.reliability_type
            
        elif loaded_model.task == 'instance_segmentation':
            predictor_kwargs['confidence_threshold'] = request.confidence_threshold
            predictor_kwargs['nms_threshold'] = request.nms_threshold
            predictor_kwargs['overlap'] = request.overlap
            
        elif loaded_model.task.startswith('yolo_'):
            predictor_kwargs['confidence_threshold'] = request.confidence_threshold
            predictor_kwargs['iou_threshold'] = request.iou_threshold
            predictor_kwargs['overlap'] = request.overlap
            predictor_kwargs['simplify_tolerance'] = request.simplify_tolerance
        
        predictor = create_predictor(
            task=loaded_model.task,
            model=loaded_model.model,
            device=device,
            model_info=loaded_model.loader.model_info,
            output_shape=request.output_shape,
            **predictor_kwargs
        )
        
        # Run prediction
        print(f"🚀 Running inference...")
        predictor._initialize_accumulation()
        
        # Process tiles in batches
        for i in range(0, len(tile_tensors), request.batch_size):
            batch_tiles = tile_tensors[i:i + request.batch_size]
            batch_metadata = tile_metadata_dicts[i:i + request.batch_size]
            
            predictor._process_batch(batch_tiles, batch_metadata)
        
        # Finalize predictions
        final_prediction = predictor._finalize_predictions()
        
        # Create outputs
        output_data = predictor._create_outputs(final_prediction)
        
        # Prepare response based on output format
        if loaded_model.task == 'semantic_segmentation':
            # Encode numpy array
            encoded_prediction = encode_prediction(output_data['prediction'])
            
            return PredictResponse(
                prediction=encoded_prediction,
                output_format=output_data['output_format'],
                classes=output_data.get('classes'),
                reliability_type=output_data.get('reliability_type')
            )
            
        elif loaded_model.task == 'instance_segmentation':
            # Return instances as JSON (already dict)
            import json
            
            return PredictResponse(
                prediction=json.dumps(output_data['instances']),
                output_format='instances',
                classes=output_data['classes'],
                num_instances=output_data['num_instances']
            )
            
        elif loaded_model.task.startswith('yolo_'):
            # Return detections as JSON
            import json
            import numpy as np
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            clean_detections = convert_numpy(output_data['detections'])
            
            return PredictResponse(
                prediction=json.dumps(clean_detections),
                output_format='detections',
                classes=output_data['classes'],
                num_detections=output_data['num_detections'],
                yolo_task=output_data['yolo_task']
            )
        
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unknown task type: {loaded_model.task}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )