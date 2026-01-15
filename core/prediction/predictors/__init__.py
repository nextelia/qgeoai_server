"""
Predictors for QPredict Server

Factory pattern to create task-specific predictors for inference.
Server version: works with tiles+metadata, no raster_processor.

Supported tasks:
- Semantic Segmentation (pixel-wise classification)
- Instance Segmentation (object detection + segmentation)
- YOLO Detection/Segmentation/OBB
"""

from .base_predictor import BasePredictor
from .semantic_predictor import SemanticSegmentationPredictor
from .instance_predictor import InstanceSegmentationPredictor
from .yolo_predictor import YOLOPredictor


def create_predictor(
    task: str,
    model,
    device,
    model_info: dict,
    output_shape: tuple,
    **kwargs
):
    """
    Factory function to create appropriate predictor based on task.
    
    Server version: predictors work with pre-generated tiles.
    
    Args:
        task: Task type ('semantic_segmentation', 'instance_segmentation', 'yolo_*')
        model: PyTorch model (already loaded and on device)
        device: Device where model is located (torch.device)
        model_info: Model information from QMTPLoader
        output_shape: (height, width) of the output
        **kwargs: Additional task-specific arguments (batch_size, thresholds, etc.)
        
    Returns:
        Appropriate predictor instance (BasePredictor subclass)
        
    Raises:
        ValueError: If task is not supported
    """
    # Normalize task name
    task_lower = task.lower()
    
    # Route to appropriate predictor
    if task_lower == 'semantic_segmentation':
        # Extract batch_size to avoid duplicate in kwargs
        batch_size = kwargs.pop('batch_size', 4)
        return SemanticSegmentationPredictor(
            model=model,
            batch_size=batch_size,
            device=device,
            model_info=model_info,
            output_shape=output_shape,
            **kwargs
        )
    
    elif task_lower == 'instance_segmentation':
        # Extract batch_size to avoid duplicate in kwargs
        batch_size = kwargs.pop('batch_size', 2)
        return InstanceSegmentationPredictor(
            model=model,
            batch_size=batch_size,
            device=device,
            model_info=model_info,
            output_shape=output_shape,
            **kwargs
        )
    
    elif task_lower.startswith('yolo_'):
        # YOLO tasks: yolo_detect, yolo_segment, yolo_obb
        # Extract batch_size to avoid duplicate in kwargs
        batch_size = kwargs.pop('batch_size', 4)
        return YOLOPredictor(
            model=model,
            batch_size=batch_size,
            device=device,
            model_info=model_info,
            output_shape=output_shape,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unsupported task: {task}\n"
            f"Supported tasks: semantic_segmentation, instance_segmentation, yolo_detect, yolo_segment, yolo_obb"
        )


__all__ = [
    'BasePredictor',
    'SemanticSegmentationPredictor',
    'InstanceSegmentationPredictor',
    'YOLOPredictor',
    'create_predictor'
]