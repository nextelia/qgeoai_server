"""
Model Loaders for QPredict

Factory pattern to load PyTorch models from .qmtp files based on task type.

Supported tasks:
- Semantic Segmentation (UNet, FPN, DeepLabV3, etc.)
- Instance Segmentation (Mask R-CNN)
- Object Detection (YOLO)

Usage:
    >>> from core.prediction.model_loaders import create_model_loader
    >>> loader = create_model_loader(qmtp_loader, device='cuda')
    >>> model = loader.load_model()
"""


from core.prediction.model_loaders.base_loader import BaseModelLoader
from core.prediction.model_loaders.semantic_loader import SemanticSegmentationLoader
from core.prediction.model_loaders.instance_loader import InstanceSegmentationLoader
# YOLOLoader imported lazily in create_model_loader() to avoid circular imports


def create_model_loader(qmtp_loader, device='cpu', cuda_device_id=0):
    """
    Factory function to create appropriate model loader based on task.
    
    Args:
        qmtp_loader: Instance of QMTPLoader with loaded model info
        device: Device to load model on ('cpu' or 'cuda')
        cuda_device_id: CUDA device ID if using CUDA
        
    Returns:
        Appropriate model loader instance (BaseModelLoader subclass)
        
    Raises:
        ValueError: If task is not supported
    """
    # Load model info to determine task
    model_info = qmtp_loader.load_info()
    task = model_info['configuration']['model'].get('task', 'semantic_segmentation')
    
    # Route to appropriate loader
    if task == 'semantic_segmentation':
        return SemanticSegmentationLoader(qmtp_loader, device, cuda_device_id)
    
    elif task == 'instance_segmentation':
        return InstanceSegmentationLoader(qmtp_loader, device, cuda_device_id)
    
    elif task.startswith('yolo_'):
        # YOLO tasks: yolo_detect, yolo_segment, yolo_obb
        from core.prediction.model_loaders.yolo_loader import YOLOLoader
        return YOLOLoader(qmtp_loader, device, cuda_device_id)
    
    elif task == 'object_detection':
        # Deprecated: old task name, redirect to YOLO
        from core.prediction.model_loaders.yolo_loader import YOLOLoader
        return YOLOLoader(qmtp_loader, device, cuda_device_id)
    
    else:
        raise ValueError(
            f"Unsupported task: {task}\n"
            f"Supported tasks: semantic_segmentation, instance_segmentation, yolo_detect, yolo_segment, yolo_obb"
        )


__all__ = [
    'BaseModelLoader',
    'SemanticSegmentationLoader',
    'InstanceSegmentationLoader',
    'YOLOLoader',
    'create_model_loader'
]