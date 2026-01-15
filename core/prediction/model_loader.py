"""
Model Loader for QPredict - SERVER VERSION

This module handles reconstruction and loading of PyTorch models from .qmtp files.
Uses the new model_loaders factory system to support multiple tasks:
- Semantic Segmentation (UNet, FPN, DeepLabV3, etc.)
- Instance Segmentation (Mask R-CNN)
- Object Detection (YOLO)

The model is returned ready for inference:
- In eval() mode
- On the specified device (CPU/CUDA)
- With gradients disabled

"""

import torch
import warnings
from pathlib import Path
from typing import Optional
from core.prediction.model_loaders import create_model_loader


class ModelLoader:
    """
    Load and reconstruct PyTorch models from .qmtp files.
    
    This class handles:
    - Model architecture reconstruction using segmentation_models_pytorch
    - Weight loading from .qmtp archive
    - Device placement (CPU/CUDA)
    - Preparation for inference (eval mode)
    
    Args:
        qmtp_loader: Instance of QMTPLoader with loaded model info
        device: Device to load model on ('cpu' or 'cuda')
        cuda_device_id: CUDA device ID if using CUDA (default: 0)
        
    Example:
        >>> from qmtp_loader import QMTPLoader
        >>> qmtp_loader = QMTPLoader("model.qmtp")
        >>> qmtp_loader.validate()
        >>> 
        >>> model_loader = ModelLoader(qmtp_loader, device='cuda')
        >>> model = model_loader.load_model()
        >>> # Model is ready for inference
    """
    
    def __init__(
        self,
        qmtp_loader,
        device: str = 'cpu',
        cuda_device_id: int = 0
    ):
        """
        Initialize model loader.
        
        Uses new factory system to automatically select the correct loader
        based on the model's task (semantic, instance, detection).
        
        Args:
            qmtp_loader: Instance of QMTPLoader with loaded model info
            device: Device to load model on ('cpu' or 'cuda')
            cuda_device_id: CUDA device ID if using CUDA
        """
        self.qmtp_loader = qmtp_loader
        self.device_type = device.lower()
        self.cuda_device_id = cuda_device_id
        
        # Create task-specific loader using factory
        self._task_loader = create_model_loader(
            qmtp_loader=qmtp_loader,
            device=device,
            cuda_device_id=cuda_device_id
        )
        
        # Expose commonly used attributes
        self.device = self._task_loader.device
        self.model_info = self._task_loader.model_info
        self.config = self._task_loader.config
    
    def load_model(self) -> torch.nn.Module:
        """
        Load and reconstruct the PyTorch model.
        
        Delegates to task-specific loader (semantic, instance, detection).
        
        This method:
        1. Extracts model.pth from .qmtp to temp directory
        2. Reconstructs model architecture (task-specific)
        3. Loads trained weights
        4. Moves to target device
        5. Sets model to eval() mode
        
        Returns:
            PyTorch model ready for inference
            
        Raises:
            ValueError: If architecture is not supported
            RuntimeError: If model loading fails
            
        Example:
            >>> loader = ModelLoader(qmtp_loader, device='cuda')
            >>> model = loader.load_model()
            >>> # Model automatically selected based on task
            >>> with torch.no_grad():
            ...     output = model(input_tensor)
        """
        # Delegate to task-specific loader
        return self._task_loader.load_model()

    
    def get_device_info(self) -> str:
        """
        Get information about the device being used.
        
        Delegates to task-specific loader.
        
        Returns:
            Human-readable device info string
            
        Example:
            >>> loader = ModelLoader(qmtp_loader, device='cuda')
            >>> print(loader.get_device_info())
            CUDA (GPU 0: NVIDIA GeForce RTX 4050)
        """
        return self._task_loader.get_device_info()