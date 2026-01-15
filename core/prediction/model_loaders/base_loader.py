"""
Base Model Loader for QPredict

Abstract base class for all model loaders.
Defines common interface and shared functionality.
"""

import torch
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseModelLoader(ABC):
    """
    Abstract base class for model loaders.
    
    All task-specific loaders must inherit from this class and implement:
    - _create_model() - How to construct the model architecture
    
    Common functionality provided:
    - Device management (CPU/CUDA)
    - Weight extraction from .qmtp
    - Weight loading and device placement
    - Model preparation for inference (eval mode)
    
    Args:
        qmtp_loader: Instance of QMTPLoader with loaded model info
        device: Device to load model on ('cpu' or 'cuda')
        cuda_device_id: CUDA device ID if using CUDA (default: 0)
    """
    
    def __init__(
        self,
        qmtp_loader,
        device: str = 'cpu',
        cuda_device_id: int = 0
    ):
        """
        Initialize base model loader.
        
        Args:
            qmtp_loader: Instance of QMTPLoader with loaded model info
            device: Device to load model on ('cpu' or 'cuda')
            cuda_device_id: CUDA device ID if using CUDA
        """
        self.qmtp_loader = qmtp_loader
        self.device_type = device.lower()
        self.cuda_device_id = cuda_device_id
        
        # Determine actual device
        if self.device_type == 'cuda':
            if not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(f'cuda:{cuda_device_id}')
        else:
            self.device = torch.device('cpu')
        
        # Load model info from qmtp_loader
        self.model_info = self.qmtp_loader.load_info()
        self.config = self.model_info['configuration']
    
    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        """
        Create model architecture (task-specific).
        
        Must be implemented by subclasses.
        
        Returns:
            Model instance (not yet loaded with trained weights)
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    def load_model(self) -> torch.nn.Module:
        """
        Load and reconstruct the PyTorch model.
        
        This method:
        1. Extracts model.pth from .qmtp to temp directory
        2. Reconstructs model architecture using _create_model()
        3. Loads trained weights
        4. Moves model to specified device
        5. Sets model to eval() mode
        
        Returns:
            PyTorch model ready for inference
            
        Raises:
            RuntimeError: If model loading fails
            
        Example:
            >>> loader = SemanticSegmentationLoader(qmtp_loader, device='cuda')
            >>> model = loader.load_model()
            >>> # Run inference
            >>> with torch.no_grad():
            ...     output = model(input_tensor)
        """
        # Create temporary directory for model weights
        temp_dir = tempfile.mkdtemp(prefix='qpredict_')
        
        try:
            # Extract model.pth from .qmtp
            weights_path = self.qmtp_loader.extract_model_weights(temp_dir)
            
            # Reconstruct model architecture (task-specific)
            model = self._create_model()
            
            # Load trained weights
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            # Move to target device
            model = model.to(self.device)
            
            # Set to evaluation mode (disables dropout, batchnorm, etc.)
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
        finally:
            # Cleanup temporary directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  
    
    def get_device_info(self) -> str:
        """
        Get information about the device being used.
        
        Returns:
            Human-readable device info string
            
        """
        if self.device.type == 'cpu':
            return "CPU"
        else:
            gpu_name = torch.cuda.get_device_name(self.device.index)
            return f"CUDA (GPU {self.device.index}: {gpu_name})"