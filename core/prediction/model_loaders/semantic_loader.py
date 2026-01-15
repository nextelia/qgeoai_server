"""
Semantic Segmentation Model Loader for QPredict

Loads semantic segmentation models (UNet, FPN, DeepLabV3, etc.)
using segmentation-models-pytorch library.
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

from .base_loader import BaseModelLoader


class SemanticSegmentationLoader(BaseModelLoader):
    """
    Loader for semantic segmentation models.
    
    Supports architectures from segmentation-models-pytorch:
    - UNet, UNet++
    - FPN
    - DeepLabV3, DeepLabV3+
    - PSPNet
    - LinkNet
    - MANet, PAN
    
    Args:
        qmtp_loader: Instance of QMTPLoader with loaded model info
        device: Device to load model on ('cpu' or 'cuda')
        cuda_device_id: CUDA device ID if using CUDA
        
    Example:
        >>> loader = SemanticSegmentationLoader(qmtp_loader, device='cuda')
        >>> model = loader.load_model()
        >>> print(f"Model loaded on {loader.get_device_info()}")
    """

  # Architecture name mapping (UI display name → SMP class name)
    # Must match QModel Trainer's SMP_ARCHITECTURES
    ARCHITECTURE_MAPPING = {
        'UNet': 'Unet',
        'UNet++': 'UnetPlusPlus',
        'DeepLabV3': 'DeepLabV3',
        'DeepLabV3+': 'DeepLabV3Plus',
        'LinkNet': 'Linknet',
        'Segformer': 'Segformer',
        'DPT': 'DPT',
        'FPN': 'FPN',
        'PSPNet': 'PSPNet',
        'MANet': 'MAnet',
        'PAN': 'PAN'
    }

    # Divisibility constraints per architecture
    # Some architectures require input dimensions to be divisible by specific factors
    ARCHITECTURE_DIVISIBILITY = {
        'UNet': 1,          # No strict constraint
        'UNet++': 32,       # Deep encoder requires 32x divisibility
        'DeepLabV3': 8,     # Standard constraint
        'DeepLabV3+': 16,   # ASPP + decoder
        'LinkNet': 32,      # Encoder-decoder structure
        'Segformer': 32,    # Transformer patches
        'DPT': 16,          # Vision transformer
        'FPN': 32,          # Feature pyramid
        'PSPNet': 8,        # Pyramid pooling
        'MANet': 32,        # Multi-scale attention
        'PAN': 32           # Pyramid attention
    }
    
    def _create_model(self) -> nn.Module:
        """
        Create semantic segmentation model architecture.
        
        Uses segmentation_models_pytorch to reconstruct the model
        with the same architecture and encoder as during training.
        
        Returns:
            Model instance (not yet loaded with trained weights)
            
        Raises:
            ImportError: If segmentation-models-pytorch is not installed
            ValueError: If architecture is not supported
        """
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation-models-pytorch is not installed.\n"
                "Install with: pip install segmentation-models-pytorch"
            )
        
        # Extract model configuration
        architecture = self.config['model']['architecture']
        encoder = self.config['model']['encoder']
        encoder_weights = self.config['model']['encoder_weights']
        in_channels = self.config['model']['in_channels']
        num_classes = self.config['model']['num_classes']
        
        # Normalize encoder_weights
        # In .qmtp: 'imagenet', 'ssl', 'swsl', 'random', or None
        # For smp: 'imagenet', 'ssl', 'swsl', or None
        if encoder_weights == 'random' or encoder_weights is None:
            encoder_weights = None
        
        # Get SMP class name (handle special characters like + in UNet++)
        smp_class_name = self.ARCHITECTURE_MAPPING.get(architecture)
        
        if smp_class_name is None:
            raise ValueError(
                f"Unknown architecture: {architecture}\n"
                f"Supported architectures: {', '.join(self.ARCHITECTURE_MAPPING.keys())}"
            )
        
        # Suppress warnings from timm/huggingface during model creation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            
            try:
                # Get model class from segmentation_models_pytorch
                model_class = getattr(smp, smp_class_name)
                
                # Create model
                model = model_class(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=num_classes
                )
                
            except AttributeError:
                raise ValueError(
                    f"Architecture '{architecture}' (SMP class: {smp_class_name}) not found in segmentation-models-pytorch.\n"
                    f"Available models: {', '.join([m for m in dir(smp) if not m.startswith('_')])}\n"
                    f"Make sure segmentation-models-pytorch is up to date."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create model {architecture} with encoder {encoder}.\n"
                    f"Error: {str(e)}\n"
                    f"Check that encoder '{encoder}' is compatible with architecture '{architecture}'."
                )
        
        # Store divisibility constraint in model for raster processor
        # This ensures tiles will be padded to compatible dimensions
        architecture = self.config['model']['architecture']
        model._qpredict_divisibility = self.ARCHITECTURE_DIVISIBILITY.get(architecture, 1)
        
        return model