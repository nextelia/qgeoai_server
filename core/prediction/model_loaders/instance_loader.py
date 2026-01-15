"""
Instance Segmentation Model Loader for QPredict

Loads instance segmentation models (Mask R-CNN) using torchvision.
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional

try:
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from .base_loader import BaseModelLoader


class InstanceSegmentationLoader(BaseModelLoader):
    """
    Loader for instance segmentation models (Mask R-CNN).
    
    Supports architectures from torchvision:
    - Mask R-CNN with ResNet50-FPN backbone
    - Mask R-CNN with ResNet50-FPN-v2 backbone
    
    Handles multispectral inputs (RGB, RGBN, etc.) by adapting
    the first convolutional layer when num_channels != 3.
    
    Args:
        qmtp_loader: Instance of QMTPLoader with loaded model info
        device: Device to load model on ('cpu' or 'cuda')
        cuda_device_id: CUDA device ID if using CUDA
        
    Example:
        >>> loader = InstanceSegmentationLoader(qmtp_loader, device='cuda')
        >>> model = loader.load_model()
        >>> print(f"Model loaded on {loader.get_device_info()}")
    """
    
    def _create_model(self) -> nn.Module:
        """
        Create Mask R-CNN model architecture.
        
        Uses torchvision to reconstruct the model with the same
        backbone and number of classes as during training.
        
        Returns:
            Model instance (not yet loaded with trained weights)
            
        Raises:
            ImportError: If torchvision is not installed
            ValueError: If architecture or backbone is not supported
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is not installed.\n"
                "Install with: pip install torchvision"
            )
        
        # Extract model configuration
        architecture = self.config['model']['architecture']
        backbone = self.config['model']['backbone']
        num_channels = self.config['model']['in_channels']
        num_classes = self.config['model']['num_classes']
        pretrained_weights = self.config['model'].get('pretrained_weights', 'random')
        
        # Validate architecture
        if architecture != 'Mask R-CNN':
            raise ValueError(
                f"Unsupported architecture: {architecture}\n"
                f"For instance segmentation, only 'Mask R-CNN' is supported."
            )
        
        # Create base model based on backbone
        if backbone == 'resnet50_fpn':
            # Don't load pretrained weights here - we'll load our trained weights
            model = maskrcnn_resnet50_fpn(weights=None)
        
        elif backbone == 'resnet50_fpn_v2':
            try:
                from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
                model = maskrcnn_resnet50_fpn_v2(weights=None)
            except ImportError:
                raise ImportError(
                    "maskrcnn_resnet50_fpn_v2 requires torchvision >= 0.13\n"
                    "Please upgrade: pip install --upgrade torchvision"
                )
        
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}\n"
                f"Supported: resnet50_fpn, resnet50_fpn_v2"
            )
        
        # ====================================================================
        # HANDLE num_channels != 3 (for multispectral images)
        # ====================================================================
        if num_channels != 3:
            # 1. Modify transform normalization for additional channels
            transform = model.transform
            
            # Extend ImageNet mean/std to num_channels
            rgb_mean = [0.485, 0.456, 0.406]
            rgb_std = [0.229, 0.224, 0.225]
            
            mean_of_means = sum(rgb_mean) / len(rgb_mean)
            mean_of_stds = sum(rgb_std) / len(rgb_std)
            
            transform.image_mean = rgb_mean + [mean_of_means] * (num_channels - 3)
            transform.image_std = rgb_std + [mean_of_stds] * (num_channels - 3)
            
            # 2. Modify first convolutional layer to accept num_channels
            original_layer = model.backbone.body.conv1
            
            model.backbone.body.conv1 = torch.nn.Conv2d(
                num_channels,
                original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=original_layer.bias is not None,
            )
            
            # Initialize weights (will be overwritten when loading state_dict)
            # But this ensures correct architecture
            with torch.no_grad():
                # Initialize all channels with small random values
                torch.nn.init.kaiming_normal_(model.backbone.body.conv1.weight, mode='fan_out', nonlinearity='relu')
                
                if original_layer.bias is not None:
                    model.backbone.body.conv1.bias.zero_()
        
        # ====================================================================
        # Replace prediction heads with correct num_classes
        # ====================================================================
        
        # Replace box predictor head
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features_box,
            num_classes
        )
        
        # Replace mask predictor head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        
        return model