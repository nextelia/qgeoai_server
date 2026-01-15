"""
Model factory for QModel Trainer

This module creates PyTorch models based on user configuration.
Uses segmentation-models-pytorch for semantic segmentation architectures.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

try:
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Supported SMP architectures
SMP_ARCHITECTURES = {
    'UNet': 'Unet',
    'UNet++': 'UnetPlusPlus',
    'DeepLabV3': 'DeepLabV3',
    'DeepLabV3+': 'DeepLabV3Plus',
    'LinkNet': 'Linknet',
    'Segformer': 'Segformer',
    'DPT': 'DPT'
}

# Encoder families (organized for UI)
SMP_ENCODER_FAMILIES = {
    'ResNet': {
        'encoders': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'display_names': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152']
    },
    'ResNeXt': {
        'encoders': ['resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d'],
        'display_names': ['ResNeXt-50', 'ResNeXt-101 (32x4d)', 'ResNeXt-101 (32x8d)']
    },
    'EfficientNet': {
        'encoders': [
            'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
            'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7'
        ],
        'display_names': [
            'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3',
            'EfficientNet-B4', 'EfficientNet-B5', 'EfficientNet-B6', 'EfficientNet-B7'
        ]
    },
    'MobileNet': {
        'encoders': ['mobilenet_v2'],
        'display_names': ['MobileNet V2']
    },
    'DenseNet': {
        'encoders': ['densenet121', 'densenet169', 'densenet201'],
        'display_names': ['DenseNet-121', 'DenseNet-169', 'DenseNet-201']
    },
    'VGG': {
        'encoders': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
        'display_names': ['VGG-11', 'VGG-13', 'VGG-16', 'VGG-19']
    },
    'MIT (Segformer)': {
        'encoders': ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
        'display_names': ['MiT-B0', 'MiT-B1', 'MiT-B2', 'MiT-B3', 'MiT-B4', 'MiT-B5']
    },
    'ViT (DPT)': {
        'encoders': ['vit_base_patch16_384', 'vit_large_patch16_384'],
        'display_names': ['ViT-Base', 'ViT-Large']
    }
}

# Available pretrained weights per encoder
ENCODER_PRETRAINED_WEIGHTS = {
    # ResNet family - ssl/swsl for specific versions
    'resnet18': ['imagenet', 'ssl', 'swsl'],
    'resnet34': ['imagenet'],
    'resnet50': ['imagenet', 'ssl', 'swsl'],
    'resnet101': ['imagenet'],
    'resnet152': ['imagenet'],
    
    # ResNeXt - ssl/swsl available
    'resnext50_32x4d': ['imagenet', 'ssl', 'swsl'],
    'resnext101_32x4d': ['imagenet', 'ssl', 'swsl'],
    'resnext101_32x8d': ['imagenet', 'ssl', 'swsl'],
    
    # EfficientNet - imagenet only (native SMP)
    'efficientnet-b0': ['imagenet'],
    'efficientnet-b1': ['imagenet'],
    'efficientnet-b2': ['imagenet'],
    'efficientnet-b3': ['imagenet'],
    'efficientnet-b4': ['imagenet'],
    'efficientnet-b5': ['imagenet'],
    'efficientnet-b6': ['imagenet'],
    'efficientnet-b7': ['imagenet'],
    
    # Others - imagenet only
    'mobilenet_v2': ['imagenet'],
    'densenet121': ['imagenet'],
    'densenet169': ['imagenet'],
    'densenet201': ['imagenet'],
    'vgg11': ['imagenet'],
    'vgg13': ['imagenet'],
    'vgg16': ['imagenet'],
    'vgg19': ['imagenet'],
    
    # Transformer encoders
    'mit_b0': ['imagenet'],
    'mit_b1': ['imagenet'],
    'mit_b2': ['imagenet'],
    'mit_b3': ['imagenet'],
    'mit_b4': ['imagenet'],
    'mit_b5': ['imagenet'],
    'vit_base_patch16_384': ['imagenet'],
    'vit_large_patch16_384': ['imagenet'],
}

# Architecture-specific encoder constraints
ARCHITECTURE_ENCODER_CONSTRAINTS = {
    'Segformer': ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
    'DPT': ['vit_base_patch16_384', 'vit_large_patch16_384']
}

# Supported backbones for Mask R-CNN (torchvision)
MASKRCNN_BACKBONES = ['resnet50_fpn', 'resnet50_fpn_v2']

# =============================================================================
# MODEL CREATION FUNCTIONS
# =============================================================================

def create_segmentation_model(
    architecture: str,
    backbone: str,
    num_classes: int,
    in_channels: int = 3,
    pretrained: bool = True,
    encoder_weights: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Create a semantic segmentation model using segmentation-models-pytorch.
    
    This function creates models for semantic segmentation tasks using
    the segmentation-models-pytorch library. It supports various architectures
    (UNet, DeepLabV3, etc.) with different encoder backbones.
    
    Args:
        architecture: Architecture name (e.g., 'UNet', 'DeepLabV3')
        backbone: Encoder backbone name (e.g., 'resnet50', 'efficientnet-b0')
        num_classes: Number of output classes (including background)
        in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        pretrained: If True, use ImageNet pretrained weights for encoder
        encoder_weights: Specific pretrained weights to use (e.g., 'imagenet', 'ssl', 'swsl').
                         If None, defaults to 'imagenet' when pretrained=True, or None when pretrained=False.
                         Takes precedence over the pretrained flag.
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        PyTorch model ready for training
        
    Raises:
        ImportError: If segmentation-models-pytorch is not installed
        ValueError: If architecture or backbone is not supported
        
    Example:
        >>> model = create_segmentation_model(
        ...     architecture='UNet',
        ...     backbone='resnet50',
        ...     num_classes=4,
        ...     pretrained=True
        ... )
    """
    if not SMP_AVAILABLE:
        raise ImportError(
            "segmentation-models-pytorch is not installed.\n"
            "Install with: pip install segmentation-models-pytorch"
        )
    
    # Validate architecture
    if architecture not in SMP_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture: {architecture}\n"
            f"Supported architectures: {', '.join(SMP_ARCHITECTURES.keys())}"
        )
    
    # Get SMP model class name
    smp_model_name = SMP_ARCHITECTURES[architecture]
    
    # Set encoder weights
    # Priority: explicit encoder_weights > pretrained flag
    if encoder_weights is None:
        encoder_weights = "imagenet" if pretrained else None
    # If encoder_weights is explicitly provided, use it (even if pretrained=False)
    
    # Get model class from smp
    try:
        model_class = getattr(smp, smp_model_name)
    except AttributeError:
        raise ValueError(
            f"Model {smp_model_name} not found in segmentation-models-pytorch.\n"
            f"Available models: {', '.join(dir(smp))}"
        )
    
    # Create model
    try:
        model = model_class(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None, 
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create model {architecture} with backbone {backbone}.\n"
            f"Error: {str(e)}\n"
            f"Check that the backbone is compatible with the architecture."
        )
    
    return model


def create_detection_model(
    architecture: str,
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create an object detection model.
    
    Args:
        architecture: Architecture name (e.g., 'Faster-RCNN', 'RetinaNet')
        backbone: Backbone name (e.g., 'resnet50')
        num_classes: Number of classes (including background)
        pretrained: If True, use pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        PyTorch model for object detection
        
    Note:
        Not implemented yet. Placeholder for future.
    """
    raise NotImplementedError(
        "Object detection models not implemented yet.\n"
        "Currently only semantic segmentation and Mask-rcnn is supported.\n"
        "Future versions will support Faster-RCNN, RetinaNet, etc."
    )


def create_instance_segmentation_model(
    architecture: str,
    backbone: str,
    num_classes: int,
    num_channels: int = 3,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create an instance segmentation model (Mask R-CNN).

    Uses torchvision's Mask R-CNN implementation with ResNet-FPN backbones.
    Supports multispectral inputs (RGB, RGBN, etc.) by adapting the first
    convolutional layer when num_channels != 3.

    Args:
        architecture: Architecture name (must be 'Mask R-CNN')
        backbone: Backbone name (e.g., 'resnet50_fpn')
        num_classes: Number of classes (INCLUDING background, so real_classes + 1)
        num_channels: Number of input channels (3 for RGB, 4 for RGBN, etc.)
        pretrained: If True, use COCO pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        PyTorch model for instance segmentation
        
    Raises:
        ImportError: If torchvision is not installed
        ValueError: If architecture or backbone is not supported
        
    Example:
        >>> model = create_instance_segmentation_model(
        ...     architecture='Mask R-CNN',
        ...     backbone='resnet50_fpn',
        ...     num_classes=5,  # 4 classes + 1 background
        ...     pretrained=True
        ... )
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError(
            "torchvision is not installed.\n"
            "Install with: pip install torchvision"
        )
    
    # Validate architecture
    if architecture != 'Mask R-CNN':
        raise ValueError(
            f"Unsupported architecture: {architecture}\n"
            f"For instance segmentation, only 'Mask R-CNN' is currently supported."
        )
    
    # Validate backbone
    if backbone not in MASKRCNN_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {backbone}\n"
            f"Supported backbones for Mask R-CNN: {', '.join(MASKRCNN_BACKBONES)}"
        )
    
    # Create base model with COCO pretrained weights (91 classes)
    if backbone == 'resnet50_fpn':
        if pretrained:
            # Load pretrained on COCO (91 classes)
            model = maskrcnn_resnet50_fpn(weights='DEFAULT')
        else:
            # Random initialization
            model = maskrcnn_resnet50_fpn(weights=None)
    
    elif backbone == 'resnet50_fpn_v2':
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
            if pretrained:
                model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
            else:
                model = maskrcnn_resnet50_fpn_v2(weights=None)
        except ImportError:
            raise ImportError(
                "maskrcnn_resnet50_fpn_v2 requires torchvision >= 0.13\n"
                "Please upgrade: pip install --upgrade torchvision"
            )
    
    else:
        raise ValueError(f"Backbone {backbone} not implemented yet")

    # ============================================================================
    # HANDLE num_channels != 3 (for multispectral images, RGBN, etc.)
    # for compatibility with satellite imagery
    # ============================================================================
    if num_channels != 3:
        # 1. Modify transform normalization for additional channels
        transform = model.transform
        
        # Extend ImageNet mean/std to num_channels
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        
        # Calculate mean of means/stds for additional channels
        mean_of_means = sum(rgb_mean) / len(rgb_mean)
        mean_of_stds = sum(rgb_std) / len(rgb_std)
        
        # Extend to num_channels
        transform.image_mean = rgb_mean + [mean_of_means] * (num_channels - 3)
        transform.image_std = rgb_std + [mean_of_stds] * (num_channels - 3)
        
        # 2. Modify first convolutional layer of backbone to accept num_channels
        original_layer = model.backbone.body.conv1
        
        # Create new Conv2d with num_channels input
        model.backbone.body.conv1 = torch.nn.Conv2d(
            num_channels,
            original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            bias=original_layer.bias is not None,
        )
        
        # Copy pretrained weights for first 3 channels
        with torch.no_grad():
            # Copy RGB channels weights
            model.backbone.body.conv1.weight[:, :3, :, :] = original_layer.weight
            
            # Initialize additional channels with mean of RGB weights
            mean_weight = original_layer.weight.mean(dim=1, keepdim=True)
            for i in range(3, num_channels):
                model.backbone.body.conv1.weight[:, i : i + 1, :, :] = mean_weight
            
            # Copy bias if exists
            if original_layer.bias is not None:
                model.backbone.body.conv1.bias = original_layer.bias

    # Replace the box predictor head
    # Get number of input features for the classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the box predictor with a new one (for num_classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features_box,
        num_classes  # num_classes already includes background
    )
    
    # Replace the mask predictor head
    # Get number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # Standard hidden layer size
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes  # num_classes already includes background
    )
    
    return model


# =============================================================================
# MAIN MODEL FACTORY
# =============================================================================

def _normalize_architecture_name(arch: str) -> str:
    """
    Normalize architecture name to canonical form.
    
    Accepts: 'unet', 'UNet', 'UNET' → Returns: 'UNet'
    """
    mapping = {
        'unet': 'UNet',
        'unet++': 'UNet++',
        'deeplabv3': 'DeepLabV3',
        'deeplabv3+': 'DeepLabV3+',
        'linknet': 'LinkNet',
        'segformer': 'Segformer',
        'dpt': 'DPT',
        'mask r-cnn': 'Mask R-CNN',
        'mask-rcnn': 'Mask R-CNN'
    }
    return mapping.get(arch.lower(), arch)


def create_model(
    task: str,
    architecture: str,
    backbone: str,
    num_classes: int,
    in_channels: int = 3,
    pretrained: bool = True,
    encoder_weights: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Main model factory function.
    
    This is the main entry point for creating models. It routes to the
    appropriate model creation function based on the task.
    
    Args:
        task: Task name ('Semantic Segmentation', 'Detection', 'Instance Segmentation')
        architecture: Architecture name (task-dependent)
        backbone: Encoder backbone name
        num_classes: Number of output classes
        in_channels: Number of input channels (default: 3 for RGB)
        pretrained: Whether to use pretrained weights
        encoder_weights: Specific pretrained weights to use (e.g., 'imagenet', 'ssl', 'swsl').
                         If None, defaults to 'imagenet' when pretrained=True.
                         Only applies to semantic segmentation models.
        **kwargs: Additional model-specific arguments
        
    Returns:
        Configured PyTorch model ready for training
        
    Raises:
        ValueError: If task is not supported
        
    Example:
        >>> model = create_model(
        ...     task='Semantic Segmentation',
        ...     architecture='UNet',
        ...     backbone='resnet50',
        ...     num_classes=4,
        ...     pretrained=True
        ... )
        >>> print(f"Model has {count_parameters(model):,} parameters")
    """
    task_lower = task.lower()
    # Normalize architecture name (case-insensitive)
    architecture = _normalize_architecture_name(architecture)
    
    if 'segmentation' in task_lower and 'instance' not in task_lower:
        # Semantic segmentation
        return create_segmentation_model(
            architecture=architecture,
            backbone=backbone,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            encoder_weights=encoder_weights,
            **kwargs
        )
    
    elif 'detection' in task_lower:
        # Object detection
        return create_detection_model(
            architecture=architecture,
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    
    elif 'instance' in task_lower:
        # Instance segmentation
        return create_instance_segmentation_model(
            architecture=architecture,
            backbone=backbone,
            num_classes=num_classes,
            num_channels=in_channels,
            pretrained=pretrained,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unsupported task: {task}\n"
            f"Supported tasks: Semantic Segmentation, Detection, Instance Segmentation"
        )


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
        
    Example:
        >>> model = create_model(...)
        >>> total_params = count_parameters(model)
        >>> trainable_params = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_encoder(model: nn.Module) -> None:
    """
    Freeze encoder weights (for transfer learning with frozen backbone).
    
    This is useful when you want to train only the decoder/head while
    keeping the pretrained encoder frozen.
    
    Args:
        model: Segmentation model from segmentation-models-pytorch
        
    Example:
        >>> model = create_model(...)
        >>> freeze_encoder(model)  # Only decoder will be trained
        >>> # Train for a few epochs
        >>> unfreeze_encoder(model)  # Fine-tune entire model
    """
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen. Only decoder will be trained.")
    else:
        print("Warning: Model doesn't have 'encoder' attribute. Nothing frozen.")


def unfreeze_encoder(model: nn.Module) -> None:
    """
    Unfreeze encoder weights (for fine-tuning entire model).
    
    Args:
        model: Segmentation model from segmentation-models-pytorch
        
    Example:
        >>> model = create_model(...)
        >>> freeze_encoder(model)
        >>> # Train decoder only for N epochs
        >>> unfreeze_encoder(model)
        >>> # Fine-tune entire model
    """
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen. Entire model will be trained.")
    else:
        print("Warning: Model doesn't have 'encoder' attribute. Nothing unfrozen.")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
        
    Example:
        >>> model = create_model(...)
        >>> info = get_model_info(model)
        >>> print(f"Architecture: {info['architecture']}")
        >>> print(f"Total params: {info['total_parameters']:,}")
    """
    info = {
        'total_parameters': count_parameters(model, trainable_only=False),
        'trainable_parameters': count_parameters(model, trainable_only=True),
        'architecture': model.__class__.__name__,
    }
    
    # Try to get encoder info if it's an smp model
    if hasattr(model, 'encoder'):
        info['has_encoder'] = True
        if hasattr(model.encoder, 'name'):
            info['encoder_name'] = model.encoder.name
        else:
            info['encoder_name'] = 'Unknown'
    else:
        info['has_encoder'] = False
    
    # Calculate frozen/unfrozen parameters
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    info['frozen_parameters'] = frozen_params
    info['encoder_frozen'] = frozen_params > 0
    
    return info


def validate_model_config(
    task: str,
    architecture: str,
    backbone: str
) -> tuple[bool, Optional[str]]:
    """
    Validate model configuration before creation.
    
    Args:
        task: Task name
        architecture: Architecture name
        backbone: Backbone name
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate task
    valid_tasks = ['Semantic Segmentation', 'Detection', 'Instance Segmentation']
    if task not in valid_tasks:
        return False, f"Invalid task: {task}. Must be one of {valid_tasks}"
    
    # Task-specific validation
    if task == 'Instance Segmentation':
        if architecture != 'Mask R-CNN':
            return False, f"For Instance Segmentation, only 'Mask R-CNN' architecture is supported, got '{architecture}'"
        return validate_maskrcnn_config(backbone, num_classes=None)
    
    # Semantic Segmentation validation
    if architecture not in SMP_ARCHITECTURES:
        return False, f"Invalid architecture: {architecture}. Supported: {', '.join(SMP_ARCHITECTURES.keys())}"
    
    # Check architecture-specific constraints
    if architecture in ARCHITECTURE_ENCODER_CONSTRAINTS:
        allowed_encoders = ARCHITECTURE_ENCODER_CONSTRAINTS[architecture]
        if backbone not in allowed_encoders:
            return False, (
                f"Architecture '{architecture}' requires specific encoders.\n"
                f"Allowed: {', '.join(allowed_encoders)}\n"
                f"Got: {backbone}"
            )
    
    # Check if encoder exists in any family
    all_encoders = []
    for family_data in SMP_ENCODER_FAMILIES.values():
        all_encoders.extend(family_data['encoders'])
    
    if backbone not in all_encoders:
        return False, f"Unknown encoder: {backbone}"
    
    return True, None

def validate_maskrcnn_config(
    backbone: str,
    num_classes: Optional[int] = None
) -> tuple[bool, Optional[str]]:
    """
    Validate Mask R-CNN configuration.
    
    Args:
        backbone: Backbone name
        num_classes: Number of classes (must include background), optional
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate backbone
    if backbone not in MASKRCNN_BACKBONES:
        return False, f"Invalid backbone: {backbone}. Supported: {', '.join(MASKRCNN_BACKBONES)}"
    
    # Validate num_classes (only if provided)
    if num_classes is not None and num_classes < 2:
        return False, f"num_classes must be >= 2 (background + at least 1 class), got {num_classes}"
    
    return True, None