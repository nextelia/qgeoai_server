"""
Optimizer and Scheduler configuration factory for QModel Trainer

Provides default parameters for user-selected optimizers and schedulers.
"""

from typing import Dict, Any, Optional


# =============================================================================
# TASK MAPPING
# =============================================================================

TASK_MAP = {
    'Semantic Segmentation': 'semantic_segmentation',
    'Instance Segmentation': 'instance_segmentation',
    'YOLO11 Detection': 'yolo_detection',
    'YOLO11 Segmentation': 'yolo_segmentation',
    'YOLO11 OBB': 'yolo_obb'
}


# =============================================================================
# OPTIMIZER CONFIGURATIONS (PyTorch only)
# =============================================================================

OPTIMIZER_CONFIGS = {
    'Adam': {
        'semantic_segmentation': {
            'lr': 1e-3,  
            'weight_decay': 1e-4,
        },
        'instance_segmentation': {
            'lr': 1e-4, 
            'weight_decay': 1e-4,
        }
    },
    'AdamW': {
        'semantic_segmentation': {
            'lr': 1e-4,  
            'weight_decay': 1e-4,
        },
        'instance_segmentation': {
            'lr': 1e-4,  
            'weight_decay': 1e-4,
        }
    },
    'SGD': {
        'semantic_segmentation': {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 1e-4,
        },
        'instance_segmentation': {
            'lr': 5e-3,  
            'momentum': 0.9,
            'weight_decay': 5e-4,
        }
    }
}


# =============================================================================
# SCHEDULER CONFIGURATIONS (PyTorch only)
# =============================================================================

SCHEDULER_CONFIGS = {
    'ReduceLROnPlateau': {
        'base_params': {
            'mode': 'min',
            'factor': 0.5,  # PyTorch recommended range 0.1-0.5
            'min_lr': 1e-7
        },
        'adaptive_patience': True  # Patience varies with dataset size
    },
    'StepLR': {
        'base_params': {
            'step_size': 10,
            'gamma': 0.1
        }
    },
    'OneCycleLR': {
        'base_params': {
            'pct_start': 0.3,
            'div_factor': 25.0,
            'final_div_factor': 10000.0,
            'anneal_strategy': 'cos'
        },
        'requires_total_steps': True,
        'warning': 'OneCycleLR may conflict with models using internal warmup.'
    },
    'CosineAnnealingLR': {
        'base_params': {
            'eta_min': 1e-6
        },
        'adaptive_t_max': True  # T_max = num_epochs
    },
    'None': {}  # No scheduler
}


# =============================================================================
# ADAPTIVE PATIENCE HEURISTIC (ReduceLROnPlateau)
# =============================================================================

def _compute_adaptive_patience(num_images: Optional[int]) -> int:
    """
    Compute patience for ReduceLROnPlateau based on dataset size.
    
    Heuristic validated for GeoAI small datasets:
    - Small datasets (< 50 images): patience = 2 (faster reaction)
    - Medium datasets (< 100 images): patience = 3
    - Large datasets (>= 100 images): patience = 5 (more stability)
    
    Args:
        num_images: Number of training images
        
    Returns:
        Patience value (2, 3, or 5)
    """
    if num_images is None:
        return 5  # Default for unknown dataset size
    
    if num_images < 50:
        return 2
    elif num_images < 100:
        return 3
    else:
        return 5


# =============================================================================
# CONFIGURATION GETTERS
# =============================================================================

def _normalize_optimizer_name(opt: str) -> str:
    """Normalize optimizer name: 'adam' → 'Adam'"""
    mapping = {
        'adam': 'Adam',
        'adamw': 'AdamW',
        'sgd': 'SGD'
    }
    return mapping.get(opt.lower(), opt)

def _normalize_scheduler_name(sched: str) -> str:
    """Normalize scheduler name: 'plateau' → 'ReduceLROnPlateau'"""
    mapping = {
        'plateau': 'ReduceLROnPlateau',
        'reducelronplateau': 'ReduceLROnPlateau',
        'step': 'StepLR',
        'steplr': 'StepLR',
        'onecycle': 'OneCycleLR',
        'onecyclelr': 'OneCycleLR',
        'cosine': 'CosineAnnealingLR',
        'cosineannealinglr': 'CosineAnnealingLR',
        'none': 'None'
    }
    return mapping.get(sched.lower(), sched) if sched else 'None'

def get_optimizer_config(
    optimizer_name: str,
    task: str,
    custom_lr: Optional[float] = None
) -> Dict[str, Any]:
    """
    Get optimizer configuration with default parameters.
    
    Parameters validated against:
    - OpenGeoAI library (semantic segmentation)
    - PyTorch Torchvision (Mask R-CNN)
    - NVIDIA NGC/TAO Toolkit (instance segmentation)
    
    Args:
        optimizer_name: Optimizer name ('Adam', 'AdamW', 'SGD')
        task: Task name ('Semantic Segmentation', 'Instance Segmentation')
        custom_lr: Optional custom learning rate (overrides default)
        
    Returns:
        Dictionary with optimizer parameters
        
    Raises:
        ValueError: If optimizer or task is unknown
    """
    # Normalize task name
    task_key = TASK_MAP.get(task)
    if task_key is None:
        task_key = task.lower().replace(' ', '_')

    # Normalize optimizer name (case-insensitive)
    optimizer_name = _normalize_optimizer_name(optimizer_name)
    
    # Get base config
    if optimizer_name not in OPTIMIZER_CONFIGS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    if task_key not in OPTIMIZER_CONFIGS[optimizer_name]:
        # Fallback to semantic segmentation
        task_key = 'semantic_segmentation'
    
    config = OPTIMIZER_CONFIGS[optimizer_name][task_key].copy()
    
    # Override LR if provided
    if custom_lr is not None:
        config['lr'] = custom_lr
        config['_lr_overridden'] = True 
    
    return config


def get_scheduler_config(
    scheduler_name: str,
    task: str,
    num_images: Optional[int] = None,
    num_epochs: int = 50,
    steps_per_epoch: int = 100
) -> Dict[str, Any]:
    """
    Get scheduler configuration with default parameters.
    
    Adapts parameters based on:
    - Dataset size (patience for ReduceLROnPlateau)
    - Number of epochs (T_max for CosineAnnealingLR)
    
    Parameters validated against PyTorch documentation and common practices.
    
    Args:
        scheduler_name: Scheduler name ('ReduceLROnPlateau', 'StepLR', etc.)
        task: Task name ('Semantic Segmentation', 'Instance Segmentation')
        num_images: Number of training images (for adaptive patience)
        num_epochs: Total number of epochs (for adaptive schedulers)
        steps_per_epoch: Steps per epoch (for OneCycleLR)
        
    Returns:
        Dictionary with scheduler parameters
        
    Raises:
        ValueError: If scheduler is unknown
        
    Note:
        - ReduceLROnPlateau requires scheduler.step(val_loss) in training loop
        - OneCycleLR requires total_steps calculation in create_scheduler()
    """
    # Normalize scheduler name
    scheduler_name = _normalize_scheduler_name(scheduler_name)
    
    if scheduler_name == 'None' or scheduler_name is None:
        return {}
    
    if scheduler_name not in SCHEDULER_CONFIGS:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    config_template = SCHEDULER_CONFIGS[scheduler_name]
    config = config_template['base_params'].copy()
    
    # =========================================================================
    # ADAPTIVE PATIENCE (ReduceLROnPlateau)
    # =========================================================================
    if config_template.get('adaptive_patience'):
        config['patience'] = _compute_adaptive_patience(num_images)
    
    # =========================================================================
    # ADAPTIVE T_MAX (CosineAnnealingLR)
    # =========================================================================
    if config_template.get('adaptive_t_max'):
        config['T_max'] = num_epochs
    
    # =========================================================================
    # ONECYCLELR WARNING (handled in UI, not here)
    # =========================================================================
    if config_template.get('warning'):
        config['_warning'] = config_template['warning']
    
    return config