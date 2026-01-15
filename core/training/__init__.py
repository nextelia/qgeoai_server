"""
Core training module for QModelTrainer

This module contains all ML-related training logic:
- Model creation and loading
- Training loops (PyTorch and YOLO)
- Metrics computation
- Visualization
- Report generation
"""

from .trainer import (
    SemanticSegmentationTrainer,
    InstanceSegmentationTrainer,
    create_optimizer,
    create_scheduler
)

from .model_factory import (
    create_model,
    get_model_info,
    validate_model_config
)

from .training_config import (
    get_optimizer_config,
    get_scheduler_config
)

from .checkpoint_loader import (
    load_checkpoint,
    save_checkpoint
)

from .lr_finder import LRFinder

from .qmtp_exporter import QMTPExporter

from .dataset_loader import (
    SemanticSegmentationDataset,
    InstanceSegmentationDataset,
    create_dataloaders,
    get_training_transforms,
    get_validation_transforms,
    load_mask_dataset,
    load_coco_dataset,
    collate_fn_instance
)

__all__ = [
    "SemanticSegmentationTrainer",
    "InstanceSegmentationTrainer",
    "create_optimizer",
    "create_scheduler",
    "create_model",
    "get_model_info",
    "validate_model_config",
    "get_optimizer_config",
    "get_scheduler_config",
    "load_checkpoint",
    "save_checkpoint",
    "LRFinder",
    "QMTPExporter",
    "SemanticSegmentationDataset",
    "InstanceSegmentationDataset",
    "create_dataloaders",
    "get_training_transforms",
    "get_validation_transforms",
    "load_mask_dataset",
    "load_coco_dataset",
    "collate_fn_instance"
]