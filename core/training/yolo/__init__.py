"""
YOLO training module for QModel Trainer

This module provides YOLO11 (Ultralytics) training capabilities for:
- Object Detection (bounding boxes)
- Instance Segmentation (polygons)
- Oriented Bounding Boxes (OBB)

Uses the Ultralytics YOLO API with custom callbacks for server integration.
"""

from .yolo_trainer import YOLODetectionTrainer, YOLOSegmentationTrainer, YOLOOBBTrainer
from .yolo_validator import validate_yolo_dataset

__all__ = [
    'YOLODetectionTrainer',
    'YOLOSegmentationTrainer', 
    'YOLOOBBTrainer',
    'validate_yolo_dataset'
]