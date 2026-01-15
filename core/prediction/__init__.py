"""
QPredict Core Prediction Module

This module contains pure ML/PyTorch code for QPredict predictions:
- Model loading (PyTorch, segmentation_models_pytorch, torchvision, ultralytics)
- Inference execution (semantic, instance, YOLO)
- Reliability calculations (entropy, max probability, variance)

Modules:
- model_loader: High-level wrapper for model loading
- model_loaders/: Task-specific model loaders (factory pattern)
- predictors/: Task-specific predictors for inference (factory pattern)
- reliability_maps: Reliability/uncertainty calculations
"""

from core.prediction.model_loader import ModelLoader
from core.prediction.reliability_maps import ReliabilityMapCalculator

__all__ = [
    'ModelLoader',
    'ReliabilityMapCalculator'
]