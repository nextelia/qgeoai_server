"""
QPredict Endpoints

FastAPI endpoints for QPredict plugin:
- Model loading from .qmtp files
- Inference on tiles
- Model management

These endpoints handle the ML/PyTorch parts that were moved from the plugin.
"""

from .models import router as models_router
from .predict import router as predict_router

__all__ = ['models_router', 'predict_router']