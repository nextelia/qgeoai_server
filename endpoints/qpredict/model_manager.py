"""
Model Manager for QPredict

Manages loaded PyTorch models in memory:
- Load models from .qmtp files
- Keep models in memory with UUID
- Unload models to free memory
- Track model metadata
"""

import uuid
import torch
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
import tempfile
import shutil

# ABSOLUTE IMPORTS
from core.prediction import ModelLoader


class LoadedModel:
    """Container for a loaded model with metadata."""
    
    def __init__(
        self,
        model_id: str,
        model: torch.nn.Module,
        loader: ModelLoader,
        qmtp_path: Path,
        device: str
    ):
        self.model_id = model_id
        self.model = model
        self.loader = loader
        self.qmtp_path = qmtp_path 
        self.device = device
        self.loaded_at = datetime.now()
        
        # Extract metadata from loader
        self.task = loader.config['model'].get('task', 'semantic_segmentation')
        self.architecture = loader.config['model'].get('architecture', 'unknown')
        self.num_classes = loader.config['model'].get('num_classes', 0)
        self.classes = loader.config.get('classes', [])
        self.input_channels = loader.config['model'].get('in_channels', 3)

        # Extract divisibility from model (set by semantic_loader)
        self.divisibility = getattr(model, '_qpredict_divisibility', 1)
    
    def get_info(self) -> Dict:
        """Get model information as dict."""
        return {
            'model_id': self.model_id,
            'task': self.task,
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'device': self.device,
            'input_channels': self.input_channels,
            'divisibility': self.divisibility,
            'loaded_at': self.loaded_at
        }
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.qmtp_path and self.qmtp_path.exists():
            try:
                self.qmtp_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete temp file {self.qmtp_path}: {e}")


class ModelManager:
    """
    Singleton manager for loaded models.
    
    Keeps models in memory and provides access by UUID.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.models: Dict[str, LoadedModel] = {}
        self._initialized = True
    
    def load_model(
        self,
        qmtp_file_content: bytes,
        device: str = 'cpu',
        cuda_device_id: int = 0
    ) -> LoadedModel:
        """
        Load a model from .qmtp file content.
        
        Args:
            qmtp_file_content: Raw bytes of .qmtp file
            device: 'cpu' or 'cuda'
            cuda_device_id: CUDA device ID if using GPU
            
        Returns:
            LoadedModel instance
            
        Raises:
            ValueError: If model loading fails
        """
        # Generate unique ID
        model_id = str(uuid.uuid4())
        
        # Save to temporary file
        temp_dir = Path(tempfile.gettempdir()) / 'qgeoai_server' / 'qmtp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_path = temp_dir / f"{model_id}.qmtp"
        
        try:
            # Write file
            with open(temp_path, 'wb') as f:
                f.write(qmtp_file_content)
            
            # Import QMTPLoader (need to add this to server)
            from core.prediction.qmtp_loader import QMTPLoader
            
            # Load and validate
            qmtp_loader = QMTPLoader(str(temp_path))
            qmtp_loader.validate()
            
            # Create model loader
            model_loader = ModelLoader(
                qmtp_loader=qmtp_loader,
                device=device,
                cuda_device_id=cuda_device_id
            )
            
            # Load model
            model = model_loader.load_model()
            
            # Create loaded model container
            loaded_model = LoadedModel(
                model_id=model_id,
                model=model,
                loader=model_loader,
                qmtp_path=temp_path,
                device=model_loader.get_device_info()
            )
            
            # Store
            self.models[model_id] = loaded_model
            
            print(f"✅ Loaded model {model_id}: {loaded_model.architecture} ({loaded_model.task})")
            
            return loaded_model
            
        except Exception as e:
            # Cleanup on failure
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def get_model(self, model_id: str) -> Optional[LoadedModel]:
        """Get a loaded model by ID."""
        return self.models.get(model_id)
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model and free memory.
        
        Args:
            model_id: Model UUID
            
        Returns:
            True if model was unloaded, False if not found
        """
        loaded_model = self.models.pop(model_id, None)
        
        if loaded_model is None:
            return False
        
        # Cleanup
        loaded_model.cleanup()
        
        # Force garbage collection
        del loaded_model.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ Unloaded model {model_id}")
        
        return True
    
    def list_models(self) -> list:
        """Get list of all loaded models."""
        return [
            {
                'model_id': m.model_id,
                'task': m.task,
                'architecture': m.architecture,
                'device': m.device,
                'loaded_at': m.loaded_at
            }
            for m in self.models.values()
        ]
    
    def cleanup_all(self):
        """Unload all models and cleanup."""
        for model_id in list(self.models.keys()):
            self.unload_model(model_id)


# Singleton instance
model_manager = ModelManager()