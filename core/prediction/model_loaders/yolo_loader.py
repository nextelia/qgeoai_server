"""
YOLO Model Loader for QPredict

Loads YOLO11 models (detection, segmentation, OBB) from .qmtp packages.
Uses Ultralytics API to load pretrained weights.

Supported tasks:
- yolo_detect: Object detection with bounding boxes
- yolo_segment: Instance segmentation with masks
- yolo_obb: Oriented bounding boxes

Model files in .qmtp:
- model.pth: YOLO11 weights (not .pt)
- configuration.json: Model architecture and training config
- metadata.json: Training metrics and info
"""

import torch
import tempfile
from pathlib import Path
from typing import Optional

from .base_loader import BaseModelLoader

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None


class YOLOLoader(BaseModelLoader):
    """
    Load YOLO11 models from .qmtp packages.
    
    This loader handles all YOLO11 variants:
    - Detection (yolo_detect): Bounding boxes
    - Segmentation (yolo_segment): Instance masks
    - OBB (yolo_obb): Oriented bounding boxes
    
    Args:
        qmtp_loader: Instance of QMTPLoader with loaded model info
        device: Device to load model on ('cpu' or 'cuda')
        cuda_device_id: CUDA device ID if using CUDA
        
    Example:
        >>> from core.qmtp_loader import QMTPLoader
        >>> qmtp_loader = QMTPLoader("yolo_model.qmtp")
        >>> qmtp_loader.validate()
        >>> 
        >>> loader = YOLOLoader(qmtp_loader, device='cuda')
        >>> model = loader.load_model()
    """
    
    def __init__(self, qmtp_loader, device='cpu', cuda_device_id=0):
        """
        Initialize YOLO loader.
        
        Args:
            qmtp_loader: QMTPLoader instance
            device: 'cpu' or 'cuda'
            cuda_device_id: CUDA device ID (0-7)
        """
        super().__init__(qmtp_loader, device, cuda_device_id)
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is not installed.\n"
                "Install with: pip install ultralytics"
            )
        
        # Load model info
        self.model_info = qmtp_loader.load_info()
        self.config = self.model_info['configuration']
        
        # Validate YOLO task
        self.task = self.config['model'].get('task')
        valid_tasks = ['yolo_detect', 'yolo_segment', 'yolo_obb']
        if self.task not in valid_tasks:
            raise ValueError(
                f"Invalid YOLO task: {self.task}\n"
                f"Expected one of: {valid_tasks}"
            )
        
        # Extract YOLO parameters
        self.yolo_task = self.config['model']['yolo_task']  # 'detect', 'segment', 'obb'
        self.model_size = self.config['model']['model_size']  # 'n', 's', 'm', 'l', 'x'
        self.num_classes = self.config['model']['num_classes']
        self.image_size = self.config['model'].get('image_size', 640)

    def _create_model(self):
        """
        Create YOLO model architecture (required by BaseModelLoader).
        
        For YOLO, we load weights directly from .pth file using Ultralytics API,
        so this method is not used. We override load_model() instead.
        
        This method exists only to satisfy the abstract base class requirement.
        """
        raise NotImplementedError(
            "YOLO models are loaded directly from .pth using Ultralytics API.\n"
            "Use load_model() instead of _create_model()."
        )  
    
    def load_model(self) -> torch.nn.Module:
        """
        Load YOLO11 model from .qmtp package.
        
        Process:
        1. Extract model.pth from .qmtp archive to temp directory
        2. Load weights using Ultralytics YOLO API
        3. Move model to specified device
        4. Set model to evaluation mode
        5. Return model with metadata attached
        
        Returns:
            YOLO model ready for inference
            
        Raises:
            ImportError: If Ultralytics not installed
            FileNotFoundError: If model.pth not found in archive
            RuntimeError: If model loading fails
            
        Example:
            >>> loader = YOLOLoader(qmtp_loader, device='cuda')
            >>> model = loader.load_model()
            >>> # Model is ready for inference
            >>> results = model(image)
        """
        print(f"\n🔧 Loading YOLO11 model...")
        print(f"   Task: {self.task}")
        print(f"   Architecture: YOLO11{self.model_size}-{self.yolo_task}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Device: {self.device if self.device == 'cpu' else f'cuda:{self.cuda_device_id}'}")
        
        # Extract model weights to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"   Extracting weights from .qmtp...")
            
            # Extract model.pth
            weights_path = self.qmtp_loader.extract_model_weights(temp_dir)
            
            if not Path(weights_path).exists():
                raise FileNotFoundError(
                    f"Model weights not found after extraction: {weights_path}"
                )
            
            # Rename .pth to .pt for Ultralytics compatibility
            weights_path_pt = Path(temp_dir) / "model.pt"
            Path(weights_path).rename(weights_path_pt)
            
            print(f"   Loading YOLO11 weights...")
            
            # Load YOLO model using Ultralytics API
            try:
                model = YOLO(str(weights_path_pt))
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load YOLO model: {str(e)}\n"
                    f"Weights path: {weights_path}"
                )
        
        # Move model to device
        device_str = self.device if self.device == 'cpu' else f'cuda:{self.cuda_device_id}'
        print(f"   Moving model to {device_str}...")
        
        
        # Attach metadata for QPredict
        model._qpredict_task = self.task
        model._qpredict_yolo_task = self.yolo_task
        model._qpredict_num_classes = self.num_classes
        model._qpredict_image_size = self.image_size
        
        print(f"   ✅ YOLO11 model loaded successfully!")
        
        return model
    
    def get_model_summary(self) -> str:
        """
        Get human-readable summary of model architecture.
        
        Returns:
            Formatted string with model details
            
        Example:
            >>> loader = YOLOLoader(qmtp_loader)
            >>> print(loader.get_model_summary())
            YOLO11m-OBB
            Task: Oriented Bounding Boxes
            Classes: 1 (boat)
            Image Size: 512x512
            Device: CUDA (GPU 0)
        """
        task_names = {
            'yolo_detect': 'Object Detection',
            'yolo_segment': 'Instance Segmentation',
            'yolo_obb': 'Oriented Bounding Boxes'
        }
        
        task_display = task_names.get(self.task, self.task)
        
        # Get class names
        classes = self.config['classes']
        if len(classes) <= 3:
            class_names = ', '.join([c['name'] for c in classes])
        else:
            class_names = ', '.join([c['name'] for c in classes[:3]]) + f', ... (+{len(classes)-3})'
        
        summary = [
            f"YOLO11{self.model_size.upper()}-{self.yolo_task.upper()}",
            f"Task: {task_display}",
            f"Classes: {self.num_classes} ({class_names})",
            f"Image Size: {self.image_size}x{self.image_size}",
            f"Device: {self.device if self.device == 'cpu' else f'CUDA (GPU {self.cuda_device_id})'}"
        ]
        
        return "\n".join(summary)