"""
QMTP Loader for QPredict

This module handles loading and validation of .qmtp files (QModel Trainer Package).
It extracts metadata and configuration needed for inference without loading PyTorch weights.

"""

import json
import zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


class QMTPLoader:
    """
    Load and validate .qmtp model packages.
    
    This class handles:
    - Validation of .qmtp file structure
    - Extraction of metadata and configuration
    - Providing model information for UI display
    
    The actual PyTorch model loading is handled separately by ModelLoader.
    
    Args:
        qmtp_path: Path to .qmtp file
        
    Example:
        >>> loader = QMTPLoader("vegetation_model.qmtp")
        >>> is_valid, error = loader.validate()
        >>> if is_valid:
        ...     info = loader.load_info()
        ...     print(f"Task: {info['task']}")
        ...     print(f"Classes: {info['num_classes']}")
    """
    
    # Required files in .qmtp archive
    REQUIRED_FILES = ["model.pth", "metadata.json", "configuration.json"]
    
    def __init__(self, qmtp_path: str):
        """
        Initialize QMTP loader.
        
        Args:
            qmtp_path: Path to .qmtp file
        """
        self.qmtp_path = Path(qmtp_path)
        self._metadata = None
        self._configuration = None
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate .qmtp file structure.
        
        Checks:
        1. File exists and is readable
        2. File is a valid file
        3. Required files are present (model.pth, metadata.json, configuration.json)
        4. JSON files are valid and parseable
        5. Configuration has required keys
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if file is valid
            - error_message: None if valid, error description if invalid
            
        Example:
            >>> loader = QMTPLoader("model.qmtp")
            >>> valid, error = loader.validate()
            >>> if not valid:
            ...     print(f"Validation failed: {error}")
        """
        # Check file exists
        if not self.qmtp_path.exists():
            return False, f"File not found: {self.qmtp_path}"
        
        # Check file is readable
        if not self.qmtp_path.is_file():
            return False, f"Path is not a file: {self.qmtp_path}"
        
        # Check it's a valid ZIP
        try:
            with zipfile.ZipFile(self.qmtp_path, 'r') as zf:
                # Check for required files
                existing_files = zf.namelist()
                
                for required in self.REQUIRED_FILES:
                    if required not in existing_files:
                        return False, f"Missing required file: {required}"
                
                # Validate JSON structure
                try:
                    # Load and parse metadata.json
                    metadata_bytes = zf.read("metadata.json")
                    metadata = json.loads(metadata_bytes.decode('utf-8'))
                    
                    # Load and parse configuration.json
                    config_bytes = zf.read("configuration.json")
                    configuration = json.loads(config_bytes.decode('utf-8'))
                    
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON in QMTP file: {str(e)}"
                except UnicodeDecodeError as e:
                    return False, f"Invalid encoding in JSON files: {str(e)}"
                
                # Validate configuration structure
                required_config_keys = ["model", "input", "inference", "geospatial", "classes"]
                for key in required_config_keys:
                    if key not in configuration:
                        return False, f"Missing required key in configuration: {key}"
                
                # Validate model configuration (task-agnostic)
                # Common keys for all tasks
                required_model_keys = ["architecture", "in_channels", "num_classes"]
                for key in required_model_keys:
                    if key not in configuration["model"]:
                        return False, f"Missing required key in model configuration: {key}"
                
                # Task-specific validation
                task = configuration["model"].get("task", "semantic_segmentation")
                
                if task == "semantic_segmentation":
                    # Semantic segmentation requires 'encoder'
                    if "encoder" not in configuration["model"]:
                        return False, "Missing required key in model configuration: encoder"
                
                elif task == "instance_segmentation":
                    # Instance segmentation requires 'backbone'
                    if "backbone" not in configuration["model"]:
                        return False, "Missing required key in model configuration: backbone"
                
                elif task == "object_detection":
                    # Object detection requires 'backbone'
                    if "backbone" not in configuration["model"]:
                        return False, "Missing required key in model configuration: backbone"
                
                elif task.startswith("yolo_"):
                    # YOLO tasks require 'yolo_task', 'model_size', 'image_size'
                    yolo_required = ["yolo_task", "model_size", "image_size"]
                    for key in yolo_required:
                        if key not in configuration["model"]:
                            return False, f"Missing required key in model configuration: {key}"
                    
                    # Validate yolo_task value
                    valid_yolo_tasks = ["detect", "segment", "obb"]
                    yolo_task = configuration["model"]["yolo_task"]
                    if yolo_task not in valid_yolo_tasks:
                        return False, f"Invalid yolo_task: {yolo_task}. Must be one of {valid_yolo_tasks}"
                
                # Cache parsed data for later use
                self._metadata = metadata
                self._configuration = configuration
                
                return True, None
                
        except zipfile.BadZipFile:
            return False, f"File is not a valid ZIP archive: {self.qmtp_path}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def load_info(self) -> Dict[str, Any]:
        """
        Load model information from .qmtp file.
        
        This method extracts all information needed for:
        - UI display (task, architecture, classes, training metrics)
        - Model reconstruction (configuration)
        - Validation (expected input format)
        
        Must call validate() first, or this will validate automatically.
        
        Returns:
            Dictionary with complete model information:
            {
                'metadata': {...},              # Full metadata.json content
                'configuration': {...},         # Full configuration.json content
                'task': str,                    # Task type (e.g., 'Semantic Segmentation')
                'architecture': str,            # Architecture name (e.g., 'Unet')
                'encoder': str,                 # Encoder name (e.g., 'resnet34')
                'num_classes': int,             # Total number of classes (with background)
                'classes': list,                # List of class dicts with id, name, color
                'tile_size': int,               # Expected input tile size
                'in_channels': int,             # Number of input bands
                'crs': str,                     # CRS (e.g., 'EPSG:2154')
                'pixel_size_x': float,          # Pixel size X
                'pixel_size_y': float,          # Pixel size Y
                'normalization': dict,          # mean and std for normalization
                'training_metrics': dict        # Training summary (epochs, IoU, F1)
            }
            
        Raises:
            ValueError: If file is invalid or not yet validated
            
        Example:
            >>> loader = QMTPLoader("model.qmtp")
            >>> info = loader.load_info()
            >>> print(f"Model: {info['architecture']} + {info['encoder']}")
            >>> print(f"Classes: {[c['name'] for c in info['classes']]}")
        """
        # Validate if not already done
        if self._metadata is None or self._configuration is None:
            is_valid, error = self.validate()
            if not is_valid:
                raise ValueError(f"Invalid QMTP file: {error}")
        
        # Extract key information from configuration
        config = self._configuration
        metadata = self._metadata
        
        # Determine task from configuration
        task_type = config['model'].get('task', 'semantic_segmentation')
        
        # Map internal task names to display names
        task_display_names = {
            'semantic_segmentation': 'Semantic Segmentation',
            'instance_segmentation': 'Instance Segmentation',
            'object_detection': 'Object Detection',
            'yolo_detect': 'YOLO Detection',
            'yolo_segment': 'YOLO Segmentation',
            'yolo_obb': 'YOLO OBB'
        }
        task = task_display_names.get(task_type, 'Unknown Task')
        
        # Model info (common to all tasks)
        architecture = config['model']['architecture']
        num_classes = config['model']['num_classes']
        in_channels = config['model']['in_channels']
        
        # Encoder/Backbone (task-specific)
        # Semantic segmentation uses 'encoder', instance/detection use 'backbone'
        encoder = config['model'].get('encoder')  
        backbone = config['model'].get('backbone')  
        
        # Classes info
        classes = config['classes']
        
        # Input info
        tile_size = config['input']['tile_size']
        normalization = config['input']['normalization']
        
        # Geospatial info
        crs = config['geospatial']['crs']
        pixel_size_x = config['geospatial']['pixel_size_x']
        pixel_size_y = config['geospatial']['pixel_size_y']
        
        # Training metrics (may be empty)
        training_metrics = metadata.get('training_summary', {})
        
        return {
            # Raw data (for model reconstruction later)
            'metadata': metadata,
            'configuration': config,
            
            # Extracted info (for UI display and validation)
            'task': task,
            'task_type': task_type,  
            'architecture': architecture,
            'encoder': encoder,  
            'backbone': backbone,  
            'num_classes': num_classes,
            'classes': classes,
            'tile_size': tile_size,
            'in_channels': in_channels,
            'crs': crs,
            'pixel_size_x': pixel_size_x,
            'pixel_size_y': pixel_size_y,
            'normalization': normalization,
            'training_metrics': training_metrics
        }
    
    def extract_model_weights(self, output_path: str) -> str:
        """
        Extract model.pth from .qmtp archive to a temporary location.
        
        This is used by ModelLoader to load PyTorch weights.
        
        Args:
            output_path: Directory where to extract model.pth
            
        Returns:
            Path to extracted model.pth file
            
        Raises:
            ValueError: If QMTP file is invalid
            IOError: If extraction fails
            
        Example:
            >>> loader = QMTPLoader("model.qmtp")
            >>> weights_path = loader.extract_model_weights("/tmp")
            >>> # Now ModelLoader can load weights from weights_path
        """
        # Validate first
        if self._metadata is None or self._configuration is None:
            is_valid, error = self.validate()
            if not is_valid:
                raise ValueError(f"Invalid QMTP file: {error}")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_weights_path = output_dir / "model.pth"
        
        try:
            with zipfile.ZipFile(self.qmtp_path, 'r') as zf:
                # Extract model.pth
                zf.extract("model.pth", output_dir)
            
            return str(model_weights_path)
            
        except Exception as e:
            raise IOError(f"Failed to extract model weights: {str(e)}")
    
    def format_info_for_display(self) -> str:
        """
        Format model information for UI display.
        
        Returns a compact string with key model information suitable
        for display in the UI label.
        
        Returns:
            Formatted string with model info
            
        Example:
            >>> loader = QMTPLoader("model.qmtp")
            >>> print(loader.format_info_for_display())
            Task: Semantic Segmentation
            Architecture: Unet+resnet34
            Classes: 5 (Background, Grass, Tree, Road, Building)
        """
        try:
            info = self.load_info()
            
            # Format architecture string (encoder for semantic, backbone for instance/detection, standalone for YOLO)
            if info['encoder']:
                arch_str = f"{info['architecture']}+{info['encoder']}"
            elif info['backbone']:
                arch_str = f"{info['architecture']}+{info['backbone']}"
            else:
                # For YOLO, architecture already includes size (e.g., "YOLO11n")
                arch_str = info['architecture']
            
            # Format classes string (show first 3 if more than 4)
            class_names = [c['name'] for c in info['classes']]
            if len(class_names) <= 4:
                classes_str = ", ".join(class_names)
            else:
                classes_str = ", ".join(class_names[:3]) + f", ... (+{len(class_names)-3} more)"
            
            # Build info string
            lines = [
                f"Task: {info['task']}",
                f"Architecture: {arch_str}",
                f"Classes: {info['num_classes']} ({classes_str})"
            ]
            
            # Add training metrics if available (task-specific)
            if info['training_metrics']:
                metrics = info['training_metrics']
                
                # Semantic segmentation metrics
                if 'best_val_iou' in metrics:
                    lines.append(f"Best IoU: {metrics['best_val_iou']:.3f}")
                if 'best_val_f1' in metrics:
                    lines.append(f"Best F1: {metrics['best_val_f1']:.3f}")
                
                # Instance segmentation metrics
                if 'best_val_ap' in metrics:
                    lines.append(f"Best AP: {metrics['best_val_ap']:.3f}")
                if 'best_val_ar' in metrics:
                    lines.append(f"Best AR: {metrics['best_val_ar']:.3f}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error loading model info: {str(e)}"