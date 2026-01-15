"""
QMTP (QModel Trainer Package) Exporter

Exports trained models to .qmtp format

The .qmtp format is designed to be self-contained and portable,
allowing trained models to be easily shared and deployed for inference.

"""

import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Literal


class QMTPExporter:
    """
    Export trained models to .qmtp format.
    
    This class handles the creation of .qmtp files containing
    everything needed for inference: model weights, configuration, and metadata.
    
    Supports Semantic Segmentation (SMP), Instance Segmentation (Mask R-CNN) and YOLO11.
    
    Args:
        output_dir: Directory where to save the .qmtp file
        model_name: Name for the model (will be sanitized for filename)
    """
    
    QMTP_VERSION = "1.0"
    PLUGIN_VERSION = "1.0.0"
    
    def __init__(self, output_dir: str, model_name: str):
        """
        Initialize the exporter.
        
        Args:
            output_dir: Directory where to save the .qmtp
            model_name: Name without spaces (e.g., 'model_name')
        """
        self.output_dir = Path(output_dir)
        # Sanitize model name (remove spaces, dots, special chars)
        self.model_name = model_name.replace(" ", "_").replace(".", "_")
        self.qmtp_path = self.output_dir / f"{self.model_name}.qmtp"
    
    def export(
        self,
        model_path: str,
        dataset_path: str,
        task: str,  # "Semantic Segmentation", "Instance Segmentation", "YOLO Detection", "YOLO Segmentation", "YOLO OBB"
        architecture: str,
        # PyTorch models (UNet, Mask R-CNN)
        encoder: Optional[str] = None,
        backbone: Optional[str] = None,
        encoder_weights: Optional[str] = None,
        use_pretrained: bool = True,
        # YOLO-specific parameters
        model_size: Optional[str] = None,  # 'n', 's', 'm', 'l', 'x'
        image_size: Optional[int] = None,  # Training image size
        # Inference parameters
        tile_size: int = 512,
        stride: Optional[int] = None,
        padding: int = 64,
        # Metrics for Semantic Segmentation
        best_iou: Optional[float] = None,
        best_f1: Optional[float] = None,
        # Metrics for Instance Segmentation
        best_ap: Optional[float] = None,
        best_ap50: Optional[float] = None,
        best_ap75: Optional[float] = None,
        best_ar: Optional[float] = None,
        # Metrics for YOLO (all tasks)
        best_map50_95: Optional[float] = None,
        best_map50: Optional[float] = None,
        best_precision: Optional[float] = None,
        best_recall: Optional[float] = None,
        # Common
        total_epochs: Optional[int] = None
    ) -> Path:
        """
        Create .qmtp package with all necessary information.
        
        Args:
            model_path: Path to saved best_model_weights.pth
            dataset_path: Path to QAnnotate dataset (to read metadata)
            task: Task type ("Semantic Segmentation" or "Instance Segmentation")
            architecture: Model architecture (e.g., 'UNet', 'Mask R-CNN')
            encoder: Encoder backbone for UNet (e.g., 'resnet50')
            backbone: Backbone for Mask R-CNN (e.g., 'resnet50_fpn')
            encoder_weights: Encoder weights ('imagenet' or None)
            use_pretrained: Whether pretrained weights were used
            tile_size: Input tile size used during training
            stride: Stride for inference (default: tile_size // 2)
            padding: Padding for inference tiles
            best_iou: Best validation IoU (semantic seg)
            best_f1: Best validation F1 (semantic seg)
            best_ap: Best validation AP (instance seg)
            best_ap50: Best validation AP50 (instance seg)
            best_ap75: Best validation AP75 (instance seg)
            best_ar: Best validation AR (instance seg)
            total_epochs: Total number of epochs trained
            
        Returns:
            Path to created .qmtp file
            
        Raises:
            FileNotFoundError: If qannotate_metadata.json not found
            ValueError: If task is not supported
            Exception: If any error occurs during export
        """
        
        # Validate task
        valid_tasks = ['Semantic Segmentation', 'Instance Segmentation', 'YOLO Detection', 'YOLO Segmentation', 'YOLO OBB']
        if task not in valid_tasks:
            raise ValueError(f"Unsupported task: {task}. Must be one of {valid_tasks}.")
        
        # Default stride = 50% overlap
        if stride is None:
            stride = tile_size // 2

        # Load metadata based on task
        is_yolo = task.startswith('YOLO')

        # CRITICAL: For YOLO, we need BOTH data.yaml AND qannotate_metadata.json
        # - data.yaml: class names (YOLO format)
        # - qannotate_metadata.json: geospatial info (CRS, resolution, tiling strategy)
        if is_yolo:
            # Load YOLO class names from data.yaml
            yolo_metadata = self._load_yolo_metadata(dataset_path)
            # Load geospatial info from qannotate_metadata.json
            try:
                qannotate_metadata = self._load_qannotate_metadata(dataset_path)
            except FileNotFoundError:
                # Fallback: YOLO dataset without QAnnotate metadata
                qannotate_metadata = None
            metadata = {'yolo': yolo_metadata, 'qannotate': qannotate_metadata}
        else:
            # PyTorch: Load QAnnotate metadata only
            metadata = self._load_qannotate_metadata(dataset_path)
        
        # Extract information from metadata
        if is_yolo:
            # YOLO metadata (from data.yaml)
            tile_size_from_metadata = image_size if image_size else 640
            stride_from_metadata = tile_size // 2  # Default 50% overlap
            
            # Geospatial info: read from qannotate_metadata.json if available
            if metadata['qannotate'] is not None:
                # YOLO dataset exported from QAnnotate - use original geospatial info
                qmeta = metadata['qannotate']
                crs = qmeta['raster_info']['crs']['authid']
                pixel_size_x = qmeta['raster_info']['pixel_size']['x']
                pixel_size_y = qmeta['raster_info']['pixel_size']['y']
                num_bands = qmeta['raster_info']['dimensions']['bands']
                
                # Also use tiling strategy from QAnnotate if available
                if qmeta['tiling_strategy']['enabled']:
                    tile_size_from_metadata = qmeta['tiling_strategy']['tile_size']
                    stride_from_metadata = qmeta['tiling_strategy']['stride']
                    if stride == tile_size // 2:  # Default value, update it
                        stride = stride_from_metadata
            else:
                # YOLO dataset from another source - use defaults
                crs = "EPSG:3857"  # Default Web Mercator
                pixel_size_x = 1.0
                pixel_size_y = 1.0
                num_bands = 3  # YOLO always uses RGB
            
        else:
            # PyTorch metadata (from QAnnotate)
            tile_size_from_metadata = metadata['tiling_strategy']['tile_size']
            
            # Use stride from metadata if tiling was enabled
            if metadata['tiling_strategy']['enabled']:
                stride_from_metadata = metadata['tiling_strategy']['stride']
                if stride == tile_size // 2:  # Default value
                    stride = stride_from_metadata
            
            # Geospatial information
            crs = metadata['raster_info']['crs']['authid']
            pixel_size_x = metadata['raster_info']['pixel_size']['x']
            pixel_size_y = metadata['raster_info']['pixel_size']['y']
            
            num_bands = metadata['raster_info']['dimensions']['bands']
        
        # Classes information
        if is_yolo:
            # YOLO: classes from data.yaml
            yolo_data = metadata['yolo']
            num_classes_without_bg = len(yolo_data['names'])
            yolo_classes = yolo_data['names']  # Dict {0: 'class1', 1: 'class2', ...}
            
            # Build classes list (YOLO has no explicit background class in data.yaml)
            classes = []
            
            # Try to get colors from qannotate_metadata.json if available
            if metadata['qannotate'] is not None and 'class_catalog' in metadata['qannotate']:
                # Use colors from QAnnotate metadata (matches classes by name)
                qannotate_classes = metadata['qannotate']['class_catalog']['classes']
                
                # Create a mapping: class_name -> color
                color_map = {}
                for qcls in qannotate_classes:
                    color_map[qcls['label']] = qcls['color_rgb']
                
                # Build classes list with colors
                for class_id, class_name in yolo_classes.items():
                    # Get color from QAnnotate metadata, fallback to red
                    color = color_map.get(class_name, [255, 0, 0])
                    classes.append({
                        "id": class_id,
                        "name": class_name,
                        "color": color
                    })
            else:
                # Fallback: YOLO dataset without QAnnotate metadata
                for class_id, class_name in yolo_classes.items():
                    classes.append({
                        "id": class_id,
                        "name": class_name,
                        "color": [255, 0, 0]  # Default red (no color info in YOLO)
                    })
            
            num_classes_total = len(classes)
            
        else:
            # PyTorch: classes from QAnnotate
            num_classes_without_bg = metadata['class_catalog']['num_classes']
            qannotate_classes = metadata['class_catalog']['classes']
            
            # Build classes list with background class 0
            classes = [
                {"id": 0, "name": "Background", "color": [0, 0, 0]}
            ]
            for idx, cls in enumerate(qannotate_classes, start=1):
                classes.append({
                    "id": idx,
                    "name": cls['label'],
                    "color": cls['color_rgb']
                })
            
            num_classes_total = num_classes_without_bg + 1
        
        # Create ZIP archive
        with zipfile.ZipFile(self.qmtp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            
            # ================================================================
            # 1. MODEL WEIGHTS
            # ================================================================
            zf.write(model_path, "model.pth")
            
            # ================================================================
            # 2. METADATA (informative only)
            # ================================================================
            export_metadata = {
                "model_name": self.model_name,
                "qmtp_version": self.QMTP_VERSION,
                "created_with": f"QModel Trainer v{self.PLUGIN_VERSION}",
                "creation_date": datetime.now().isoformat(),
                "task": task,
                "training_summary": {}
            }

            # Add training metrics based on task
            if total_epochs is not None:
                export_metadata["training_summary"]["total_epochs"] = total_epochs

            if 'Semantic' in task:
                # Semantic Segmentation metrics
                if best_iou is not None:
                    export_metadata["training_summary"]["best_val_iou"] = round(best_iou, 3)
                if best_f1 is not None:
                    export_metadata["training_summary"]["best_val_f1"] = round(best_f1, 3)
                    
            elif 'Instance' in task:
                # Instance Segmentation metrics
                if best_ap is not None:
                    export_metadata["training_summary"]["best_val_ap"] = round(best_ap, 3)
                if best_ap50 is not None:
                    export_metadata["training_summary"]["best_val_ap50"] = round(best_ap50, 3)
                if best_ap75 is not None:
                    export_metadata["training_summary"]["best_val_ap75"] = round(best_ap75, 3)
                if best_ar is not None:
                    export_metadata["training_summary"]["best_val_ar"] = round(best_ar, 3)
                    
            elif 'YOLO' in task:
                # YOLO metrics (all tasks) ← NEW
                if best_map50_95 is not None:
                    export_metadata["training_summary"]["best_val_map50_95"] = round(best_map50_95, 3)
                if best_map50 is not None:
                    export_metadata["training_summary"]["best_val_map50"] = round(best_map50, 3)
                if best_precision is not None:
                    export_metadata["training_summary"]["best_val_precision"] = round(best_precision, 3)
                if best_recall is not None:
                    export_metadata["training_summary"]["best_val_recall"] = round(best_recall, 3)

            zf.writestr("metadata.json", json.dumps(export_metadata, indent=2))
            
            # ================================================================
            # 3. CONFIGURATION (critical for inference)
            # ================================================================
            
            # Determine normalization stats
            if is_yolo:
                # YOLO does internal normalization - use passthrough
                norm_mean = [0.0, 0.0, 0.0]
                norm_std = [1.0, 1.0, 1.0]
            elif use_pretrained:
                # ImageNet normalization (PyTorch pretrained)
                norm_mean = [0.485, 0.456, 0.406]
                norm_std = [0.229, 0.224, 0.225]
            else:
                # No normalization
                norm_mean = [0.0, 0.0, 0.0]
                norm_std = [1.0, 1.0, 1.0]
            
            # Band names (for MVP: RGB only, future: multispectral)
            if num_bands == 3:
                bands = ["red", "green", "blue"]
            else:
                # Future: support for multispectral
                bands = [f"band_{i+1}" for i in range(num_bands)]
            
            # Build model configuration based on task
            if 'Semantic' in task:
                # Semantic Segmentation (UNet)
                model_config = {
                    "task": "semantic_segmentation",
                    "architecture": architecture,
                    "encoder": encoder,
                    "encoder_weights": encoder_weights if encoder_weights else "random",
                    "in_channels": num_bands,
                    "num_classes": num_classes_total
                }
                
            elif 'Instance' in task:
                # Instance Segmentation (Mask R-CNN)
                model_config = {
                    "task": "instance_segmentation",
                    "architecture": architecture,
                    "backbone": backbone,
                    "pretrained_weights": "coco" if use_pretrained else "random",
                    "in_channels": num_bands,
                    "num_classes": num_classes_total
                }
                
            elif 'YOLO' in task:
                # YOLO (Detection / Segmentation / OBB)
                if 'Detection' in task:
                    yolo_task = 'detect'
                elif 'Segmentation' in task:
                    yolo_task = 'segment'
                elif 'OBB' in task:
                    yolo_task = 'obb'
                else:
                    raise ValueError(f"Unknown YOLO task: {task}")
                
                model_config = {
                    "task": f"yolo_{yolo_task}",
                    "architecture": f"YOLO11{model_size}",
                    "yolo_task": yolo_task,
                    "model_size": model_size,
                    "pretrained_weights": "coco" if use_pretrained else "random",
                    "image_size": image_size if image_size else 640,
                    "in_channels": 3,
                    "num_classes": num_classes_total
                }
                
            else:
                raise ValueError(f"Unsupported task: {task}")

            configuration = {
                "model": model_config,
                "input": {
                    "bands": bands,
                    "tile_size": tile_size,
                    "normalization": {
                        "mean": norm_mean,
                        "std": norm_std
                    }
                },
                "inference": {
                    "stride": stride,
                    "padding": padding
                },
                "geospatial": {
                    "crs": crs,
                    "pixel_size_x": pixel_size_x,
                    "pixel_size_y": pixel_size_y
                },
                "classes": classes
            }
            
            zf.writestr("configuration.json", json.dumps(configuration, indent=2))
        
        return self.qmtp_path
    
    def _load_qannotate_metadata(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load qannotate_metadata.json from dataset directory.
        
        This metadata file contains all information about the dataset:
        - Classes and their colors
        - CRS and geospatial information
        - Tiling strategy
        - Image format and dimensions
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary with complete metadata
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file is not valid JSON
        """
        metadata_path = Path(dataset_path) / 'qannotate_metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"qannotate_metadata.json not found in {dataset_path}\n"
                f"Expected path: {metadata_path}\n"
                f"Make sure this is a valid QAnnotate export directory."
            )
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in qannotate_metadata.json: {str(e)}",
                e.doc,
                e.pos
            )
        
        return metadata
    
    def _load_yolo_metadata(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load data.yaml from YOLO dataset directory.
        
        Args:
            dataset_path: Path to YOLO dataset directory
            
        Returns:
            Dictionary with YOLO metadata (simplified structure)
            
        Raises:
            FileNotFoundError: If data.yaml not found
        """
        import yaml
        
        yaml_path = Path(dataset_path) / 'data.yaml'
        
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"data.yaml not found in {dataset_path}\n"
                f"Expected path: {yaml_path}\n"
                f"Make sure this is a valid YOLO dataset directory."
            )
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load data.yaml: {str(e)}")
        
        # Validate required fields
        if 'names' not in metadata:
            raise ValueError("data.yaml must contain 'names' field with class names")
        
        return metadata

    @staticmethod
    def validate_qmtp(qmtp_path: str) -> bool:
        """
        Validate .qmtp file structure.
        
        This method checks that a .qmtp file has the correct structure
        and all required files. Useful for QPredict to verify files before loading.
        
        Args:
            qmtp_path: Path to .qmtp file to validate
            
        Returns:
            True if valid .qmtp file, False otherwise
            
        Example:
            >>> if QMTPExporter.validate_qmtp('model.qmtp'):
            ...     print("Valid QMTP file!")
        """
        try:
            with zipfile.ZipFile(qmtp_path, 'r') as zf:
                # Check required files exist
                required_files = ["model.pth", "metadata.json", "configuration.json"]
                existing_files = zf.namelist()
                
                for required in required_files:
                    if required not in existing_files:
                        print(f"Missing required file: {required}")
                        return False
                
                # Validate JSON structure
                try:
                    metadata = json.loads(zf.read("metadata.json"))
                    config = json.loads(zf.read("configuration.json"))
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON in QMTP file: {e}")
                    return False
                
                # Check critical keys in configuration
                required_keys = ["model", "input", "inference", "geospatial", "classes"]
                for key in required_keys:
                    if key not in config:
                        print(f"Missing required key in configuration: {key}")
                        return False
                
                # Check model configuration (must have task)
                if "task" not in config["model"]:
                    print("Missing 'task' in model configuration")
                    return False
                
                task = config["model"]["task"]
                
                # Task-specific validation
                if task == "semantic_segmentation":
                    required_model_keys = ["architecture", "encoder", "in_channels", "num_classes"]
                elif task == "instance_segmentation":
                    required_model_keys = ["architecture", "backbone", "in_channels", "num_classes"]
                elif task.startswith("yolo_"):
                    required_model_keys = ["architecture", "yolo_task", "model_size", "num_classes"]
                else:
                    print(f"Unknown task type: {task}")
                    return False
                
                for key in required_model_keys:
                    if key not in config["model"]:
                        print(f"Missing required key in model configuration: {key}")
                        return False
                
                return True
                
        except zipfile.BadZipFile:
            print(f"File is not a valid ZIP archive: {qmtp_path}")
            return False
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    @staticmethod
    def load_qmtp_info(qmtp_path: str) -> Dict[str, Any]:
        """
        Load and return information from a .qmtp file without loading the model.
        
        Useful for displaying model information in UI before loading.
        
        Args:
            qmtp_path: Path to .qmtp file
            
        Returns:
            Dictionary with metadata and configuration
            
        Raises:
            Exception: If file cannot be read or is invalid
            
        Example:
            >>> info = QMTPExporter.load_qmtp_info('model.qmtp')
            >>> print(f"Model: {info['metadata']['model_name']}")
            >>> print(f"Task: {info['metadata']['task']}")
            >>> print(f"Classes: {len(info['configuration']['classes'])}")
        """
        try:
            with zipfile.ZipFile(qmtp_path, 'r') as zf:
                metadata = json.loads(zf.read("metadata.json"))
                configuration = json.loads(zf.read("configuration.json"))
                
                return {
                    "metadata": metadata,
                    "configuration": configuration
                }
        except Exception as e:
            raise Exception(f"Failed to load QMTP info: {str(e)}")