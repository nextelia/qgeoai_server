"""
Input validation utilities for QModel Trainer

This module provides validation functions for all user inputs before training:
- Dataset directory validation (QAnnotate format)
- Training configuration validation
- Device availability checks

All validators return (is_valid: bool, error_message: str or None)
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_valid_image_file(filename: str) -> bool:
    """
    Check if a file is a valid image file (not temporary or hidden).
    
    QGIS and other GIS software create temporary files that should be ignored:
    - .tif.aux.xml (GDAL auxiliary files)
    - .xml (metadata files)
    - files starting with . (hidden files)
    - files ending with ~ (backup files)
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file should be processed, False if it should be ignored
    """
    filename_lower = filename.lower()
    
    # Ignore hidden files (starting with .)
    if filename.startswith('.'):
        return False
    
    # Ignore backup files (ending with ~)
    if filename.endswith('~'):
        return False
    
    # Ignore GDAL auxiliary files
    if filename_lower.endswith('.aux.xml'):
        return False
    
    # Ignore XML metadata files
    if filename_lower.endswith('.xml'):
        return False
    
    # Ignore other common temporary extensions
    temp_extensions = ['.tmp', '.temp', '.bak', '.lock']
    if any(filename_lower.endswith(ext) for ext in temp_extensions):
        return False
    
    # Accept only valid image extensions
    valid_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    if not any(filename_lower.endswith(ext) for ext in valid_extensions):
        return False
    
    return True

# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset_path(dataset_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that dataset path exists and is a directory.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid
    """
    if not dataset_path:
        return False, "Dataset path is empty"
    
    if not os.path.exists(dataset_path):
        return False, f"Dataset path does not exist: {dataset_path}"
    
    if not os.path.isdir(dataset_path):
        return False, f"Dataset path is not a directory: {dataset_path}"
    
    return True, None


def validate_qannotate_metadata(dataset_path: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate QAnnotate metadata file exists and is valid.
    
    This function checks for the presence of qannotate_metadata.json
    and validates its structure for training compatibility.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (is_valid, error_message, metadata_dict)
        - (True, None, metadata) if valid
        - (False, error_message, None) if invalid
    """
    metadata_path = Path(dataset_path) / 'qannotate_metadata.json'
    
    if not metadata_path.exists():
        return False, (
            "QAnnotate metadata file not found.\n"
            "Expected: qannotate_metadata.json\n"
            "This file should be created by QAnnotate export."
        ), None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in metadata file: {str(e)}", None
    except Exception as e:
        return False, f"Error reading metadata file: {str(e)}", None
    
    # Validate metadata structure
    required_fields = ['export_info', 'class_catalog', 'statistics']
    missing_fields = [field for field in required_fields if field not in metadata]
    
    if missing_fields:
        return False, (
            f"Metadata file is missing required fields: {', '.join(missing_fields)}\n"
            "The metadata file may be corrupted or from an older version of QAnnotate."
        ), None
    
    # Check export format
    export_format = metadata.get('export_info', {}).get('export_format')
    if not export_format:
        return False, "Export format not specified in metadata", None
    
    return True, None, metadata


def validate_dataset_format(metadata: Dict, allowed_formats: list = None) -> Tuple[bool, Optional[str]]:
    """
    Validate that dataset format is supported for training.
    
    Args:
        metadata: Parsed metadata dictionary
        allowed_formats: List of allowed formats (default: all supported)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if allowed_formats is None:
        allowed_formats = ['mask', 'coco', 'yolo11-detect', 'yolo11-seg', 'yolo11-obb']
    
    export_format = metadata.get('export_info', {}).get('export_format', '').lower()
    
    if export_format not in allowed_formats:
        return False, (
            f"Dataset format not supported.\n"
            f"Found: {export_format}\n"
            f"Supported formats: {', '.join(allowed_formats)}\n"
            f"Please export dataset in a supported format from QAnnotate."
        )
    
    return True, None


def validate_dataset_structure(dataset_path: str, metadata: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate that dataset has the expected directory structure.
    
    For mask format, expects:
    - images/ directory
    - masks/ directory
    
    Args:
        dataset_path: Path to dataset directory
        metadata: Parsed metadata dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    dataset_path = Path(dataset_path)
    export_format = metadata.get('export_info', {}).get('export_format', '').lower()
    
    if export_format == 'mask':
        # Check for images directory
        images_dir = dataset_path / 'images'
        if not images_dir.exists():
            return False, "Images directory not found: images/"
        
        if not images_dir.is_dir():
            return False, "images/ is not a directory"
        
        # Check for masks directory
        masks_dir = dataset_path / 'masks'
        if not masks_dir.exists():
            return False, "Masks directory not found: masks/"
        
        if not masks_dir.is_dir():
            return False, "masks/ is not a directory"
        
        # Get valid image/mask files (filtering out temporary files)
        image_files = [f for f in images_dir.iterdir() if is_valid_image_file(f.name)]
        mask_files = [f for f in masks_dir.iterdir() if is_valid_image_file(f.name)]
        
        if not image_files:
            return False, "Images directory is empty (no valid image files found)"
        
        if not mask_files:
            return False, "Masks directory is empty (no valid mask files found)"
        
        # Check that number of images matches number of masks
        if len(image_files) != len(mask_files):
            return False, (
                f"Number of images ({len(image_files)}) does not match "
                f"number of masks ({len(mask_files)}). "
                f"Note: Temporary files (.aux.xml, .xml, hidden files) are ignored."
            )
    
    elif export_format == 'coco':
        # Check for images directory
        images_dir = dataset_path / 'images'
        if not images_dir.exists():
            return False, "Images directory not found: images/"
        
        if not images_dir.is_dir():
            return False, "images/ is not a directory"
        
        # Check for annotations.json file
        annotations_file = dataset_path / 'annotations.json'
        if not annotations_file.exists():
            return False, "COCO annotations file not found: annotations.json"
        
        # Validate annotations.json structure
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Check required COCO fields
            required_fields = ['images', 'annotations', 'categories']
            missing_fields = [field for field in required_fields if field not in coco_data]
            
            if missing_fields:
                return False, f"COCO annotations missing required fields: {', '.join(missing_fields)}"
            
            # Check that we have data
            if not coco_data['images']:
                return False, "COCO annotations contains no images"
            
            if not coco_data['categories']:
                return False, "COCO annotations contains no categories"
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in annotations.json: {str(e)}"
        except Exception as e:
            return False, f"Error reading annotations.json: {str(e)}"
    
    elif export_format in ['yolo11-detect', 'yolo11-seg', 'yolo11-obb']:
        # Import YOLO validator
        from core.training.yolo.yolo_validator import validate_yolo_dataset
        
        # Validate YOLO dataset structure
        valid, error = validate_yolo_dataset(dataset_path, metadata)
        if not valid:
            return False, f"YOLO dataset validation failed: {error}"
    
    else:
        return False, f"Unknown export format: {export_format}"
    
    return True, None


def validate_classes(metadata: Dict) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Validate class catalog in metadata.
    
    Args:
        metadata: Parsed metadata dictionary
        
    Returns:
        Tuple of (is_valid, error_message, num_classes)
        - For semantic segmentation (mask): num_classes includes background (0)
        - For instance segmentation (coco): num_classes includes background (0)
    """
    class_catalog = metadata.get('class_catalog', {})
    classes = class_catalog.get('classes', [])
    
    if not classes:
        return False, "No classes defined in dataset", None
    
    # Check for duplicate class IDs
    class_ids = [c['id'] for c in classes]
    if len(class_ids) != len(set(class_ids)):
        return False, "Duplicate class IDs found in metadata", None
    
    # Get export format to determine how to count classes
    export_format = metadata.get('export_info', {}).get('export_format', '').lower()
    
    if export_format == 'coco':
        # For instance segmentation (COCO), background is implicit (class 0)
        # QAnnotate uses class IDs starting from 1
        num_classes = len(classes) + 1  
    else:
        # For semantic segmentation (mask), background is explicit
        # QAnnotate uses class IDs starting from 1
        num_classes = len(classes) + 1  
    
    return True, None, num_classes


def validate_full_dataset(dataset_path: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Perform complete dataset validation.
    
    This is the main validation function that chains all validations.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (is_valid, error_message, dataset_info)
        - dataset_info contains: metadata, num_classes, num_images, etc.
    """
    # Step 1: Validate path
    valid, error = validate_dataset_path(dataset_path)
    if not valid:
        return False, error, None
    
    # Step 2: Validate metadata file
    valid, error, metadata = validate_qannotate_metadata(dataset_path)
    if not valid:
        return False, error, None
    
    # Step 3: Validate format (support mask, coco, and YOLO)
    valid, error = validate_dataset_format(
        metadata, 
        allowed_formats=['mask', 'coco', 'yolo11-detect', 'yolo11-seg', 'yolo11-obb']
    )
    if not valid:
        return False, error, None
    
    # Step 4: Validate directory structure
    valid, error = validate_dataset_structure(dataset_path, metadata)
    if not valid:
        return False, error, None
    
    # Step 5: Validate classes
    valid, error, num_classes = validate_classes(metadata)
    if not valid:
        return False, error, None
    
    # Extract dataset info
    statistics = metadata.get('statistics', {})
    num_images = statistics.get('num_images', 0)
    
    class_catalog = metadata.get('class_catalog', {})
    classes_list = class_catalog.get('classes', [])
    
    # Build class_names with background at index 0
    class_names = ['Background']  # Start with background at index 0
    class_names.extend([c['label'] for c in classes_list])  
    
    dataset_info = {
        'metadata': metadata,
        'num_classes': num_classes,
        'num_images': num_images,
        'class_names': class_names,
        'export_format': metadata.get('export_info', {}).get('export_format'),
        'image_format': metadata.get('export_info', {}).get('image_format'),
        'mask_format': metadata.get('export_info', {}).get('mask_format'),
        'num_bands': metadata.get('raster_info', {}).get('dimensions', {}).get('bands', 3) 
    }
    
    return True, None, dataset_info


# =============================================================================
# TRAINING CONFIGURATION VALIDATION
# =============================================================================

def validate_output_directory(output_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Validate output directory (create if doesn't exist).
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not output_dir:
        return False, "Output directory is empty"
    
    # Try to create directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create output directory: {str(e)}"
    
    # Check write permissions
    if not os.access(output_dir, os.W_OK):
        return False, f"No write permission for output directory: {output_dir}"
    
    return True, None


def validate_hyperparameters(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate training hyperparameters.
    
    Args:
        config: Training configuration dictionary from UI
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate epochs
    epochs = config.get('epochs', 0)
    if epochs <= 0:
        return False, "Number of epochs must be greater than 0"
    
    # Validate batch size
    batch_size = config.get('batch_size', 0)
    if batch_size <= 0:
        return False, "Batch size must be greater than 0"
    
    # Validate learning rate
    lr = config.get('learning_rate', 0)
    if lr <= 0:
        return False, "Learning rate must be greater than 0"
    
    # Validate image size
    image_size = config.get('image_size', 0)
    if image_size <= 0:
        return False, "Image size must be greater than 0"
    
    # Check if image size is multiple of 32 
    if image_size % 32 != 0:
        # This is a warning, not an error
        pass
    
    # Validate validation split
    val_split = config.get('val_split', 0)
    if not (0 <= val_split < 1):
        return False, "Validation split must be between 0 and 1"
    
    if val_split == 0:
        return False, "Validation split cannot be 0 (no validation data)"
    
    # Validate patience (if early stopping enabled)
    if config.get('early_stopping'):
        patience = config.get('patience', 0)
        if patience <= 0:
            return False, "Patience must be greater than 0 when early stopping is enabled"
    
    # Validate checkpoint path (if resume training enabled)
    if config.get('resume_training'):
        checkpoint_path = config.get('checkpoint_path')
        if not checkpoint_path:
            return False, "Checkpoint path is required when resume training is enabled"
        if not os.path.exists(checkpoint_path):
            return False, f"Checkpoint file not found: {checkpoint_path}"
    
    return True, None


# =============================================================================
# DEVICE VALIDATION
# =============================================================================

def validate_device(device: str, cuda_device_id: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate device availability.
    
    Args:
        device: Device type ('CPU' or 'CUDA')
        cuda_device_id: CUDA device ID (0-7) if device is 'CUDA'
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not TORCH_AVAILABLE:
        return False, "PyTorch is not installed. Please install PyTorch first."
    
    if device == 'CPU':
        return True, None
    
    elif device == 'CUDA':
        if not torch.cuda.is_available():
            return False, (
                "CUDA is not available.\n"
                "Please check:\n"
                "- NVIDIA GPU is installed\n"
                "- CUDA drivers are installed\n"
                "- PyTorch was installed with CUDA support"
            )
        
        # Validate CUDA device ID
        if cuda_device_id is not None:
            num_devices = torch.cuda.device_count()
            if cuda_device_id >= num_devices:
                return False, (
                    f"CUDA device {cuda_device_id} not available.\n"
                    f"Available devices: 0-{num_devices - 1}"
                )
        
        return True, None
    
    else:
        return False, f"Unknown device type: {device}"


def get_device_info() -> str:
    """
    Get human-readable device information.
    
    Returns:
        String describing available devices
    """
    if not TORCH_AVAILABLE:
        return "PyTorch not installed"
    
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        return f"CUDA available: {num_devices} device(s) - {', '.join(device_names)}"
    else:
        return "CPU only (CUDA not available)"


# =============================================================================
# COMPLETE VALIDATION
# =============================================================================

def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Perform complete training configuration validation.
    
    This is the main entry point that validates everything before training.
    
    Args:
        config: Complete training configuration from UI
        
    Returns:
        Tuple of (is_valid, error_message, validated_info)
    """
    # Step 1: Validate dataset
    valid, error, dataset_info = validate_full_dataset(config['dataset_path'])
    if not valid:
        return False, f"Dataset validation failed:\n{error}", None
    
    # Step 2: Validate output directory
    valid, error = validate_output_directory(config['output_dir'])
    if not valid:
        return False, f"Output directory validation failed:\n{error}", None
    
    # Step 3: Validate hyperparameters
    valid, error = validate_hyperparameters(config)
    if not valid:
        return False, f"Hyperparameter validation failed:\n{error}", None
    
    # Step 4: Validate device
    valid, error = validate_device(config['device'], config.get('cuda_device_id'))
    if not valid:
        return False, f"Device validation failed:\n{error}", None
    
    # All validations passed
    validated_info = {
        'dataset_info': dataset_info,
        'device_info': get_device_info()
    }
    
    return True, None, validated_info


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_dataset_info(dataset_info: Dict) -> str:
    """
    Format dataset info for display in UI.
    
    Args:
        dataset_info: Dataset information dictionary
        
    Returns:
        Formatted string for display
    """
    lines = [
        f"Format: {dataset_info['export_format']}",
        f"Images: {dataset_info['num_images']}",
        f"Classes: {dataset_info['num_classes']} (including background)",
        f"Class names: {', '.join(dataset_info['class_names'])}"
    ]
    return '\n'.join(lines)