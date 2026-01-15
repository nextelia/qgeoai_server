"""
YOLO dataset validation utilities

Validates YOLO11 datasets exported by QAnnotate:
- Detection (bounding boxes)
- Segmentation (polygons)
- OBB (oriented bounding boxes)
"""

import os
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


def validate_yolo_dataset(dataset_path: str, metadata: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate YOLO dataset structure and data.yaml file.
    
    Args:
        dataset_path: Path to dataset directory
        metadata: Parsed QAnnotate metadata
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    dataset_path = Path(dataset_path)
    
    # Step 1: Check data.yaml exists
    data_yaml_path = dataset_path / 'data.yaml'
    if not data_yaml_path.exists():
        return False, "data.yaml file not found"
    
    # Step 2: Parse data.yaml
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, f"Invalid YAML in data.yaml: {str(e)}"
    except Exception as e:
        return False, f"Error reading data.yaml: {str(e)}"
    
    # Step 3: Validate data.yaml structure
    required_fields = ['path', 'train', 'names', 'nc']
    missing_fields = [field for field in required_fields if field not in data_yaml]
    
    if missing_fields:
        return False, f"data.yaml missing required fields: {', '.join(missing_fields)}"
    
    # Step 4: Validate paths in data.yaml
    yaml_path = Path(data_yaml['path'])
    
    # Check train directory
    train_images_dir = yaml_path / data_yaml['train']
    if not train_images_dir.exists():
        return False, f"Train images directory not found: {train_images_dir}"
    
    # Check val directory
    if 'val' in data_yaml:
        val_images_dir = yaml_path / data_yaml['val']
        if not val_images_dir.exists():
            return False, f"Val images directory not found: {val_images_dir}"
    
    # Step 5: Check labels directory structure
    # YOLO expects labels/train/ and labels/val/ parallel to images/
    labels_base_dir = yaml_path / 'labels'
    if not labels_base_dir.exists():
        return False, "labels/ directory not found"
    
    train_labels_dir = labels_base_dir / 'train'
    if not train_labels_dir.exists():
        return False, "labels/train/ directory not found"
    
    if 'val' in data_yaml:
        val_labels_dir = labels_base_dir / 'val'
        if not val_labels_dir.exists():
            return False, "labels/val/ directory not found"
    
    # Step 6: Validate classes count matches
    nc_yaml = data_yaml['nc']
    nc_metadata = metadata.get('class_catalog', {}).get('num_classes', 0)
    
    if nc_yaml != nc_metadata:
        return False, (
            f"Class count mismatch: data.yaml has {nc_yaml} classes, "
            f"metadata has {nc_metadata} classes"
        )
    
    # Step 7: Check that we have some images and labels
    train_images = list(train_images_dir.glob('*.tif*')) + list(train_images_dir.glob('*.png')) + list(train_images_dir.glob('*.jpg'))
    if not train_images:
        return False, "No images found in train directory"
    
    train_labels = list(train_labels_dir.glob('*.txt'))
    if not train_labels:
        return False, "No labels found in labels/train/ directory"
    
    # Step 8: Validate format-specific requirements
    yolo_variant = metadata.get('yolo_info', {}).get('variant')
    
    if yolo_variant not in ['detect', 'seg', 'obb']:
        return False, f"Unknown YOLO variant: {yolo_variant}"
    
    # Step 9: Validate a few label files for correct format
    valid, error = _validate_label_format(train_labels[:5], yolo_variant)
    if not valid:
        return False, error
    
    return True, None


def _validate_label_format(
    label_files: list,
    variant: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate label file format matches YOLO variant.
    
    Args:
        label_files: List of label file paths to check
        variant: YOLO variant ('detect', 'seg', 'obb')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    for label_path in label_files:
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Skip empty files 
            if not lines:
                continue
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # First value must be class_id (integer)
                try:
                    class_id = int(parts[0])
                except ValueError:
                    return False, (
                        f"Invalid class_id in {label_path.name} line {line_num}: "
                        f"expected integer, got '{parts[0]}'"
                    )
                
                # Validate format based on variant
                if variant == 'detect':
                    # Format: class_id x_center y_center width height (5 values)
                    if len(parts) != 5:
                        return False, (
                            f"Invalid detection format in {label_path.name} line {line_num}: "
                            f"expected 5 values (class_id x y w h), got {len(parts)}"
                        )
                
                elif variant == 'seg':
                    # Format: class_id x1 y1 x2 y2 ... (class_id + at least 6 coords = 7 values min)
                    if len(parts) < 7:
                        return False, (
                            f"Invalid segmentation format in {label_path.name} line {line_num}: "
                            f"expected at least 7 values (class_id + 3 points), got {len(parts)}"
                        )
                    # Must have even number of coordinates (x,y pairs)
                    if (len(parts) - 1) % 2 != 0:
                        return False, (
                            f"Invalid segmentation format in {label_path.name} line {line_num}: "
                            f"coordinates must come in x,y pairs"
                        )
                
                elif variant == 'obb':
                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 values)
                    if len(parts) != 9:
                        return False, (
                            f"Invalid OBB format in {label_path.name} line {line_num}: "
                            f"expected 9 values (class_id + 4 corner points), got {len(parts)}"
                        )
                
                # Validate that coordinates are floats between 0 and 1
                try:
                    coords = [float(x) for x in parts[1:]]
                    if not all(0 <= c <= 1 for c in coords):
                        return False, (
                            f"Invalid coordinates in {label_path.name} line {line_num}: "
                            f"all coordinates must be normalized (0-1)"
                        )
                except ValueError as e:
                    return False, (
                        f"Invalid coordinate value in {label_path.name} line {line_num}: {str(e)}"
                    )
        
        except Exception as e:
            return False, f"Error reading {label_path.name}: {str(e)}"
    
    return True, None


def get_yolo_task_from_metadata(metadata: Dict) -> str:
    """
    Get YOLO task name from metadata.
    
    Args:
        metadata: Parsed QAnnotate metadata
        
    Returns:
        Task name: 'detect', 'segment', or 'obb'
    """
    variant = metadata.get('yolo_info', {}).get('variant', 'detect')
    
    # Map QAnnotate variants to Ultralytics task names
    variant_map = {
        'detect': 'detect',
        'seg': 'seg', 
        'obb': 'obb'
    }
    
    return variant_map.get(variant, 'detect')