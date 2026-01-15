"""
Dataset loading utilities for QModel Trainer

This module handles loading QAnnotate exports into PyTorch datasets.
Supports semantic segmentation with mask format (TIFF/PNG).

"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image
import json

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# =============================================================================
# PYTORCH DATASET CLASSES
# =============================================================================

class SemanticSegmentationDataset(Dataset):
    """
    PyTorch Dataset for semantic segmentation.
    
    Loads image-mask pairs from QAnnotate mask export format.
    Supports both TIFF and PNG formats.
    
    Args:
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        transform: Optional transforms to apply to images and masks
        num_classes: Number of classes (including background)
    """
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Any] = None,
        num_classes: int = 2
    ):
        """Initialize the dataset."""
        assert len(image_paths) == len(mask_paths), \
            f"Number of images ({len(image_paths)}) must match masks ({len(mask_paths)})"
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.num_classes = num_classes
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a sample from the dataset.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
            - image_tensor: Shape (C, H, W), float32, normalized
            - mask_tensor: Shape (H, W), int64, class indices
        """
        # Load image and mask
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        
        # Apply transforms if provided (Albumentations)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure image is float tensor (should already be from ToTensorV2)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            # Image is already normalized to [0, 1] by _load_image
        
        # CRITICAL: Ensure mask is Long (int64) for CrossEntropyLoss
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            # Convert from uint8 (Byte) to int64 (Long)
            mask = mask.long()
        
        return image, mask
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from disk with support for multiple data types.
        
        Supports:
        - Uint8 images (0-255) - standard RGB imagery
        - Float32/Float64 images - satellite/remote sensing data
        - Uint16 images - high bit-depth sensors
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array of shape (H, W, C), dtype float32, range [0, 1]
            
        Note:
            All images are normalized to [0, 1] float32 for consistency.
            Albumentations expects this format and will handle it correctly.
        """
        try:
            # Use rasterio for TIFF files (better format support)
            if image_path.lower().endswith(('.tif', '.tiff')):
                return self._load_image_gdal(image_path)
            else:
                # Fallback to PIL for other formats (PNG, JPEG)
                return self._load_image_pil(image_path)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")

    def _load_image_gdal(self, image_path: str) -> np.ndarray:
        """
        Load TIFF image using rasterio (supports Float32/Float64).
        
        This method properly handles various data types common in remote sensing:
        - Uint8: Standard RGB (0-255)
        - Float32/Float64: Reflectance data (0-1 or 0-10000)
        - Uint16: High bit-depth sensors (0-65535)
        
        Args:
            image_path: Path to TIFF file
            
        Returns:
            numpy array (H, W, C), dtype float32, range [0, 1]
        """
        import rasterio
        
        # Open with rasterio
        with rasterio.open(image_path) as src:
            # Read all bands (rasterio returns as (bands, height, width))
            image = src.read()  # Shape: (C, H, W)
            
            # Transpose to (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        
        # Handle band count
        num_bands = image.shape[2]
        if num_bands == 1:
            # Grayscale → replicate to 3 channels for consistency
            image = np.stack([image[:, :, 0]] * 3, axis=-1)
        elif num_bands >= 3:
            # Multi-band → take first 3 channels (RGB or first 3 bands)
            image = image[:, :, :3]
        
        # Normalize to [0, 1] float32 based on data type
        if image.dtype == np.uint8:
            # Uint8: divide by 255
            image = image.astype(np.float32) / 255.0
        
        elif image.dtype in [np.float32, np.float64]:
            # Float: handle different value ranges
            image = image.astype(np.float32)
            
            # Detect range and normalize accordingly
            img_max = image.max()
            
            if img_max > 1.0:
                # Likely scaled reflectance (0-10000 or similar)
                if img_max <= 10000.0:
                    image = image / 10000.0
                else:
                    # Use dynamic scaling for other ranges
                    image = image / img_max
            
            # Clip to [0, 1] for safety
            image = np.clip(image, 0.0, 1.0)
        
        elif image.dtype in [np.uint16, np.int16]:
            # Uint16: scale from [0, 65535] to [0, 1]
            image = image.astype(np.float32) / 65535.0
        
        else:
            # Other types: use min-max normalization
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min)).astype(np.float32)
            else:
                image = image.astype(np.float32)
        
        return image

    def _load_image_pil(self, image_path: str) -> np.ndarray:
        """
        Load image using PIL (for PNG, JPEG, etc.).
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array (H, W, C), dtype float32, range [0, 1]
        """
        from PIL import Image
        
        # Open with PIL and convert to RGB
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.uint8)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image


    def _load_mask(self, mask_path: str) -> np.ndarray:
        """
        Load a segmentation mask from disk using PIL.
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            numpy array of shape (H, W), dtype uint8
        """
        try:
            # Use PIL
            mask = Image.open(mask_path)
            mask = np.array(mask, dtype=np.uint8)
            
            # Ensure mask is 2D
            if len(mask.shape) == 3:
                # Take first channel if multi-channel
                mask = mask[:, :, 0]
            
            return mask
            
        except Exception as e:
            raise RuntimeError(f"Failed to load mask {mask_path}: {str(e)}")


# =============================================================================
# TRANSFORM UTILITIES
# =============================================================================

def get_training_transforms(
    image_size: int = 512, 
    use_imagenet_stats: bool = True,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Get data augmentation transforms for training.
    
    Args:
        image_size: Target image size (height and width)
        use_imagenet_stats: If True, normalize with ImageNet statistics
        augmentation_config: Augmentation configuration from UI (optional)
        
    Returns:
        Albumentations Compose transform
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError("albumentations is required for data augmentation")
    
    # =========================================================================
    # LOG 1: RECEPTION CONFIG
    # =========================================================================
    import logging
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*80)
    logger.info("🔍 DATASETS.PY → get_training_transforms() CALLED")
    logger.info("="*80)
    logger.info(f"image_size: {image_size}")
    logger.info(f"use_imagenet_stats: {use_imagenet_stats}")
    if augmentation_config:
        logger.info(f"augmentation_config received:")
        logger.info(f"  enabled: {augmentation_config.get('enabled')}")
        logger.info(f"  hflip: {augmentation_config.get('hflip')}")
        logger.info(f"  vflip: {augmentation_config.get('vflip')}")
        logger.info(f"  rotate90: {augmentation_config.get('rotate90')}")
        logger.info(f"  brightness: {augmentation_config.get('brightness')}")
        logger.info(f"  contrast: {augmentation_config.get('contrast')}")
        logger.info(f"  hue: {augmentation_config.get('hue')}")
        logger.info(f"  saturation: {augmentation_config.get('saturation')}")
        logger.info(f"  blur: {augmentation_config.get('blur')}")
        logger.info(f"  noise: {augmentation_config.get('noise')}")
    else:
        logger.info("⚠️  NO augmentation_config provided (will use defaults)")
    logger.info("="*80 + "\n")
    
    transform_list = [A.Resize(image_size, image_size)]
    
    # Apply augmentations if provided and enabled
    if augmentation_config and augmentation_config.get('enabled', False):
        logger.info("✅ Augmentations ENABLED - Building pipeline...")
        
        # Geometric transforms
        if augmentation_config.get('hflip', 0) > 0:
            prob = augmentation_config['hflip'] / 100.0
            transform_list.append(A.HorizontalFlip(p=prob))
            logger.info(f"  + HorizontalFlip(p={prob})")
        
        if augmentation_config.get('vflip', 0) > 0:
            prob = augmentation_config['vflip'] / 100.0
            transform_list.append(A.VerticalFlip(p=prob))
            logger.info(f"  + VerticalFlip(p={prob})")
        
        if augmentation_config.get('rotate90', 0) > 0:
            prob = augmentation_config['rotate90'] / 100.0
            transform_list.append(A.RandomRotate90(p=prob))
            logger.info(f"  + RandomRotate90(p={prob})")
        
        # Radiometric transforms (in OneOf with p=0.5)
        radiometric_transforms = []
        
        if augmentation_config.get('brightness', 0) > 0:
            limit = augmentation_config['brightness'] / 100.0
            radiometric_transforms.append(A.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=0, p=1.0))
            logger.info(f"  + Brightness(limit={limit})")
        
        if augmentation_config.get('contrast', 0) > 0:
            limit = augmentation_config['contrast'] / 100.0
            radiometric_transforms.append(A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=limit, p=1.0))
            logger.info(f"  + Contrast(limit={limit})")
        
        if augmentation_config.get('hue', 0) > 0:
            limit = augmentation_config['hue'] * 0.2
            radiometric_transforms.append(A.HueSaturationValue(hue_shift_limit=limit, sat_shift_limit=0, val_shift_limit=0, p=1.0))
            logger.info(f"  + Hue(limit={limit:.2f})")
        
        if augmentation_config.get('saturation', 0) > 0:
            limit = augmentation_config['saturation'] * 0.3 
            radiometric_transforms.append(A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=limit, val_shift_limit=0, p=1.0))
            logger.info(f"  + Saturation(limit={limit:.2f})")
        
        if augmentation_config.get('blur', 0) > 0:
            blur_limit = 3 + int(augmentation_config['blur'] * 0.04)
            radiometric_transforms.append(A.GaussianBlur(blur_limit=(blur_limit, blur_limit), p=1.0))
            logger.info(f"  + GaussianBlur(blur_limit={blur_limit})")
        
        if augmentation_config.get('noise', 0) > 0:
            var_limit = (10, 10 + int(augmentation_config['noise'] * 0.4))
            radiometric_transforms.append(A.GaussNoise(var_limit=var_limit, p=1.0))
            logger.info(f"  + GaussianNoise(var_limit={var_limit})")
        
        # Add radiometric transforms with OneOf
        if radiometric_transforms:
            transform_list.append(A.OneOf(radiometric_transforms, p=0.5))
            logger.info(f"  + OneOf({len(radiometric_transforms)} radiometric transforms, p=0.5)")
    else:
        logger.info("❌ Augmentations DISABLED or config missing")
    
    # Normalization
    if use_imagenet_stats:
        transform_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0))
        logger.info("  + Normalize(ImageNet)")
    else:
        transform_list.append(A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0))
        logger.info("  + Normalize(Simple)")
    
    transform_list.append(ToTensorV2())
    logger.info("  + ToTensorV2()")

    # =========================================================================
    # LOG 2: PIPELINE FINAL
    # =========================================================================
    logger.info(f"\n📊 Final Albumentations pipeline: {len(transform_list)} transforms")
    logger.info("="*80 + "\n")
    
    return A.Compose(transform_list)

def get_validation_transforms(image_size: int = 512, use_imagenet_stats: bool = True) -> Any:
    """
    Get transforms for validation (no augmentation).
    
    Args:
        image_size: Target image size (height and width)
        use_imagenet_stats: If True, normalize with ImageNet statistics (for pretrained models).
                           If False, simple normalization to [0, 1].
        
    Returns:
        Albumentations Compose transform
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError(
            "albumentations is required.\n"
            "Install with: pip install albumentations"
        )
    
    transform_list = [
        A.Resize(image_size, image_size),
    ]
    
    # Add normalization based on pretrained model usage
    if use_imagenet_stats:
        transform_list.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            )
        )
    else:
        transform_list.append(
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            )
        )
    
    transform_list.append(ToTensorV2())
    
    transform = A.Compose(transform_list)
    return transform



# =============================================================================
# DATASET LOADING FUNCTIONS
# =============================================================================

def load_mask_dataset(
    dataset_path: str,
    dataset_info: Dict[str, Any],
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load image and mask file paths from QAnnotate mask export.
    
    Args:
        dataset_path: Path to dataset root directory
        dataset_info: Dataset information from validators
        val_split: Fraction of data to use for validation (0-1)
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_images, train_masks, val_images, val_masks)
        Each is a list of file paths
    """
    from core.training.utils.validators import is_valid_image_file  # ← Import the filter
    
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    # Get image format from metadata
    image_format = dataset_info.get('image_format', 'tif')
    mask_format = dataset_info.get('mask_format', 'tif')
    
    # Find all valid image files (filtering out temporary files)
    all_image_files = sorted(images_dir.glob(f'*.{image_format}'))
    image_files = [f for f in all_image_files if is_valid_image_file(f.name)]
    
    # Find corresponding mask files
    mask_files = []
    for img_path in image_files:
        # Build mask path (same name, different directory and extension)
        mask_name = img_path.stem + f'.{mask_format}'
        mask_path = masks_dir / mask_name
        
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for image {img_path.name}:\n"
                f"Expected: {mask_path}"
            )
        
        mask_files.append(mask_path)
    
    # Convert to string paths
    image_files = [str(p) for p in image_files]
    mask_files = [str(p) for p in mask_files]
    
    # Split into train and validation
    num_samples = len(image_files)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val
    
    # Shuffle with fixed seed for reproducibility
    np.random.seed(random_seed)
    indices = np.random.permutation(num_samples)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_images = [image_files[i] for i in train_indices]
    train_masks = [mask_files[i] for i in train_indices]
    val_images = [image_files[i] for i in val_indices]
    val_masks = [mask_files[i] for i in val_indices]
    
    return train_images, train_masks, val_images, val_masks


def create_dataloaders(
    dataset_path: str,
    dataset_info: Dict[str, Any],
    batch_size: int = 4,
    image_size: int = 512,
    val_split: float = 0.2,
    num_workers: int = 0,  # ← Changé de None à 0 par défaut pour QGIS
    random_seed: int = 42,
    use_pretrained: bool = True,  # ← NOUVEAU paramètre
    task: str = 'Semantic Segmentation',
    augmentation_config: Optional[Dict[str, Any]] = None  
) -> Tuple[DataLoader, DataLoader, int, int]: # ← (train_loader, val_loader, num_classes, num_bands)
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        dataset_path: Path to dataset root directory
        dataset_info: Dataset information from validators
        batch_size: Batch size for training
        image_size: Target image size (height and width)
        val_split: Fraction of data for validation
        num_workers: Number of workers for data loading (0 for QGIS compatibility)
        random_seed: Random seed for reproducibility
        use_pretrained: If True, use ImageNet normalization (for pretrained backbones).
                       If False, simple normalization to [0, 1].
        
    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    # Determine task type
    task_lower = task.lower()
    
    # INSTANCE SEGMENTATION (COCO FORMAT)
    if 'instance' in task_lower:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for data loading")
        
        # Get number of classes (+1 for background)
        num_classes = dataset_info['num_classes']

        # Get number of bands from metadata (default to 3 if not present)
        num_bands = dataset_info.get('num_bands', 3)
        
        images_dir = Path(dataset_path) / 'images'
        annotations_file = Path(dataset_path) / 'annotations.json'
        
        # Create category ID mapping (COCO IDs to contiguous 1-indexed)
        category_id_to_contiguous = {cls_id: cls_id for cls_id in range(1, num_classes)}
        
        # Split dataset
        train_image_ids, val_image_ids = load_coco_dataset(
            dataset_path=dataset_path,
            dataset_info=dataset_info,
            val_split=val_split,
            random_seed=random_seed
        )
        
        # Get transforms
        train_transform = get_instance_training_transforms(
            image_size, 
            use_imagenet_stats=use_pretrained,
            augmentation_config=augmentation_config
        )
        val_transform = get_instance_validation_transforms(image_size, use_imagenet_stats=use_pretrained)
        
        # Create datasets
        train_dataset = InstanceSegmentationDataset(
            images_dir=str(images_dir),
            annotations_file=str(annotations_file),
            transform=train_transform,
            category_id_to_contiguous=category_id_to_contiguous
        )
        
        val_dataset = InstanceSegmentationDataset(
            images_dir=str(images_dir),
            annotations_file=str(annotations_file),
            transform=val_transform,
            category_id_to_contiguous=category_id_to_contiguous
        )
        
        # Filter datasets to only include train/val image IDs
        train_dataset.image_ids = train_image_ids
        val_dataset.image_ids = val_image_ids
        
        # Create dataloaders with custom collate_fn
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_instance,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_instance,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, num_classes, num_bands
    
    # SEMANTIC SEGMENTATION (MASK FORMAT)
    elif 'semantic' in task_lower:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for data loading")
        
        # Get number of classes from dataset info
        num_classes = dataset_info['num_classes']

        # Get number of bands from metadata (default to 3 if not present)
        num_bands = dataset_info.get('num_bands', 3)
        
        # Load file paths and split
        train_images, train_masks, val_images, val_masks = load_mask_dataset(
            dataset_path=dataset_path,
            dataset_info=dataset_info,
            val_split=val_split,
            random_seed=random_seed
        )
        
        # DEBUG: Log avant appel get_training_transforms
        print(f"\n🔍 create_dataloaders() calling get_training_transforms()")
        print(f"   augmentation_config parameter: {augmentation_config}")
        print(f"   type: {type(augmentation_config)}")

        # Get transforms (with appropriate normalization)
        train_transform = get_training_transforms(
            image_size, 
            use_imagenet_stats=use_pretrained,
            augmentation_config=augmentation_config
        )
        val_transform = get_validation_transforms(image_size, use_imagenet_stats=use_pretrained)
        
        # Create datasets
        train_dataset = SemanticSegmentationDataset(
            image_paths=train_images,
            mask_paths=train_masks,
            transform=train_transform,
            num_classes=num_classes
        )
        
        val_dataset = SemanticSegmentationDataset(
            image_paths=val_images,
            mask_paths=val_masks,
            transform=val_transform,
            num_classes=num_classes
        )
        
        # Create dataloaders
        # IMPORTANT: num_workers=0 for QGIS (évite de spawner des instances QGIS)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, num_classes, num_bands

    else:
        raise ValueError(
            f"Unsupported task: {task}\n"
            f"Supported tasks: 'Semantic Segmentation', 'Instance Segmentation'"
        )

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_dataset_statistics(
    dataset_path: str,
    dataset_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute statistics about the dataset (for logging).
    
    Args:
        dataset_path: Path to dataset root directory
        dataset_info: Dataset information from validators
        
    Returns:
        Dictionary with statistics
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images'
    
    # Count files
    image_format = dataset_info.get('image_format', 'tif')
    num_images = len(list(images_dir.glob(f'*.{image_format}')))
    
    # Get sample image size
    sample_image = list(images_dir.glob(f'*.{image_format}'))[0]
    
    if sample_image.suffix.lower() in ['.tif', '.tiff']:
        # Use rasterio for TIFF files
        import rasterio
        with rasterio.open(str(sample_image)) as src:
            height = src.height
            width = src.width
            channels = src.count
    else:
        img = Image.open(sample_image)
        width, height = img.size
        channels = len(img.getbands())
    
    stats = {
        'num_images': num_images,
        'num_classes': dataset_info['num_classes'],
        'image_size': (height, width),
        'num_channels': channels,
        'class_names': dataset_info['class_names']
    }
    
    return stats


def verify_dataloader(dataloader: DataLoader) -> bool:
    """
    Verify that a dataloader can produce batches without errors.
    
    Args:
        dataloader: PyTorch DataLoader to test
        
    Returns:
        True if successful, raises exception otherwise
    """
    try:
        # Try to load one batch
        batch = next(iter(dataloader))
        images, masks = batch
        
        # Check shapes
        assert len(images.shape) == 4, f"Expected 4D images, got {images.shape}"
        assert len(masks.shape) == 3, f"Expected 3D masks, got {masks.shape}"
        
        # Check batch consistency
        assert images.shape[0] == masks.shape[0], "Batch size mismatch"
        assert images.shape[2] == masks.shape[1], "Height mismatch"
        assert images.shape[3] == masks.shape[2], "Width mismatch"
        
        return True
    
    except Exception as e:
        raise RuntimeError(f"DataLoader verification failed: {str(e)}")


# =============================================================================
# INSTANCE SEGMENTATION DATASET (COCO FORMAT)
# =============================================================================

class InstanceSegmentationDataset(Dataset):
    """
    PyTorch Dataset for instance segmentation using COCO format.
    
    Loads image-annotation pairs from QAnnotate COCO export format.
    Returns targets compatible with Mask R-CNN.
    
    Args:
        images_dir: Directory containing images
        annotations_file: Path to COCO annotations.json
        transform: Optional transforms (Albumentations)
        category_id_to_contiguous: Mapping from COCO category_id to contiguous indices
    """
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        transform: Optional[Any] = None,
        category_id_to_contiguous: Optional[Dict[int, int]] = None
    ):
        """Initialize the dataset."""
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.category_id_to_contiguous = category_id_to_contiguous or {}
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data['categories']}
        
        # Group annotations by image_id
        self.image_id_to_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)
        
        # List of image IDs (for indexing)
        self.image_ids = list(self.images.keys())
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            """
            Load and return a sample from the dataset.
            
            Args:
                idx: Index of the sample to load
                
            Returns:
                Tuple of (image_tensor, target_dict)
            """
            image_id = self.image_ids[idx]
            image_info = self.images[image_id]
            
            # Load image
            image_path = self.images_dir / image_info['file_name']
            image = self._load_image(str(image_path))
            
            # Get annotations for this image
            annotations = self.image_id_to_annotations.get(image_id, [])
            
            # Convert annotations to target format
            boxes = []
            labels = []
            masks = []
            areas = []
            iscrowd = []
            
            for ann in annotations:
                # Get category
                category_id = ann['category_id']
                if self.category_id_to_contiguous:
                    label = self.category_id_to_contiguous[category_id]
                else:
                    label = category_id
                
                # Get bbox in [x, y, w, h] COCO format (absolute pixels)
                x, y, w, h = ann['bbox']

                # FILTRAGE STRICT : Ignorer les bboxes invalides
                # 1. Largeur et hauteur doivent être > 1
                if w <= 1 or h <= 1:
                    continue

                # 2. La box doit être dans l'image
                if x < 0 or y < 0 or x + w > image_info['width'] or y + h > image_info['height']:
                    continue

                # 3. Vérifier que la conversion est valide
                x1, y1, x2, y2 = x, y, x + w, y + h
                if x2 <= x1 or y2 <= y1:
                    continue

                # Convert to [x1, y1, x2, y2] pascal_voc format (absolute pixels)
                # Add small padding (prevent edge pixels loss)
                x1_padded = max(0, x1 - 1)
                y1_padded = max(0, y1 - 1)
                x2_padded = min(image_info['width'] - 1, x2 + 1)
                y2_padded = min(image_info['height'] - 1, y2 + 1)
                boxes.append([x1_padded, y1_padded, x2_padded, y2_padded])
                labels.append(label)
                areas.append(ann['area'])
                iscrowd.append(ann['iscrowd'])
                
                # Convert polygon to mask
                mask = self._polygon_to_mask(
                    ann['segmentation'],
                    image_info['height'],
                    image_info['width']
                )
                masks.append(mask)
            
            # Convert to numpy arrays for Albumentations
            boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
            masks = np.array(masks, dtype=np.uint8) if masks else np.zeros((0, image_info['height'], image_info['width']), dtype=np.uint8)
            
            # Apply transforms if provided
            if self.transform:
                transformed = self.transform(
                    image=image,
                    masks=list(masks),
                    bboxes=boxes,
                    labels=labels
                )
                image = transformed['image']
                
                # Check if lists are not empty
                if len(transformed['bboxes']) > 0:
                    boxes = np.array(transformed['bboxes'], dtype=np.float32)
                    labels = np.array(transformed['labels'], dtype=np.int64)
                    masks = np.array(transformed['masks'], dtype=np.uint8)
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    labels = np.zeros((0,), dtype=np.int64)
                    masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)
            
            # Ensure image is float tensor
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                # Image is already normalized to [0, 1] by _load_image
            
            # Convert to PyTorch tensors (boxes stay in absolute pixel coords for Mask R-CNN)
            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'masks': torch.as_tensor(masks, dtype=torch.uint8),
                'image_id': torch.tensor([image_id], dtype=torch.int64),
                'area': torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)
            }
            
            return image, target
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image with multi-format support.
        
        Uses the same loading logic as SemanticSegmentationDataset.
        Supports Uint8, Float32/Float64, and Uint16 data types.
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array of shape (H, W, C), dtype float32, range [0, 1]
        """
        try:
            if image_path.lower().endswith(('.tif', '.tiff')):
                return self._load_image_gdal(image_path)
            else:
                return self._load_image_pil(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")

    def _load_image_gdal(self, image_path: str) -> np.ndarray:
        """
        Load TIFF image using rasterio (supports Float32/Float64).
        
        This method properly handles various data types common in remote sensing:
        - Uint8: Standard RGB (0-255)
        - Float32/Float64: Reflectance data (0-1 or 0-10000)
        - Uint16: High bit-depth sensors (0-65535)
        
        Args:
            image_path: Path to TIFF file
            
        Returns:
            numpy array (H, W, C), dtype float32, range [0, 1]
        """
        import rasterio
        
        # Open with rasterio
        with rasterio.open(image_path) as src:
            # Read all bands (rasterio returns as (bands, height, width))
            image = src.read()  # Shape: (C, H, W)
            
            # Transpose to (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        
        # Handle band count
        num_bands = image.shape[2]
        if num_bands == 1:
            # Grayscale → replicate to 3 channels for consistency
            image = np.stack([image[:, :, 0]] * 3, axis=-1)
        elif num_bands >= 3:
            # Multi-band → take first 3 channels (RGB or first 3 bands)
            image = image[:, :, :3]
        
        # Normalize to [0, 1] float32 based on data type
        if image.dtype == np.uint8:
            # Uint8: divide by 255
            image = image.astype(np.float32) / 255.0
        
        elif image.dtype in [np.float32, np.float64]:
            # Float: handle different value ranges
            image = image.astype(np.float32)
            
            # Detect range and normalize accordingly
            img_max = image.max()
            
            if img_max > 1.0:
                # Likely scaled reflectance (0-10000 or similar)
                if img_max <= 10000.0:
                    image = image / 10000.0
                else:
                    # Use dynamic scaling for other ranges
                    image = image / img_max
            
            # Clip to [0, 1] for safety
            image = np.clip(image, 0.0, 1.0)
        
        elif image.dtype in [np.uint16, np.int16]:
            # Uint16: scale from [0, 65535] to [0, 1]
            image = image.astype(np.float32) / 65535.0
        
        else:
            # Other types: use min-max normalization
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min)).astype(np.float32)
            else:
                image = image.astype(np.float32)
        
        return image

    def _load_image_pil(self, image_path: str) -> np.ndarray:
        """Load image with PIL (for PNG, JPEG)."""
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.uint8)
        image = image.astype(np.float32) / 255.0
        return image
    
    def _polygon_to_mask(
        self,
        segmentation: List[List[float]],
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Convert COCO polygon segmentation to binary mask.
        
        Args:
            segmentation: List of polygons [[x1,y1,x2,y2,...], ...]
            height: Mask height
            width: Mask width
            
        Returns:
            Binary mask of shape (H, W), dtype uint8
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise ImportError("Pillow (PIL) is required for polygon to mask conversion")
        
        # Create blank mask
        mask = Image.new('L', (width, height), 0)
        
        # Draw each polygon
        for polygon in segmentation:
            # Convert flat list to list of tuples [(x1,y1), (x2,y2), ...]
            poly_points = []
            for i in range(0, len(polygon), 2):
                x = polygon[i]
                y = polygon[i + 1]
                poly_points.append((x, y))
            
            # Draw filled polygon
            ImageDraw.Draw(mask).polygon(poly_points, outline=1, fill=1)
        
        return np.array(mask, dtype=np.uint8)


def get_instance_training_transforms(
    image_size: int = 512, 
    use_imagenet_stats: bool = True,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Get data augmentation transforms for instance segmentation training.
    
    Args:
        image_size: Target image size (height and width)
        use_imagenet_stats: If True, normalize with ImageNet statistics
        augmentation_config: Augmentation configuration from UI (optional)
        
    Returns:
        Albumentations Compose transform with bbox support
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError("albumentations is required")
    
    # DEBUG: Log augmentation config
    if augmentation_config:
        print(f"[DEBUG] get_instance_training_transforms() received augmentation_config: {augmentation_config}")
    else:
        print("[DEBUG] get_instance_training_transforms() - NO augmentation_config provided")
    
    transform_list = [A.Resize(image_size, image_size)]
    
    # Apply augmentations if provided and enabled
    if augmentation_config and augmentation_config.get('enabled', False):
        # Geometric transforms (same as semantic)
        if augmentation_config.get('hflip', 0) > 0:
            transform_list.append(A.HorizontalFlip(p=augmentation_config['hflip'] / 100.0))
        
        if augmentation_config.get('vflip', 0) > 0:
            transform_list.append(A.VerticalFlip(p=augmentation_config['vflip'] / 100.0))
        
        if augmentation_config.get('rotate90', 0) > 0:
            transform_list.append(A.RandomRotate90(p=augmentation_config['rotate90'] / 100.0))
        
        # Radiometric transforms
        radiometric_transforms = []
        
        if augmentation_config.get('brightness', 0) > 0:
            limit = augmentation_config['brightness'] / 100.0
            radiometric_transforms.append(A.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=0, p=1.0))
        
        if augmentation_config.get('contrast', 0) > 0:
            limit = augmentation_config['contrast'] / 100.0
            radiometric_transforms.append(A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=limit, p=1.0))
        
        if augmentation_config.get('hue', 0) > 0:
            limit = int(augmentation_config['hue'] * 0.2)
            radiometric_transforms.append(A.HueSaturationValue(hue_shift_limit=limit, sat_shift_limit=0, val_shift_limit=0, p=1.0))
        
        if augmentation_config.get('saturation', 0) > 0:
            limit = int(augmentation_config['saturation'] * 0.3)
            radiometric_transforms.append(A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=limit, val_shift_limit=0, p=1.0))
        
        if augmentation_config.get('blur', 0) > 0:
            blur_limit = 3 + int(augmentation_config['blur'] * 0.04)
            radiometric_transforms.append(A.GaussianBlur(blur_limit=(blur_limit, blur_limit), p=1.0))
        
        if augmentation_config.get('noise', 0) > 0:
            var_limit = (10, 10 + int(augmentation_config['noise'] * 0.4))
            radiometric_transforms.append(A.GaussNoise(var_limit=var_limit, p=1.0))
        
        if radiometric_transforms:
            transform_list.append(A.OneOf(radiometric_transforms, p=0.5))
    
    # Normalization
    if use_imagenet_stats:
        transform_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0))
    else:
        transform_list.append(A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0))
    
    transform_list.append(ToTensorV2())

    # DEBUG: Log final transform pipeline
    print(f"[DEBUG] Final Albumentations pipeline ({len(transform_list)} transforms):")
    for i, t in enumerate(transform_list):
        print(f"  [{i}] {t.__class__.__name__}")
    
    # CRITICAL: bbox_params for bounding box augmentation
    return A.Compose(
        transform_list,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1.0, min_visibility=0.1)
    )


def get_instance_validation_transforms(image_size: int = 512, use_imagenet_stats: bool = True) -> Any:
    """
    Get transforms for instance segmentation validation (no augmentation).
    
    Args:
        image_size: Target image size
        use_imagenet_stats: If True, normalize with ImageNet statistics
        
    Returns:
        Albumentations Compose transform with bbox support
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError("albumentations is required")
    
    transform_list = [A.Resize(image_size, image_size)]

    
    if use_imagenet_stats:
        transform_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0))
    else:
        transform_list.append(A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0))
    
    transform_list.append(ToTensorV2())
    
    transform = A.Compose(
        transform_list,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1.0, min_visibility=0.1)
    )
    
    return transform


def load_coco_dataset(
    dataset_path: str,
    dataset_info: Dict[str, Any],
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Load COCO dataset and split into train/val image IDs.
    
    Args:
        dataset_path: Path to dataset root directory
        dataset_info: Dataset information from validators
        val_split: Fraction of data for validation (0-1)
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_image_ids, val_image_ids)
    """
    dataset_path = Path(dataset_path)
    annotations_file = dataset_path / 'annotations.json'
    
    # Load COCO annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get all image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    
    # Split into train and validation
    num_samples = len(image_ids)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val
    
    # Shuffle with fixed seed for reproducibility
    np.random.seed(random_seed)
    indices = np.random.permutation(num_samples)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_image_ids = [image_ids[i] for i in train_indices]
    val_image_ids = [image_ids[i] for i in val_indices]
    
    return train_image_ids, val_image_ids


def collate_fn_instance(batch):
    """
    Custom collate function for instance segmentation batches.
    
    Mask R-CNN requires targets as a list of dicts (not batched tensors).
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        Tuple of (images, targets) where:
        - images: FloatTensor (B, C, H, W)
        - targets: List of dicts (length B)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images into batch
    images = torch.stack(images, dim=0)
    
    return images, targets