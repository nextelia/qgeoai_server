"""
YOLO trainers for QModel Trainer

Wrappers around Ultralytics YOLO API with QModel Trainer callbacks.
Supports YOLO11 detection, segmentation, and OBB tasks.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Callable
from datetime import datetime
from io import StringIO

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None


# =============================================================================
# BASE YOLO TRAINER
# =============================================================================

class YOLOTrainerBase:
    """
    Base class for YOLO trainers.
    
    Provides common functionality for all YOLO variants (detect, segment, obb).
    Wraps Ultralytics YOLO API with QModel Trainer callback interface.
    
    Args:
        data_yaml_path: Path to data.yaml file
        task: YOLO task ('detect', 'segment', 'obb')
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        pretrained: Whether to use pretrained weights
        output_dir: Directory to save checkpoints and logs
        callbacks: Dictionary of callback functions for progress updates
    """
    
    def __init__(
        self,
        data_yaml_path: str,
        task: str,
        model_size: str = 'n',
        pretrained: bool = True,
        output_dir: str = './outputs',
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        """Initialize the YOLO trainer."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is not installed.\n"
                "Install with: pip install ultralytics"
            )
        
        self.data_yaml_path = Path(data_yaml_path)
        self.task = task
        self.model_size = model_size
        self.pretrained = pretrained
        self.output_dir = Path(output_dir)
        self.callbacks = callbacks or {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.model = None
        self.should_stop = False
        self.best_metric = 0.0
        self.best_epoch = 0
        self.total_train_time = 0.0
        
        # History for reporting
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {},
            'learning_rate': []
        }
    
    def _create_model(self):
        """
        Create YOLO model.
        
        If pretrained=True, loads from plugin's models/ directory (manual download required).
        If pretrained=False, creates model from scratch (no weights).
        
        Returns:
            YOLO model instance
        """
        # Model naming: yolo11{size}-{task}.pt
        # Detection: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
        # Segmentation: yolo11n-seg.pt, yolo11s-seg.pt, etc.
        # OBB: yolo11n-obb.pt, yolo11s-obb.pt, etc.
        if self.task == 'detect':
            model_name = f'yolo11{self.model_size}.pt'
        elif self.task == 'segment':
            model_name = f'yolo11{self.model_size}-seg.pt'
        elif self.task == 'obb':
            model_name = f'yolo11{self.model_size}-obb.pt'
        else:
            # Fallback (should not happen)
            model_name = f'yolo11{self.model_size}-{self.task}.pt'
        
        if self.pretrained:
            # Load from server models directory
            server_models_dir = Path.home() / '.qgeoai' / 'server' / 'models'
            server_models_dir.mkdir(parents=True, exist_ok=True)
            
            local_model_path = server_models_dir / model_name
            
            if not local_model_path.exists():
                raise FileNotFoundError(
                    f"\n{'='*60}\n"
                    f"❌ PRETRAINED MODEL NOT FOUND\n"
                    f"{'='*60}\n\n"
                    f"The model '{model_name}' is required but not found.\n\n"
                    f"SOLUTION:\n"
                    f"1. Download the model from:\n"
                    f"   https://docs.ultralytics.com/models/yolo11/\n\n"
                    f"2. Place the file here:\n"
                    f"   {server_models_dir}\n\n"
                    f"3. The expected filename is:\n"
                    f"   {model_name}\n\n"
                    f"4. Retry training\n\n"
                    f"TIP: You can also uncheck 'Use pretrained weights' to train from scratch.\n"
                    f"{'='*60}\n"
                )
            
            self._log(f"📁 Loading pretrained model: {model_name}")
            self._log(f"   Path: {local_model_path}")
            
            model = YOLO(str(local_model_path))
            return model
        else:
            # Create model from scratch
            yaml_name = f'yolo11{self.model_size}.yaml'
            self._log(f"🔧 Creating model from scratch: {yaml_name}")
            model = YOLO(yaml_name)
            return model
    
    def _extract_metrics(self, metrics: Dict) -> Dict[str, float]:
        """Extract task-specific metrics from Ultralytics metrics."""
        raise NotImplementedError
    
    def _get_primary_metric(self, metrics: Dict) -> float:
        """Get primary metric for best model selection."""
        raise NotImplementedError
    
    def _get_metric_name(self) -> str:
        """Get primary metric name for display."""
        raise NotImplementedError
    
    def _log(self, message: str):
        """Log a message via callback."""
        if 'on_log' in self.callbacks:
            self.callbacks['on_log'](message)
    
    def stop(self):
        """Request to stop training."""
        self.should_stop = True
        self._log("\n⚠️  Stop requested. Training will stop after current epoch...")
    
    def train(
        self,
        epochs: int,
        batch_size: int = 16,
        image_size: int = 512,
        learning_rate: float = 0.01,
        early_stopping: bool = False,
        patience: int = 100,
        device: str = 'cuda',
        workers: int = 0,
        optimizer: str = 'auto',
        **kwargs
    ):
        kwargs.pop("early_stopping", None)
        kwargs.pop("patience", None)

        """Train YOLO model from scratch."""
        global_start_time = time.time()

        try:
            self._log("\n🔧 Preparing YOLO model...")
            self.model = self._create_model()
            
            # Apply early stopping logic
            if not early_stopping:
                patience = 0  
            
            self._log(f"✅ Model ready: YOLO11{self.model_size}-{self.task}")
            self._log(f"   Pretrained: {'Yes' if self.pretrained else 'No'}")
            self._log(f"   Image size: {image_size}")
            self._log(f"   Batch size: {batch_size}")
            self._log(f"   Learning rate: {learning_rate}")
            self._log(f"   Early stopping: {'Yes' if early_stopping else 'No'}{f' (patience={patience})' if early_stopping else ''}")
            self._log(f"   Epochs: {epochs}")

            self._log("\n" + "=" * 58)
            self._log("🎯 STARTING TRAINING")
            self._log("=" * 58)

            train_args = {
                'data': str(self.data_yaml_path),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': image_size,
                'lr0': learning_rate,
                'patience': patience,
                'device': device,
                'workers': workers,
                'project': str(self.output_dir),
                'name': 'train',
                'exist_ok': True,
                'pretrained': self.pretrained,
                'verbose': False,
                'plots': True,
                'cache': False,
            }
            
            # Apply data augmentation settings if provided
            augmentation_config = kwargs.get('augmentation_config')
            if augmentation_config and augmentation_config.get('enabled', False):
                # Geometric augmentations (compatible with YOLO)
                train_args['fliplr'] = augmentation_config.get('hflip', 0) / 100.0  # Horizontal flip probability
                train_args['flipud'] = augmentation_config.get('vflip', 0) / 100.0  # Vertical flip probability
                
                # YOLO uses 'degrees' for rotation (0-90)
                rotate90_prob = augmentation_config.get('rotate90', 0) / 100.0
                if rotate90_prob > 0:
                    # Convert probability to rotation range (YOLO rotates by random degrees)
                    train_args['degrees'] = 90.0 * rotate90_prob
                
                # Brightness (YOLO uses 'hsv_v' for value/brightness)
                brightness_intensity = augmentation_config.get('brightness', 0)
                if brightness_intensity > 0:
                    # YOLO hsv_v range: 0.0-1.0 (fraction of value range to vary)
                    train_args['hsv_v'] = brightness_intensity / 100.0
            else:
                # Disable all augmentations if not enabled
                train_args['fliplr'] = 0.0
                train_args['flipud'] = 0.0
                train_args['degrees'] = 0.0
                train_args['hsv_h'] = 0.0
                train_args['hsv_s'] = 0.0
                train_args['hsv_v'] = 0.0
            
            # Log augmentation config for debugging
            if augmentation_config:
                self._log("\n📊 Data Augmentation:")
                self._log(f"   Horizontal Flip: {train_args.get('fliplr', 0)*100:.0f}%")
                self._log(f"   Vertical Flip: {train_args.get('flipud', 0)*100:.0f}%")
                self._log(f"   Rotation: {train_args.get('degrees', 0):.0f}°")
                self._log(f"   Brightness: {train_args.get('hsv_v', 0)*100:.0f}%")
            else:
                self._log("\n📊 Data Augmentation: Disabled")
            
            # Add remaining kwargs (optimizer, etc.)
            for key, value in kwargs.items():
                if key not in ['augmentation_config']:
                    train_args[key] = value

            if optimizer != 'auto':
                train_args['optimizer'] = optimizer

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            self._log("\n🚀 Starting YOLO training...\n")

            def on_train_epoch_end(trainer):
                if self.should_stop:
                    trainer.stop = True
                    return

                epoch = trainer.epoch
                metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}

                if 'on_epoch_start' in self.callbacks:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    self.callbacks['on_epoch_start'](epoch=epoch)
                    sys.stdout = StringIO()
                    sys.stderr = StringIO()

                val_loss_raw = metrics.get('val/box_loss', 0.0)
                val_loss = float(val_loss_raw) if val_loss_raw is not None else 0.0
                
                task_metrics = self._extract_metrics(metrics)
                
                self.history['train_loss'].append(0.0)
                self.history['val_loss'].append(val_loss)
                
                for k, v in task_metrics.items():
                    self.history['metrics'].setdefault(k, []).append(v)
                
                if hasattr(trainer, 'optimizer'):
                    lr = trainer.optimizer.param_groups[0]['lr']
                    self.history['learning_rate'].append(lr)
                else:
                    lr = 0.0
                
                try:
                    primary_metric = self._get_primary_metric(metrics)
                except (KeyError, TypeError, ValueError):
                    primary_metric = 0.0
                
                is_best = primary_metric > self.best_metric
                if is_best:
                    self.best_metric = primary_metric
                    self.best_epoch = epoch
                
                if 'on_epoch_end' in self.callbacks:
                    epoch_time_raw = getattr(trainer, 'epoch_time', 0.0)
                    epoch_time = float(epoch_time_raw) if epoch_time_raw is not None else 0.0

                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

                    self.callbacks['on_epoch_end'](
                        epoch=epoch,
                        train_loss=None,
                        val_loss=val_loss,
                        val_iou=None,
                        val_ap=task_metrics.get('mAP50-95', 0.0),
                        val_f1=task_metrics.get('F1', 0.0),
                        val_precision=task_metrics.get('Precision', 0.0),
                        val_recall=task_metrics.get('Recall', 0.0),
                        learning_rate=lr,
                        epoch_time=epoch_time,
                        is_best=is_best
                    )

                    sys.stdout = StringIO()
                    sys.stderr = StringIO()

            self.model.add_callback('on_train_epoch_end', on_train_epoch_end)

            try:
                self.model.train(**train_args)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            self.total_train_time = time.time() - global_start_time
            self._save_training_summary(epochs)

        except Exception as e:
            self._log(f"\n❌ Training failed: {str(e)}")
            raise
    
    def _save_training_summary(self, num_epochs: int):
        """Save a text summary of training."""
        summary_path = self.output_dir / 'training_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*58 + "\n")
            f.write(f"TRAINING SUMMARY - YOLO11 {self.task.upper()}\n")
            f.write("="*58 + "\n\n")
            
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {num_epochs}\n")
            
            hours = int(self.total_train_time // 3600)
            minutes = int((self.total_train_time % 3600) // 60)
            seconds = int(self.total_train_time % 60)
            
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            f.write(f"Total training time: {time_str}\n\n")
            
            f.write(f"Best {self._get_metric_name()}: {self.best_metric:.4f}\n")
            f.write(f"Best epoch: {self.best_epoch + 1}\n")
            
            f.write("\n" + "="*60 + "\n")


# =============================================================================
# YOLO DETECTION TRAINER
# =============================================================================

class YOLODetectionTrainer(YOLOTrainerBase):
    """Trainer for YOLO11 object detection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task='detect', **kwargs)
    
    def _extract_metrics(self, metrics: Dict) -> Dict[str, float]:
        precision_raw = metrics.get('metrics/precision(B)')
        recall_raw = metrics.get('metrics/recall(B)')
        precision = float(precision_raw) if precision_raw else 0.0
        recall = float(recall_raw) if recall_raw else 0.0
        
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'mAP50-95': float(metrics.get('metrics/mAP50-95(B)') or 0.0),
            'mAP50': float(metrics.get('metrics/mAP50(B)') or 0.0),
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    def _get_primary_metric(self, metrics: Dict) -> float:
        return float(metrics.get('metrics/mAP50-95(B)') or 0.0)
    
    def _get_metric_name(self) -> str:
        return "mAP50-95"


# =============================================================================
# YOLO SEGMENTATION TRAINER
# =============================================================================

class YOLOSegmentationTrainer(YOLOTrainerBase):
    """Trainer for YOLO11 instance segmentation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task='segment', **kwargs)
    
    def _extract_metrics(self, metrics: Dict) -> Dict[str, float]:
        precision_raw = metrics.get('metrics/precision(M)')
        recall_raw = metrics.get('metrics/recall(M)')
        precision = float(precision_raw) if precision_raw else 0.0
        recall = float(recall_raw) if recall_raw else 0.0
        
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'mAP50-95': float(metrics.get('metrics/mAP50-95(M)') or 0.0),
            'mAP50': float(metrics.get('metrics/mAP50(M)') or 0.0),
            'mAP50-95_box': float(metrics.get('metrics/mAP50-95(B)') or 0.0),
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    def _get_primary_metric(self, metrics: Dict) -> float:
        return float(metrics.get('metrics/mAP50-95(M)') or 0.0)
    
    def _get_metric_name(self) -> str:
        return "mAP50-95 (Mask)"


# =============================================================================
# YOLO OBB TRAINER
# =============================================================================

class YOLOOBBTrainer(YOLOTrainerBase):
    """Trainer for YOLO11 oriented bounding boxes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, task='obb', **kwargs)
    
    def _extract_metrics(self, metrics: Dict) -> Dict[str, float]:
        precision_raw = metrics.get('metrics/precision(B)')
        recall_raw = metrics.get('metrics/recall(B)')
        precision = float(precision_raw) if precision_raw else 0.0
        recall = float(recall_raw) if recall_raw else 0.0
        
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'mAP50-95': float(metrics.get('metrics/mAP50-95(B)') or 0.0),
            'mAP50': float(metrics.get('metrics/mAP50(B)') or 0.0),
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    def _get_primary_metric(self, metrics: Dict) -> float:
        return float(metrics.get('metrics/mAP50-95(B)') or 0.0)
    
    def _get_metric_name(self) -> str:
        return "mAP50-95 (OBB)"