"""
Training worker for server-side training execution

Receives RAW config from endpoints and normalizes it before calling core.training functions.
"""

import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging
import torch
from .models import job_manager

logger = logging.getLogger(__name__)


class TrainingWorker:
    """
    Worker for running training in a separate thread.
    
    Responsibilities:
    1. Receive RAW config from endpoint
    2. Normalize config (task names, architecture names, device, optimizer, scheduler)
    3. Call core.training functions with normalized config
    4. Report progress via callbacks
    """
    
    def __init__(
        self,
        job_id: str,
        config: Dict[str, Any],
        dataset_info: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the worker.
        
        Args:
            job_id: Unique job identifier
            config: RAW training configuration from plugin
            dataset_info: Validated dataset information
            callbacks: Optional callbacks for progress updates
        """
        self.job_id = job_id
        self.config = config  # RAW config (will be normalized)
        self.dataset_info = dataset_info
        self.callbacks = callbacks or {}
        self.should_stop = False
        self.trainer = None
        
        # Register worker with job manager
        job_manager.set_worker(job_id, self)
    
    def _log(self, message: str):
        """Log message to both logger and job manager."""
        logger.info(f"[{self.job_id}] {message}")
        job_manager.add_log(self.job_id, message)
        
        if 'on_log' in self.callbacks:
            self.callbacks['on_log'](message)
    
    def run(self):
        """
        Main training function (runs in separate thread).
        
        Orchestrates:
        1. Device setup
        2. Config normalization
        3. Task routing (PyTorch vs YOLO)
        """
        try:
            self._log("\n" + "="*58)
            self._log("⚙️  TRAINING SETUP")
            self._log("="*58)
            
            # =================================================================
            # STEP 1: SETUP DEVICE
            # =================================================================
            device = self._setup_device()
            
            # =================================================================
            # STEP 2: ROUTE TO APPROPRIATE TRAINING PIPELINE
            # =================================================================
            export_format = self.dataset_info.get('export_format', 'mask')
            
            if export_format in ['yolo11-detect', 'yolo11-seg', 'yolo11-obb']:
                self._run_yolo_training(device)
            else:
                self._run_pytorch_training(device)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
            self._log(f"\n❌ {error_msg}")
            
            job_manager.complete_job(
                job_id=self.job_id,
                success=False,
                error=str(e),
                traceback_str=traceback.format_exc()
            )
            
            if 'on_complete' in self.callbacks:
                self.callbacks['on_complete'](False, str(e))
    
    # =================================================================
    # DEVICE SETUP
    # =================================================================
    
    def _setup_device(self) -> torch.device:
        """
        Setup training device from RAW config.
        
        Normalizes: "CUDA" → cuda, "CPU" → cpu
        
        Returns:
            torch.device object
        """
        # Normalize device string (CUDA → cuda, CPU → cpu)
        device_str = self.config.get('device', 'cpu').lower()
        
        if device_str == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available on server")
            
            # Get GPU ID (default 0, handle None)
            cuda_device_id = self.config.get('cuda_device_id')
            if cuda_device_id is None:
                cuda_device_id = 0
            
            device = torch.device(f'cuda:{cuda_device_id}')
            self._log(f"🎮 Device: CUDA (GPU {cuda_device_id})")
            self._log(f"   {torch.cuda.get_device_name(cuda_device_id)}")
        else:
            device = torch.device('cpu')
            self._log("💻 Device: CPU")
        
        return device
    
    # =================================================================
    # PYTORCH TRAINING PIPELINE
    # =================================================================
    
    def _run_pytorch_training(self, device: torch.device):
        """
        Run PyTorch training (semantic or instance segmentation).
        
        Steps:
        1. Create dataloaders
        2. Create model
        3. Create optimizer & scheduler
        4. Create trainer
        5. Train
        6. Export & report
        """
        from core.training import create_dataloaders
        
        # =================================================================
        # STEP 1: CREATE DATALOADERS
        # =================================================================
        self._log("\n📦 Loading dataset...")
        
        train_loader, val_loader, num_classes, num_bands = create_dataloaders(
            dataset_path=self.config['dataset_path'],
            dataset_info=self.dataset_info,
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            val_split=self.config.get('val_split', 0.2),
            num_workers=0,
            random_seed=42,
            use_pretrained=self.config['pretrained'],
            task=self.config['task'],
            augmentation_config=self.config.get('augmentation_config')
        )
        
        self._log(f"   Train batches: {len(train_loader)}")
        self._log(f"   Val batches: {len(val_loader)}")
        self._log(f"   Classes: {num_classes}")
        self._log(f"   Bands: {num_bands}")
        
        self._log(f"   Bands: {num_bands}")

        # =================================================================
        # STEP 1.5: LR FINDER (if enabled, Semantic Segmentation only)
        # =================================================================
        suggested_lr = None
        if self.config.get('use_lr_finder', False) and self.config['task'] == 'Semantic Segmentation':
            self._log("\n🔍 Running LR Finder...")
            self._log("   ⚠️  Using Adam optimizer for LR range test (regardless of your optimizer choice)")
            
            try:
                from core.training import create_model
                from core.training.lr_finder import LRFinder
                from core.training.trainer import SemanticSegmentationLoss
                
                # Create temporary model for LR test
                temp_model = create_model(
                    task=self.config['task'],
                    architecture=self.config['architecture'],
                    backbone=self.config['backbone'],
                    num_classes=num_classes,
                    in_channels=num_bands,
                    pretrained=self.config['pretrained'],
                    encoder_weights=self.config.get('encoder_weights')
                ).to(device)
                
                # Create temporary Adam optimizer for LR test
                temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=1e-7)
                
                # Create loss criterion
                criterion = SemanticSegmentationLoss()
                
                # Run LR Finder
                lr_finder = LRFinder(
                    model=temp_model,
                    optimizer=temp_optimizer,
                    criterion=criterion,
                    device=device,
                    callback=None  # Could add progress callback here
                )
                
                suggested_lr, _ = lr_finder.range_test(train_loader)
                
                # Save plot to report assets
                report_assets_dir = Path(self.config['output_dir']) / 'report' / 'report_assets'
                report_assets_dir.mkdir(parents=True, exist_ok=True)
                lr_finder.save_plot(str(report_assets_dir), filename='lr_finder.png')
                
                self._log(f"   ✅ Suggested LR: {suggested_lr:.2e}")
                self._log(f"   📊 LR Finder plot saved to report/report_assets/lr_finder.png")
                
                # Override learning rate with suggested value
                self.config['learning_rate'] = suggested_lr
                
                # Clean up temporary model
                del temp_model
                del temp_optimizer
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
            except Exception as e:
                self._log(f"   ⚠️  LR Finder failed: {str(e)}")
                self._log(f"   → Falling back to manual learning rate: {self.config['learning_rate']:.2e}")
                # Continue with manual LR
        
        # =================================================================
        # STEP 2: CREATE MODEL
        # =================================================================
        from core.training import create_model
        
        self._log("\n🏗️  Creating model...")
        
        # Log configuration for model creation
        import json
        self._log("\n" + "="*80)
        self._log("🔍 WORKER CREATING MODEL WITH:")
        self._log("="*80)
        self._log(json.dumps({
            'task': self.config['task'],
            'architecture': self.config.get('architecture'),
            'backbone': self.config.get('backbone'),
            'num_classes': num_classes,
            'in_channels': num_bands,
            'pretrained': self.config['pretrained'],
            'encoder_weights': self.config.get('encoder_weights')
        }, indent=2))
        self._log("="*80 + "\n")
        
        model = create_model(
            task=self.config['task'],
            architecture=self.config['architecture'],
            backbone=self.config['backbone'],
            num_classes=num_classes,
            in_channels=num_bands,
            pretrained=self.config['pretrained'],
            encoder_weights=self.config.get('encoder_weights')
        )
        
        model = model.to(device)
        self._log(f"   Architecture: {self.config['architecture']}")
        self._log(f"   Backbone: {self.config['backbone']}")
        
        # =================================================================
        # STEP 3: CREATE OPTIMIZER
        # =================================================================
        from core.training.training_config import get_optimizer_config
        
        # Get optimizer config with defaults
        optimizer_config = get_optimizer_config(
            optimizer_name=self.config['optimizer'],
            task=self.config['task'],
            custom_lr=self.config.get('learning_rate')
        )
        
        # Create optimizer using trainers.py helper
        from core.training.trainer import create_optimizer
        
        optimizer = create_optimizer(
            model=model,
            optimizer_name=self.config['optimizer'],
            optimizer_params=optimizer_config
        )
        
        self._log(f"\n⚙️  Optimizer: {self.config['optimizer']}")
        self._log(f"   Learning rate: {optimizer_config['lr']}")
        
        # =================================================================
        # STEP 4: CREATE SCHEDULER (optional)
        # =================================================================
        scheduler = None
        if self.config.get('scheduler'):
            from core.training.training_config import get_scheduler_config
            
            # Get scheduler config with defaults
            scheduler_config = get_scheduler_config(
                scheduler_name=self.config['scheduler'],
                task=self.config['task'],
                num_images=len(train_loader.dataset),
                num_epochs=self.config['epochs'],
                steps_per_epoch=len(train_loader)
            )
            
            # Create scheduler using trainers.py helper
            from core.training.trainer import create_scheduler
            
            scheduler = create_scheduler(
                optimizer=optimizer,
                scheduler_name=self.config['scheduler'],
                scheduler_params=scheduler_config,
                num_epochs=self.config['epochs'],
                steps_per_epoch=len(train_loader)
            )
            
            self._log(f"   Scheduler: {self.config['scheduler']}")
        
        # =================================================================
        # STEP 5: CREATE TRAINER
        # =================================================================
        task = self.config['task']
        
        # Prepare callbacks for trainer
        trainer_callbacks = {
            'on_log': self._log,
            'on_epoch_end': self._on_epoch_end,
            'on_train_end': self._on_train_end
        }
        
        # Create appropriate trainer based on task
        if 'Instance' in task:
            from core.training.trainer import InstanceSegmentationTrainer
            
            self._log("\n🏋️  Initializing Instance Segmentation Trainer...")
            
            self.trainer = InstanceSegmentationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=str(device),
                num_classes=num_classes,
                architecture=self.config['architecture'],
                encoder_name=self.config['backbone'],
                output_dir=self.config['output_dir'],
                callbacks=trainer_callbacks,
                save_best=self.config.get('save_best', True)
            )
        else:
            from core.training.trainer import SemanticSegmentationTrainer
            
            self._log("\n🏋️  Initializing Semantic Segmentation Trainer...")
            
            self.trainer = SemanticSegmentationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=str(device),
                num_classes=num_classes,
                architecture=self.config['architecture'],
                encoder_name=self.config['backbone'],
                output_dir=self.config['output_dir'],
                callbacks=trainer_callbacks,
                save_best=self.config.get('save_best', True)
            )
        
        # =================================================================
        # STEP 6: START TRAINING
        # =================================================================
        self._log("\n" + "="*58)
        self._log("🚀 STARTING TRAINING")
        self._log("="*58)
        self._log("ℹ The first epoch is now running. Metrics will only be printed once the entire epoch is completed. This is expected and may take several minutes for large datasets.")
        
        # Start training
        self.trainer.train(
            num_epochs=self.config['epochs'],
            early_stopping_patience=self.config.get('patience'),
            resume_from=self.config.get('checkpoint_path')
        )
        
        # Training completed successfully
        self._log("\n✅ Training completed successfully!")
        
        # =================================================================
        # STEP 7: EXPORT QMTP
        # =================================================================
        self._log("\n📦 Exporting model to .qmtp format...")
        
        try:
            from core.training.qmtp_exporter import QMTPExporter
            
            # Determine encoder vs backbone
            is_instance = 'Instance' in self.config['task']
            encoder = None if is_instance else self.config.get('backbone')
            backbone = self.config.get('backbone') if is_instance else None
            
            # Create exporter
            exporter = QMTPExporter(
                output_dir=self.config['output_dir'],
                model_name=self.config.get('model_name', 'trained_model')
            )
            
            # Export (semantic vs instance segmentation)
            qmtp_path = exporter.export(
                model_path=str(Path(self.config['output_dir']) / 'best_model_weights.pth'),
                dataset_path=self.config['dataset_path'],
                task=self.config['task'],
                architecture=self.config['architecture'],
                encoder=encoder,
                backbone=backbone,
                encoder_weights=self.config.get('encoder_weights'),
                use_pretrained=self.config.get('pretrained', True),
                tile_size=self.config.get('image_size', 512),
                best_iou=self.trainer.best_iou if not is_instance else None,
                best_ap=self.trainer.best_ap if is_instance else None,
                total_epochs=self.config['epochs']
            )
            
            self._log(f"✅ Model exported: {qmtp_path.name}")
            
        except Exception as e:
            self._log(f"⚠️  QMTP export failed: {str(e)}")
            # Continue anyway - training is done
        
        # =================================================================
        # STEP 8: CALCULATE CLASS DISTRIBUTION FOR REPORT
        # =================================================================
        self._log("\n📊 Calculating class distribution for report...")
        
        # Calculate class distribution from dataset_info
        class_distribution = {}
        
        # Determine if it's instance or semantic segmentation
        is_instance = 'Instance' in self.config['task']
        
        if is_instance:
            # For instance segmentation with COCO format: use annotations_per_class
            annotations_per_class = self.dataset_info.get('metadata', {}).get('statistics', {}).get('annotations_per_class', {})
            
            if annotations_per_class:
                # Convert string keys to int and calculate total
                total_annotations = sum(int(count) for count in annotations_per_class.values())
                
                # Add background with 0 instances (not annotated in COCO)
                class_distribution[0] = {
                    'count': 0,
                    'percentage': 0.0
                }
                
                for class_id_str, count in annotations_per_class.items():
                    class_id = int(class_id_str)
                    count = int(count)
                    percentage = (count / total_annotations * 100) if total_annotations > 0 else 0.0
                    
                    class_distribution[class_id] = {
                        'count': count,
                        'percentage': percentage
                    }
                
                self._log(f"   Instance distribution: {len(annotations_per_class)} classes, {total_annotations} total instances")
            else:
                self._log("   ⚠️  No annotations_per_class found in dataset_info")
                self._log("   Distribution table will show N/A")
                self._log("   (Update QAnnotate to add this statistic during COCO export)")
        else:
            # For semantic segmentation: count pixels
            pixels_per_class = self.dataset_info.get('metadata', {}).get('statistics', {}).get('pixels_per_class', {})
            
            if pixels_per_class:
                total_pixels = sum(pixels_per_class.values())
                
                for class_id_str, pixel_count in pixels_per_class.items():
                    class_id = int(class_id_str)
                    percentage = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0
                    
                    class_distribution[class_id] = {
                        'count': pixel_count,
                        'percentage': percentage
                    }
                
                self._log(f"   Pixel distribution: {len(pixels_per_class)} classes, {total_pixels:,} total pixels")
            else:
                self._log("   ⚠️  No pixel distribution found in dataset_info")
        
        # =================================================================
        # STEP 9: GENERATE HTML REPORT (optional)
        # =================================================================
        report_path = None
        if self.config.get('generate_report', True):
            self._log("\n📊 Generating training report...")
            
            try:
                from core.training.reporting.html_generator import generate_training_report
                
                # Build config for report generator
                report_config = {
                    'model_name': self.config.get('model_name', 'trained_model'),
                    'architecture': self.config['architecture'],
                    'backbone': self.config['backbone'],
                    'auto_lr': self.config.get('use_lr_finder', False),
                    'learning_rate': self.config.get('learning_rate', 0.001),
                    'batch_size': self.config.get('batch_size', 4),
                    'image_size': self.config.get('image_size', 512),
                    'val_split': self.config.get('val_split', 0.2),
                    'num_classes': num_classes,
                    'total_images': len(train_loader.dataset) + len(val_loader.dataset),
                    'num_train_images': len(train_loader.dataset),
                    'num_val_images': len(val_loader.dataset),
                    'total_train_time': self.trainer.total_train_time,
                    'device': self.config.get('device', 'cpu'),
                    'optimizer_name': self.config.get('optimizer', 'Adam'),
                    'scheduler_name': self.config.get('scheduler', 'ReduceLROnPlateau'),
                    'early_stopping': self.config.get('patience') is not None,
                    'patience': self.config.get('patience'),
                    'pretrained': self.config.get('pretrained', True),
                    # Extract class names directly from dataset_info
                    'class_names': self.dataset_info.get('class_names', []),
                    # Extract full class info with colors from metadata.class_catalog.classes
                    'dataset_classes': self.dataset_info.get('metadata', {}).get('class_catalog', {}).get('classes', []),
                    # Add computed class distribution
                    'class_distribution': class_distribution,
                    'plugin_version': '0.9.0'
                }
                
                # Determine task string for report
                task_str = 'instance_segmentation' if 'Instance' in self.config['task'] else 'semantic_segmentation'
                
                # Generate report
                report_path = generate_training_report(
                    output_dir=self.config['output_dir'],
                    config=report_config,
                    trainer=self.trainer,
                    dataset_info=self.dataset_info,
                    task=task_str,
                    val_loader=val_loader
                )
                
                self._log(f"✅ Report generated: report/training_report.html")
                
            except Exception as e:
                import traceback
                self._log("=" * 80)
                self._log("✗ REPORT GENERATION FAILED")
                self._log("=" * 80)
                self._log(f"Exception type: {type(e).__name__}")
                self._log(f"Error message: {str(e)}")
                self._log("")
                self._log("Full traceback:")
                self._log(traceback.format_exc())
                self._log("=" * 80)
                # Continue anyway - training is done
        
        # Mark job as complete
        job_manager.complete_job(
            job_id=self.job_id,
            success=True,
            checkpoint_path=str(Path(self.config['output_dir']) / 'final_model.pth'),
            best_checkpoint_path=str(Path(self.config['output_dir']) / 'best_model.pth'),
            report_path=str(report_path) if report_path else None
        )
        
        if 'on_complete' in self.callbacks:
            self.callbacks['on_complete'](True, "Training completed successfully")
    
    # =================================================================
    # TRAINER CALLBACKS
    # =================================================================
    
    def _on_epoch_end(self, **kwargs):
        """Callback for epoch end - log metrics only."""
        epoch = kwargs.get('epoch', 0)
        train_loss = kwargs.get('train_loss')
        val_loss = kwargs.get('val_loss')
        val_iou = kwargs.get('val_iou')
        val_ap = kwargs.get('val_ap')
        
        # Build log message based on available metrics
        log_parts = [f"Epoch {epoch + 1}/{self.config['epochs']}"]
        
        if train_loss is not None:
            log_parts.append(f"Train Loss: {train_loss:.4f}")
        
        if val_loss is not None:
            log_parts.append(f"Val Loss: {val_loss:.4f}")
        
        # Use AP for instance/YOLO, IoU for semantic
        if val_ap is not None:
            log_parts.append(f"mAP: {val_ap:.4f}")
        elif val_iou is not None:
            log_parts.append(f"IoU: {val_iou:.4f}")
        
        self._log(" | ".join(log_parts))
    
    def _on_train_end(self, **kwargs):
        """Callback for training end."""
        best_iou = kwargs.get('best_iou', 0)
        best_epoch = kwargs.get('best_epoch', 0)
        
        self._log(f"\n🎉 Best model: Epoch {best_epoch + 1} (IoU/AP: {best_iou:.4f})")

    
    # =================================================================
    # YOLO TRAINING PIPELINE
    # =================================================================
    
    def _run_yolo_training(self, device: torch.device):
        """
        Run YOLO training (detection, segmentation, or OBB).
        
        Steps:
        1. Determine YOLO task
        2. Create YOLO trainer
        3. Train
        4. Export & report
        """
        # =================================================================
        # STEP 1: DETERMINE YOLO TASK
        # =================================================================
        export_format = self.dataset_info.get('export_format')
        
        # Map export format to YOLO trainer class
        task_map = {
            'yolo11-detect': ('detect', 'YOLODetectionTrainer'),
            'yolo11-seg': ('segment', 'YOLOSegmentationTrainer'),
            'yolo11-obb': ('obb', 'YOLOOBBTrainer')
        }
        
        if export_format not in task_map:
            raise ValueError(f"Invalid YOLO export format: {export_format}")
        
        yolo_task, trainer_class_name = task_map[export_format]
        
        self._log(f"\n🎯 YOLO Task: {yolo_task}")
        
        # =================================================================
        # STEP 2: PREPARE DATA.YAML PATH
        # =================================================================
        from pathlib import Path
        data_yaml_path = Path(self.config['dataset_path']) / 'data.yaml'
        
        if not data_yaml_path.exists():
            raise FileNotFoundError(
                f"data.yaml not found in dataset directory.\n"
                f"Expected: {data_yaml_path}"
            )
        
        self._log(f"📄 Dataset config: {data_yaml_path}")
        
        # =================================================================
        # STEP 3: CREATE YOLO TRAINER
        # =================================================================
        from core.training.yolo.yolo_trainer import (
            YOLODetectionTrainer,
            YOLOSegmentationTrainer,
            YOLOOBBTrainer
        )
        
        # Get trainer class
        trainer_classes = {
            'YOLODetectionTrainer': YOLODetectionTrainer,
            'YOLOSegmentationTrainer': YOLOSegmentationTrainer,
            'YOLOOBBTrainer': YOLOOBBTrainer
        }
        
        TrainerClass = trainer_classes[trainer_class_name]
        
        # Prepare callbacks
        trainer_callbacks = {
            'on_log': self._log,
            'on_epoch_end': self._on_epoch_end,
            'on_train_end': self._on_train_end
        }
        
        self._log("\n🏋️  Initializing YOLO Trainer...")
        
        # Create trainer
        self.trainer = TrainerClass(
            data_yaml_path=str(data_yaml_path),
            model_size=self.config.get('model_size', 'n'),
            pretrained=self.config.get('pretrained', True),
            output_dir=self.config['output_dir'],
            callbacks=trainer_callbacks
        )
        
        # =================================================================
        # STEP 4: START TRAINING
        # =================================================================
        self._log("\n" + "="*58)
        self._log("🚀 STARTING TRAINING")
        self._log("="*58)
        
        # Train
        self.trainer.train(
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            learning_rate=self.config['learning_rate'],
            early_stopping=self.config.get('early_stopping', False),
            patience=self.config.get('patience', 100),
            device=str(device),
            workers=0,
            optimizer=self.config.get('optimizer', 'Auto').lower(),
            augmentation_config=self.config.get('augmentation_config')
        )
        
        # Training completed
        self._log("\n✅ Training completed successfully!")
        
        # =================================================================
        # STEP 5: EXPORT QMTP
        # =================================================================
        self._log("\n📦 Exporting model to .qmtp format...")
        
        try:
            from core.training.qmtp_exporter import QMTPExporter
            
            # YOLO best model is at: output_dir/train/weights/best.pt
            yolo_best_model = Path(self.config['output_dir']) / 'train' / 'weights' / 'best.pt'
            
            # Create exporter
            exporter = QMTPExporter(
                output_dir=self.config['output_dir'],
                model_name=self.config.get('model_name', 'trained_model')
            )
            
            # Determine task string for QMTP
            qmtp_task_map = {
                'detect': 'YOLO Detection',
                'segment': 'YOLO Segmentation',
                'obb': 'YOLO OBB'
            }
            
            # Export
            qmtp_path = exporter.export(
                model_path=str(yolo_best_model),
                dataset_path=self.config['dataset_path'],
                task=qmtp_task_map[yolo_task],
                architecture=f"YOLO11-{yolo_task}",
                model_size=self.config.get('model_size', 'n'),
                image_size=self.config.get('image_size', 512),
                use_pretrained=self.config.get('pretrained', True),
                tile_size=self.config.get('image_size', 512),
                best_map50_95=self.trainer.best_metric,
                total_epochs=self.config['epochs']
            )
            
            self._log(f"✅ Model exported: {qmtp_path.name}")
            
        except Exception as e:
            self._log(f"⚠️  QMTP export failed: {str(e)}")
            # Continue anyway
        
        # =================================================================
        # STEP 6: GENERATE HTML REPORT (optional)
        # =================================================================
        report_path = None
        if self.config.get('generate_report', True):
            self._log("\n📊 Generating training report...")
            
            try:
                from core.training.reporting.yolo_html_generator import YOLOHTMLReportGenerator
                
                # Build config for report generator
                report_config = {
                    'model_name': self.config.get('model_name', 'YOLO11 Model'),
                    'model_size': self.config.get('model_size', 'n'),
                    'learning_rate': self.config.get('learning_rate', 0.01),
                    'batch_size': self.config.get('batch_size', 16),
                    'image_size': self.config.get('image_size', 512),
                    'total_images': self.dataset_info.get('num_images', 0),
                    'total_train_time': self.trainer.total_train_time,
                    'device': self.config.get('device', 'cpu'), 
                    'optimizer_name': self.config.get('optimizer', 'SGD'),
                    'early_stopping': self.config.get('early_stopping', False),
                    'patience': self.config.get('patience', 0) if self.config.get('early_stopping', False) else 0,
                    'pretrained': self.config.get('pretrained', True),
                    'class_names': self.dataset_info.get('class_names', []),
                    'num_train_images': self.dataset_info.get('num_train_images', 'N/A'),
                    'num_val_images': self.dataset_info.get('num_val_images', 'N/A'),
                    'plugin_version': '0.9.0'
                }
                
                # Determine task string for report
                yolo_report_task_map = {
                    'detect': 'yolo_detection',
                    'segment': 'yolo_segmentation',
                    'obb': 'yolo_obb'
                }
                
                # Create generator
                generator = YOLOHTMLReportGenerator(
                    output_dir=self.config['output_dir'],
                    task=yolo_report_task_map[yolo_task]
                )
                
                # Generate report
                report_path = generator.generate(
                    config=report_config,
                    history=self.trainer.history,
                    trainer=self.trainer
                )
                
                self._log(f"✅ Report generated: report/training_report.html")
                
            except Exception as e:
                import traceback
                self._log(f"⚠️  Report generation failed: {str(e)}")
                self._log(f"   {traceback.format_exc()}")
                # Continue anyway
        
        # Mark job as complete
        job_manager.complete_job(
            job_id=self.job_id,
            success=True,
            checkpoint_path=str(Path(self.config['output_dir']) / 'train' / 'weights' / 'last.pt'),
            best_checkpoint_path=str(Path(self.config['output_dir']) / 'train' / 'weights' / 'best.pt'),
            report_path=str(report_path) if report_path else None
        )
        
        if 'on_complete' in self.callbacks:
            self.callbacks['on_complete'](True, "Training completed successfully")