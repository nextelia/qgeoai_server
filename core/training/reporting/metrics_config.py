"""
Metrics configuration for different training tasks
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MetricDefinition:
    """Definition of a single metric."""
    key: str
    label: str
    format_str: str = '.4f'


@dataclass
class MetricsConfig:
    """Configuration for a training task's metrics."""
    task_name: str
    primary_metric: MetricDefinition
    display_metrics: List[MetricDefinition]
    
    def get_best_metric_value(self, trainer) -> float:
        """Extract best metric value from trainer."""
        raise NotImplementedError
    
    def get_metrics_from_history(self, history: Dict, epoch: int) -> Dict[str, float]:
        """Extract metrics at specific epoch from history."""
        metrics = {}
        for metric_def in self.display_metrics:
            if metric_def.key in history and history[metric_def.key]:
                metrics[metric_def.label] = history[metric_def.key][epoch]
        return metrics


class SemanticSegmentationMetrics(MetricsConfig):
    """Metrics for Semantic Segmentation (UNet)."""
    
    def __init__(self):
        super().__init__(
            task_name='Semantic Segmentation',
            primary_metric=MetricDefinition('val_iou', 'IoU'),
            display_metrics=[
                MetricDefinition('val_iou', 'IoU'),
                MetricDefinition('val_f1', 'F1 Score'),
                MetricDefinition('val_precision', 'Precision'),
                MetricDefinition('val_recall', 'Recall'),
            ]
        )
    
    def get_best_metric_value(self, trainer) -> float:
        return trainer.best_iou


class InstanceSegmentationMetrics(MetricsConfig):
    """Metrics for Instance Segmentation (Mask R-CNN)."""
    
    def __init__(self):
        super().__init__(
            task_name='Instance Segmentation',
            primary_metric=MetricDefinition('val_ap', 'AP (Average Precision)'),
            display_metrics=[
                MetricDefinition('val_ap', 'AP (Average Precision)'),
                MetricDefinition('val_ar', 'AR (Average Recall)'),
                MetricDefinition('val_f1', 'F1 Score'),
                MetricDefinition('val_precision', 'Precision'),
                MetricDefinition('val_recall', 'Recall'),
            ]
        )
    
    def get_best_metric_value(self, trainer) -> float:
        return trainer.best_ap

class YOLODetectionMetrics(MetricsConfig):
    """Metrics for YOLO Detection."""
    
    def __init__(self):
        super().__init__(
            task_name='YOLO Detection',
            primary_metric=MetricDefinition('mAP50-95', 'mAP50-95'),
            display_metrics=[
                MetricDefinition('mAP50-95', 'mAP50-95'),
                MetricDefinition('mAP50', 'mAP50'),
                MetricDefinition('F1', 'F1 Score'),
                MetricDefinition('Precision', 'Precision'),
                MetricDefinition('Recall', 'Recall'),
            ]
        )
    
    def get_best_metric_value(self, trainer) -> float:
        return trainer.best_metric
    
    def get_metrics_from_history(self, history: Dict, epoch: int) -> Dict[str, float]:
        """Extract YOLO metrics from history."""
        metrics = {}
        for metric_def in self.display_metrics:
            if metric_def.key in history.get('metrics', {}):
                metric_values = history['metrics'][metric_def.key]
                if metric_values and epoch < len(metric_values):
                    metrics[metric_def.label] = metric_values[epoch]
        return metrics


class YOLOSegmentationMetrics(MetricsConfig):
    """Metrics for YOLO Segmentation."""
    
    def __init__(self):
        super().__init__(
            task_name='YOLO Segmentation',
            primary_metric=MetricDefinition('mAP50-95', 'mAP50-95 (Mask)'),
            display_metrics=[
                MetricDefinition('mAP50-95', 'mAP50-95 (Mask)'),
                MetricDefinition('mAP50', 'mAP50 (Mask)'),
                MetricDefinition('mAP50-95_box', 'mAP50-95 (Box)'),
                MetricDefinition('F1', 'F1 Score'),
                MetricDefinition('Precision', 'Precision'),
                MetricDefinition('Recall', 'Recall'),
            ]
        )
    
    def get_best_metric_value(self, trainer) -> float:
        return trainer.best_metric
    
    def get_metrics_from_history(self, history: Dict, epoch: int) -> Dict[str, float]:
        """Extract YOLO metrics from history."""
        metrics = {}
        for metric_def in self.display_metrics:
            if metric_def.key in history.get('metrics', {}):
                metric_values = history['metrics'][metric_def.key]
                if metric_values and epoch < len(metric_values):
                    metrics[metric_def.label] = metric_values[epoch]
        return metrics


class YOLOOBBMetrics(MetricsConfig):
    """Metrics for YOLO OBB (Oriented Bounding Boxes)."""
    
    def __init__(self):
        super().__init__(
            task_name='YOLO OBB',
            primary_metric=MetricDefinition('mAP50-95', 'mAP50-95 (OBB)'),
            display_metrics=[
                MetricDefinition('mAP50-95', 'mAP50-95 (OBB)'),
                MetricDefinition('mAP50', 'mAP50 (OBB)'),
                MetricDefinition('F1', 'F1 Score'),
                MetricDefinition('Precision', 'Precision'),
                MetricDefinition('Recall', 'Recall'),
            ]
        )
    
    def get_best_metric_value(self, trainer) -> float:
        return trainer.best_metric
    
    def get_metrics_from_history(self, history: Dict, epoch: int) -> Dict[str, float]:
        """Extract YOLO metrics from history."""
        metrics = {}
        for metric_def in self.display_metrics:
            if metric_def.key in history.get('metrics', {}):
                metric_values = history['metrics'][metric_def.key]
                if metric_values and epoch < len(metric_values):
                    metrics[metric_def.label] = metric_values[epoch]
        return metrics

def get_metrics_config(task: str) -> MetricsConfig:
    """Get metrics configuration for a task."""
    configs = {
        'semantic_segmentation': SemanticSegmentationMetrics(),
        'instance_segmentation': InstanceSegmentationMetrics(),
        'yolo_detection': YOLODetectionMetrics(),
        'yolo_segmentation': YOLOSegmentationMetrics(),
        'yolo_obb': YOLOOBBMetrics(),
    }
    
    if task not in configs:
        raise ValueError(f"Unsupported task: {task}")
    
    return configs[task]