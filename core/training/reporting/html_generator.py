"""
HTML Report Generator for QModel Trainer
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
import numpy as np

from .templates import get_html_template
from .metrics_config import get_metrics_config
from core.training.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
    plot_confusion_matrix,
    generate_qualitative_examples
)


class HTMLReportGenerator:
    """Generator for HTML training reports."""
    
    def __init__(self, output_dir: str, task: str = 'semantic_segmentation'):
        self.output_dir = Path(output_dir)
        self.task = task
        self.metrics_config = get_metrics_config(task)
        
        self.report_dir = self.output_dir / 'report'
        self.assets_dir = self.report_dir / 'report_assets'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        config: Dict[str, Any],
        history: Dict[str, List[float]],
        dataset_info: Dict[str, Any],
        trainer: Any,
        val_loader: Optional[Any] = None
    ) -> str:
        """Generate complete HTML report."""
        
        best_metric = self.metrics_config.get_best_metric_value(trainer)
        best_epoch = trainer.best_epoch
        
        qualitative_examples = {}
        confusion_matrix = None

        if trainer is not None and val_loader is not None:
            try:
                
                class_colors = self._extract_class_colors(config)
                print(f"Class colors: {class_colors}")
                
                qualitative_examples = generate_qualitative_examples(
                    model=trainer.model,
                    val_dataset=val_loader.dataset,
                    device=trainer.device,
                    class_names=config.get('class_names', []),
                    class_colors=class_colors,
                    assets_dir=self.assets_dir,
                    task=self.task
                )
                
                if self.task == 'semantic_segmentation':
                    confusion_matrix = qualitative_examples.get('confusion_matrix', None)
                    
            except Exception as e:
                print("\n" + "="*80)
                print("❌ ERROR: Qualitative examples generation FAILED!")
                print("="*80)
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                print("\nFull traceback:")
                import traceback
                print(traceback.format_exc())
                print("="*80 + "\n")
        
        self._generate_plots(history, confusion_matrix, config.get('class_names', []))
        
        html_content = get_html_template()
        html_content = self._fill_summary_section(html_content, config, history, best_metric, best_epoch)
        html_content = self._fill_dataset_section(html_content, dataset_info, config)
        html_content = self._fill_lr_section(html_content, config)
        html_content = self._fill_training_curves(html_content)
        html_content = self._fill_qualitative_section(html_content, qualitative_examples)
        html_content = self._fill_confusion_matrix_section(html_content, confusion_matrix)
        html_content = self._fill_metrics_section(html_content, history, best_epoch)
        html_content = self._fill_technical_info(html_content, trainer, config)
        html_content = self._fill_metadata(html_content, config)
        
        report_path = self.report_dir / 'training_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_plots(
        self, 
        history: Dict[str, List[float]],
        confusion_matrix: Optional[np.ndarray],
        class_names: List[str]
    ):
        """Generate all training plots."""
        plot_loss_curves(
            train_losses=history['train_loss'],
            val_losses=history['val_loss'],
            output_path=str(self.assets_dir / 'loss_curves.png')
        )
        
        metrics_dict = {}
        for metric_def in self.metrics_config.display_metrics:
            if metric_def.key in history and history[metric_def.key]:
                metrics_dict[metric_def.label] = history[metric_def.key]
        
        plot_metrics_curves(
            metrics=metrics_dict,
            output_path=str(self.assets_dir / 'metrics_curves.png')
        )
        
        # Confusion matrix only for semantic segmentation
        if self.task == 'semantic_segmentation':
            if confusion_matrix is not None and class_names is not None:
                if confusion_matrix.size > 0 and confusion_matrix.sum() > 0:
                    plot_confusion_matrix(
                        cm=confusion_matrix,
                        class_names=class_names,
                        output_path=str(self.assets_dir / 'confusion_matrix.png'),
                        normalize=True
                    )
                else:
                    print("⚠️ Confusion matrix is empty, skipping plot")
    
    def _fill_summary_section(
        self,
        html: str,
        config: Dict[str, Any],
        history: Dict[str, List[float]],
        best_metric: float,
        best_epoch: int
    ) -> str:
        """Fill executive summary section."""
        model_name = config.get('model_name', 'Trained Model')
        architecture = f"{config['architecture']} + {config['backbone']}"
        
        total_images = config.get('total_images', config.get('num_train_images', 'N/A'))
        dataset_size_str = str(total_images) if total_images != 'N/A' else 'N/A'
        
        total_epochs = len(history['train_loss'])
        
        if 'total_train_time' in config:
            total_seconds = config['total_train_time']
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            if hours > 0:
                train_time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                train_time_str = f"{minutes}m {seconds}s"
            else:
                train_time_str = f"{seconds}s"
        else:
            train_time_str = f"~{total_epochs} min (estimated)"
        
        # Quality assessment
        if best_metric >= 0.85:
            quality = "Excellent performance."
        elif best_metric >= 0.70:
            quality = "Good performance, consider fine-tuning for improvement."
        elif best_metric >= 0.50:
            quality = "Moderate performance, additional training or data may help."
        else:
            quality = "Low performance, review dataset quality and model configuration."

        # Summary text adapted to task
        if self.task == 'instance_segmentation':
            metric_name = "AP"
        else:
            metric_name = "IoU"

        summary_text = (
            f"The {architecture} model was trained for {total_epochs} epochs on {dataset_size_str} images. "
            f"Best validation {metric_name} of <strong>{best_metric:.4f}</strong> was achieved at epoch {best_epoch + 1}. "
            f"{quality}"
        )

        html = html.replace('{{MODEL_NAME}}', model_name)
        html = html.replace('{{ARCHITECTURE}}', architecture)
        html = html.replace('{{DATASET_SIZE}}', dataset_size_str)
        html = html.replace('{{EPOCHS}}', str(total_epochs))
        html = html.replace('{{LEARNING_RATE}}', f"{config['learning_rate']:.2e}")
        html = html.replace('{{BEST_EPOCH}}', str(best_epoch + 1))
        html = html.replace('{{TRAIN_TIME}}', train_time_str)
        html = html.replace('{{SUMMARY_TEXT}}', summary_text)

        # Metric label in sidebar
        if self.task == 'instance_segmentation':
            metric_label = 'Best AP'
        else:
            metric_label = 'Best mIoU'

        html = html.replace('{{BEST_METRIC}}', f"{best_metric:.4f}")
        html = html.replace('{{BEST_METRIC_LABEL}}', metric_label)
        html = html.replace('{{QUALITY_ASSESSMENT}}', quality) 
        
        return html
    
    def _fill_dataset_section(
        self,
        html: str,
        dataset_info: Dict[str, Any],
        config: Dict[str, Any]
    ) -> str:
        """Fill dataset details section."""
        class_distribution = config.get('class_distribution', {})
        class_names = config.get('class_names', dataset_info.get('class_names', []))

        # Determine unit label based on task
        if self.task == 'instance_segmentation':
            unit_label = 'instances'
        else:
            unit_label = 'pixels'

        if class_names:
            if class_distribution:
                table_html = f'<table><thead><tr><th>ID</th><th>Class Name</th><th>Count ({unit_label})</th><th>Percentage</th></tr></thead><tbody>'
            else:
                table_html = '<table><thead><tr><th>ID</th><th>Class Name</th></tr></thead><tbody>'
            
            for idx, class_name in enumerate(class_names):
                if class_distribution and idx in class_distribution:
                    dist = class_distribution[idx]
                    count_str = f"{dist['count']:,}"
                    
                    percentage = dist['percentage']
                    bar_width = int(percentage)
                    bar_color = '#16a34a' if percentage > 5 else '#ef4444'
                    
                    percentage_cell = f'''
                        <div style="display:flex;align-items:center;gap:8px;">
                            <div style="width:100px;height:12px;background:#f1f5f9;border-radius:4px;overflow:hidden;">
                                <div style="width:{bar_width}%;height:100%;background:{bar_color};"></div>
                            </div>
                            <span style="font-weight:bold;">{percentage:.2f}%</span>
                        </div>
                    '''
                    
                    table_html += f'<tr><td><strong>{idx}</strong></td><td>{class_name}</td><td>{count_str}</td><td>{percentage_cell}</td></tr>'
                else:
                    table_html += f'<tr><td><strong>{idx}</strong></td><td>{class_name}</td><td colspan="2">N/A</td></tr>'
            
            table_html += '</tbody></table>'
            
            if class_distribution:
                # For instance segmentation, exclude background (class 0) from balance calculation
                if self.task == 'instance_segmentation':
                    relevant_classes = {k: v for k, v in class_distribution.items() if k != 0}
                else:
                    relevant_classes = class_distribution

                # DEBUG
                print(f"DEBUG relevant_classes: {relevant_classes}")


                if relevant_classes:
                    percentages = [d['percentage'] for d in relevant_classes.values()]
                    # DEBUG
                    print(f"DEBUG percentages: {percentages}")
                    min_pct = min(percentages)
                    max_pct = max(percentages)
                    imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
                    # DEBUG
                    print(f"DEBUG imbalance_ratio: {imbalance_ratio}")
                else:
                    imbalance_ratio = 1.0
                
                if imbalance_ratio > 50:
                    table_html += '''
                        <div class="notice" style="margin-top:10px;">
                            ⚠️ <strong>Class imbalance detected</strong>: Some classes are underrepresented. 
                            Consider using weighted loss or data augmentation.
                        </div>
                    '''
                elif imbalance_ratio > 10:
                    table_html += '''
                        <div class="notice" style="margin-top:10px;background:#fef3c7;color:#78350f;">
                            ℹ️ <strong>Moderate class imbalance</strong>: Dataset is reasonably balanced but could be improved.
                        </div>
                    '''
                else:
                    table_html += '''
                        <div style="margin-top:10px;padding:10px;background:#dcfce7;color:#166534;border-radius:8px;font-size:13px;">
                            ✅ <strong>Well-balanced dataset</strong>: Class distribution is good.
                        </div>
                    '''
        else:
            table_html = '<p class="small" style="color:#999;">No class information available</p>'
        
        html = html.replace('{{DATASET_TABLE}}', table_html)
        
        tile_size = config.get('image_size', 512)
        html = html.replace('{{TILE_SIZE}}', f"{tile_size}×{tile_size}")
        
        val_split_percent = int(config.get('val_split', 0.2) * 100)
        html = html.replace('{{VAL_SPLIT}}', f"{val_split_percent}%")
        
        batch_size = config.get('batch_size', 'N/A')
        html = html.replace('{{BATCH_SIZE}}', str(batch_size))
        
        return html
    
    def _fill_lr_section(self, html: str, config: Dict[str, Any]) -> str:
        """Fill LR Finder section if used."""
        if config.get('auto_lr', False):
            lr_value = f"{config['learning_rate']:.2e}"
            lr_section = f'''
        <div class="card" style="margin-top: 16px">
            <div class="section-title">Optimal Learning Rate</div>
            <div class="chart">
            <img src="report_assets/lr_finder.png" alt="LR Finder" onclick="openLightbox(this)" />
            </div>
            <div class="click-hint">🔍 Click to enlarge</div>
            <div class="explain">
            Automatic learning rate detection using LR Range Test. 
            Suggested LR: <strong>{lr_value}</strong>
            </div>
        </div>'''
        else:
            lr_section = ''
        
        html = html.replace('{{LR_FINDER_SECTION}}', lr_section)
        return html
    
    def _fill_training_curves(self, html: str) -> str:
        """Training curves are already linked via img tag."""
        curves_section = '''
            <div class="card" style="margin-top: 16px">
                <div class="section-title">Training Curves</div>
                <div class="explain" style="margin-bottom:10px;">
                Loss and metrics evolution during training. 
                Best model was saved at epoch {{BEST_EPOCH}}. 
                <strong>Click on images to enlarge.</strong>
                </div>
                <div class="two-col">
                <div>
                    <div class="small">Loss (train / val)</div>
                    <div class="chart">
                    <img src="report_assets/loss_curves.png" alt="Loss curves" onclick="openLightbox(this)" />
                    </div>
                    <div class="click-hint">🔍 Click to enlarge</div>
                </div>
                <div>
                    <div class="small">Metrics (IoU / F1 / Precision / Recall)</div>
                    <div class="chart">
                    <img src="report_assets/metrics_curves.png" alt="Metrics curves" onclick="openLightbox(this)" />
                    </div>
                    <div class="click-hint">🔍 Click to enlarge</div>
                </div>
                </div>
            </div>'''
        
        html = html.replace('{{TRAINING_CURVES_SECTION}}', curves_section)
        return html
    
    def _fill_qualitative_section(
        self,
        html: str,
        qualitative_examples: Dict[str, List[Dict[str, str]]]
    ) -> str:
        """Fill qualitative examples section."""
        if not qualitative_examples or all(not v for v in qualitative_examples.values()):
            html = html.replace('{{QUALITATIVE_EXAMPLES_SECTION}}', '')
            return html
        
        section_html = '''
          <div class="card" style="margin-top: 16px">
            <div class="section-title">Qualitative Examples</div>
            <div class="explain" style="margin-bottom:10px;">
              Visual comparison of predictions vs ground truth. 
              Use the selector to view examples by performance level.
            </div>
            
            <div style="margin-bottom:16px;">
              <label for="iou-selector" style="font-weight:600;font-size:14px;color:#374151;margin-bottom:6px;display:block;">
                Select IoU Range:
              </label>
              <select id="iou-selector" class="iou-dropdown" onchange="switchExamples(this.value)">
                <option value="medium" selected>Medium IoU (Around average)</option>
                <option value="high">High IoU (≥ 0.90) - Best predictions</option>
                <option value="low">Low IoU (≤ 0.60) - Challenging cases</option>
              </select>
            </div>
            
            <div id="examples-container">
              {{EXAMPLES_CONTENT}}
            </div>
          </div>'''
        
        examples_content = ''
        
        for category in ['high', 'medium', 'low']:
            examples = qualitative_examples.get(category, [])
            is_fallback = qualitative_examples.get(f'{category}_is_fallback', False)
            
            if not examples or (is_fallback and category != 'medium'):
                if category == 'high':
                    message = "No examples with IoU ≥ 0.90 found in validation set."
                elif category == 'low':
                    message = "No examples with IoU ≤ 0.60 found in validation set."
                else:
                    message = "No examples available in this IoU range."
                
                category_html = f'''
              <div class="examples-grid" data-category="{category}" style="display:none;">
                <p style="text-align:center;color:#999;padding:20px;">
                  {message}
                </p>
              </div>'''
            else:
                category_html = f'''
              <div class="examples-grid" data-category="{category}" style="display:none;">'''
                
                for i, example in enumerate(examples):
                    score_value = example['score']
                    # Label depends on task
                    score_label = 'Score' if self.task == 'instance_segmentation' else 'IoU'
                    category_html += f'''
                                <div class="example-triplet">
                                <div class="triplet-header">Example {i+1} — {score_label}: {score_value}</div>
                  <div class="triplet-row">
                    <div class="triplet-col">
                      <div class="triplet-label">Input Image</div>
                      <img src="{example['input']}" alt="Input" onclick="openLightbox(this)" />
                    </div>
                    <div class="triplet-col">
                      <div class="triplet-label">Ground Truth</div>
                      <img src="{example['gt']}" alt="Ground Truth" onclick="openLightbox(this)" />
                    </div>
                    <div class="triplet-col">
                      <div class="triplet-label">Prediction</div>
                      <img src="{example['pred']}" alt="Prediction" onclick="openLightbox(this)" />
                    </div>
                  </div>
                </div>'''
                
                category_html += '''
              </div>'''
            
            examples_content += category_html
        
        section_html = section_html.replace('{{EXAMPLES_CONTENT}}', examples_content)
        html = html.replace('{{QUALITATIVE_EXAMPLES_SECTION}}', section_html)
        
        return html
    
    def _fill_confusion_matrix_section(
        self,
        html: str,
        confusion_matrix: Optional[np.ndarray]
    ) -> str:
        """Fill confusion matrix section."""
        if confusion_matrix is None:
            html = html.replace('{{CONFUSION_MATRIX_SECTION}}', '')
            return html
        
        non_empty_rows = (confusion_matrix.sum(axis=1) > 0).sum()
        if non_empty_rows < 2:
            html = html.replace('{{CONFUSION_MATRIX_SECTION}}', '')
            return html
        
        section_html = '''
          <div class="card" style="margin-top: 16px">
            <div class="section-title">Confusion Matrix</div>
            <div class="explain" style="margin-bottom:10px;">
              Pixel-wise confusion matrix showing classification performance per class.
              Rows represent true classes, columns represent predicted classes.
              <strong>Click to enlarge.</strong>
            </div>
            <div class="chart">
              <img src="report_assets/confusion_matrix.png" alt="Confusion Matrix" onclick="openLightbox(this)" />
            </div>
            <div class="click-hint">🔍 Click to enlarge</div>'''
        
        missing_classes_indices = []
        for i in range(len(confusion_matrix)):
            if confusion_matrix[i].sum() == 0:
                missing_classes_indices.append(i)
        
        if missing_classes_indices:
            section_html += '''
            <div class="notice" style="margin-top:10px;background:#fef3c7;color:#78350f;">
              ℹ️ <strong>Note:</strong> Some classes have no samples in the validation set 
              (rows with all zeros). This is expected with small datasets or class imbalance.
            </div>'''
        
        section_html += '''
          </div>'''
        
        html = html.replace('{{CONFUSION_MATRIX_SECTION}}', section_html)
        
        return html
    
    def _fill_metrics_section(
        self,
        html: str,
        history: Dict[str, List[float]],
        best_epoch: int
    ) -> str:
        """Fill validation metrics table."""
        metrics_html = ''
        
        metrics_at_best_epoch = self.metrics_config.get_metrics_from_history(history, best_epoch)
        
        for metric_label, value in metrics_at_best_epoch.items():
            metric_def = next(
                (m for m in self.metrics_config.display_metrics if m.label == metric_label),
                None
            )
            
            if metric_def is None:
                continue
            
            values = history[metric_def.key]
            peak_val = max(values)
            peak_epoch = values.index(peak_val)
            
            if peak_epoch != best_epoch:
                note = f'<span style="color:#999;font-size:11px;">(peaked at {peak_val:.4f} on epoch {peak_epoch + 1})</span>'
            else:
                note = '<span style="color:#16a34a;font-size:11px;">✓ Peak value</span>'
            
            metrics_html += f'''
                <tr>
                  <td><strong>{metric_label}</strong></td>
                  <td style="font-size:16px;font-weight:bold;">{value:.4f}</td>
                  <td>{note}</td>
                </tr>'''
        
        html = html.replace('{{METRICS_TABLE}}', metrics_html)
        return html
    
    def _fill_technical_info(
        self,
        html: str,
        trainer: Any,
        config: Dict[str, Any]
    ) -> str:
        """Fill technical information section."""
        tech_html = ''
        
        if config.get('device') == 'cuda' and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            tech_html += f'<div class="kv"><b>{gpu_name}</b><span class="small">GPU</span></div>'
        else:
            device = config.get('device', 'cpu').upper()
            tech_html += f'<div class="kv"><b>{device}</b><span class="small">Device</span></div>'
        
        pytorch_version = torch.__version__
        tech_html += f'<div class="kv"><b>{pytorch_version}</b><span class="small">PyTorch</span></div>'
        
        if hasattr(trainer, 'model'):
            total_params = sum(p.numel() for p in trainer.model.parameters())
            tech_html += f'<div class="kv"><b>{total_params:,}</b><span class="small">Parameters</span></div>'
        
        html = html.replace('{{TECHNICAL_INFO}}', tech_html)
        
        optimizer_name = config.get('optimizer_name', 'Adam')
        html = html.replace('{{OPTIMIZER}}', optimizer_name)
        
        scheduler_name = config.get('scheduler_name', 'ReduceLROnPlateau')
        html = html.replace('{{SCHEDULER}}', scheduler_name)
        
        early_stopping = config.get('early_stopping', False)
        patience = config.get('patience', 'N/A')
        es_text = f"Yes (patience={patience})" if early_stopping else "No"
        html = html.replace('{{EARLY_STOPPING}}', es_text)
        
        pretrained = "Yes (ImageNet)" if config.get('pretrained', False) else "No"
        html = html.replace('{{PRETRAINED}}', pretrained)
        
        return html
    
    def _fill_metadata(self, html: str, config: Dict[str, Any]) -> str:
        """Fill metadata."""
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plugin_version = config.get('plugin_version', '1.0.0')
        
        html = html.replace('{{DATE}}', date_str)
        html = html.replace('{{PLUGIN_VERSION}}', plugin_version)
        
        return html
    
    def _extract_class_colors(self, config: Dict[str, Any]) -> Dict[int, tuple]:
        """Extract class colors from config metadata."""
        class_colors = {}
        dataset_classes = config.get('dataset_classes', [])
        
        for class_info in dataset_classes:
            if isinstance(class_info, dict) and 'color' in class_info:
                class_id = class_info.get('id', None)
                if class_id is None:
                    continue
                
                color_str = class_info['color']
                if color_str.startswith('#'):
                    color_str = color_str[1:]
                
                from core.training.utils.colors import hex_to_rgb
                
                try:
                    class_colors[class_id] = hex_to_rgb(color_str)
                except (ValueError, IndexError):
                    class_colors[class_id] = self._get_default_color_rgb(class_id)
        
        num_classes = config.get('num_classes', len(dataset_classes))
        for i in range(num_classes):
            if i not in class_colors:
                class_colors[i] = self._get_default_color_rgb(i)
        
        return class_colors
    
    def _get_default_color_rgb(self, index: int) -> tuple:
        """Get default RGB color for a class by index."""
        from core.training.utils.colors import get_default_color_rgb
        return get_default_color_rgb(index)


def generate_training_report(
    output_dir: str,
    config: Dict[str, Any],
    trainer: Any,
    dataset_info: Dict[str, Any],
    task: str = 'semantic_segmentation',
    val_loader: Any = None
) -> str:
    """Convenience function to generate training report."""
    generator = HTMLReportGenerator(output_dir=output_dir, task=task)
    
    report_path = generator.generate(
        config=config,
        history=trainer.history,
        dataset_info=dataset_info,
        trainer=trainer,
        val_loader=val_loader
    )
    
    return report_path